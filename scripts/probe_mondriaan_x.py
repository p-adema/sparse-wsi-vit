
"""Train dense ViT-5 on Mondriaan-X.

Run from the repository root, for example:

    python scripts/probe_mondriaan_x.py --num-tiles 1 --steps 2000
    python scripts/probe_mondriaan_x.py --num-tiles 4 --smoke-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan_x import (
    make_data_x,
    check_data_x,
)
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


def balanced_batch_indices(labels, batch_size, generator):
    """Sample a balanced batch with replacement."""
    positive = torch.where(labels == 1)[0]
    negative = torch.where(labels == 0)[0]

    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Dataset must contain both classes.")

    half = batch_size // 2

    pos_indices = positive[
        torch.randint(len(positive), (half,), generator=generator)
    ]
    neg_indices = negative[
        torch.randint(len(negative), (batch_size - half,), generator=generator)
    ]

    indices = torch.cat([pos_indices, neg_indices])
    permutation = torch.randperm(batch_size, generator=generator)

    return indices[permutation]


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--num-tiles",
        type=int,
        default=1,
        help="Number of 32x32 tiles: 1, 4, 16, 64, 256, ...",
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mondriaan_x"),
    )

    args = parser.parse_args()

    # Validate arguments.
    for key in ("train_size", "eval_size", "batch_size"):
        value = getattr(args, key)
        if value < 8 or value % 8 != 0:
            parser.error(f"{key} must be a positive multiple of 8")

    if args.num_tiles < 1:
        parser.error("num-tiles must be positive")

    side = int(args.num_tiles**0.5)
    if side * side != args.num_tiles:
        parser.error("num-tiles must be a perfect square")

    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")

    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("eval-batch-size must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    # Reproducibility.
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generate datasets.
    train = make_data_x(
        args.train_size,
        args.seed,
        args.num_tiles,
    )
    test = make_data_x(
        args.eval_size,
        args.seed + 1,
        args.num_tiles,
    )

    checks = {
        "train": check_data_x(train, args.num_tiles),
        "test": check_data_x(test, args.num_tiles),
    }

    print(json.dumps(checks, indent=2), flush=True)

    if args.smoke_test:
        print("\nSmoke test passed.", flush=True)
        return

    # Model.
    device = torch.device(args.device)

    model = VitDensePreEmbedded(
        in_features=48,
        out_features=2,
        downproj=256,
        depth=3,
        num_heads=4,
        rope_dynamic_high=32,
        init_scale=1e-4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    sampler = torch.Generator().manual_seed(args.seed + 1000)
    eval_batch = args.eval_batch_size or args.batch_size

    # Output.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_tokens = train["x"].shape[1]
    output = (
        args.output_dir
        / f"vit5_tokens{num_tokens}_seed{args.seed}.json"
    )

    report = {
        "settings": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "random_colors": False,
            "patterns": 6,
            "min_sep": 8,
            "tokenization": "object",
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
        },
        "checks": checks,
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "tokens_per_sample": num_tokens,
        "history": [],
    }

    # Keep the complete dataset on CPU.
    # Transfer only minibatches to the selected device.
    def model_forward(data, indices):
        x = data["x"][indices].to(device)
        coords = data["coords"][indices].to(device)

        return model(x, coords)["logits"]

    @torch.inference_mode()
    def model_evaluate(data):
        model.eval()

        predictions = []
        total_loss = 0.0

        for start in range(0, len(data["y"]), eval_batch):
            end = min(start + eval_batch, len(data["y"]))
            indices = slice(start, end)

            logits = model_forward(data, indices)
            labels = data["y"][indices].to(device)

            total_loss += F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()

            predictions.append(logits.argmax(dim=1).cpu())

        predictions = torch.cat(predictions)
        correct = predictions == data["y"]

        metrics = {
            "accuracy": correct.float().mean().item(),
            "loss": total_loss / len(data["y"]),
        }

        # Accuracy for each Mondriaan case.
        if "cases" in data:
            case_names = list(data["cases"])
            per_case = {}

            for case in sorted(set(case_names)):
                mask = torch.tensor(
                    [name == case for name in case_names],
                    dtype=torch.bool,
                )

                if mask.any():
                    per_case[str(case)] = {
                        "accuracy": correct[mask].float().mean().item(),
                        "count": int(mask.sum().item()),
                    }

            metrics["per_case"] = per_case

        return metrics

    # Training.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()

    print(
        f"\nTraining ViT-5 | "
        f"tokens={num_tokens} | "
        f"steps={args.steps} | "
        f"device={device}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        model.train()

        indices = balanced_batch_indices(
            train["y"],
            args.batch_size,
            sampler,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model_forward(train, indices)
        labels = train["y"][indices].to(device)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        should_evaluate = (
            step == 1
            or step % args.eval_every == 0
            or step == args.steps
        )

        if should_evaluate:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            metrics = model_evaluate(test)
            elapsed = time.perf_counter() - start_time

            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "test_accuracy": metrics["accuracy"],
                "test_loss": metrics["loss"],
                "test_per_case": metrics.get("per_case", {}),
                "elapsed_seconds": elapsed,
            }

            if device.type == "cuda":
                record["peak_gpu_memory_mb"] = (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                )

            report["history"].append(record)

            print(
                f"Step {step}/{args.steps} | "
                f"train_loss={loss.item():.4f} | "
                f"test_acc={metrics['accuracy']:.4f} | "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            # Save progress after every evaluation.
            with output.open("w") as file:
                json.dump(report, file, indent=2)

    # Final summary.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report["total_training_seconds"] = (
        time.perf_counter() - start_time
    )

    if device.type == "cuda":
        report["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )
    else:
        report["peak_gpu_memory_mb"] = None

    report["final_accuracy"] = (
        report["history"][-1]["test_accuracy"]
    )

    with output.open("w") as file:
        json.dump(report, file, indent=2)

    print("\nTraining complete.", flush=True)
    print(
        f"Final accuracy: {report['final_accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Total time: {report['total_training_seconds']:.1f}s",
        flush=True,
    )
    print(f"Results saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
