#!/usr/bin/env python3
"""Train/evaluate dense ViT-5 + pixel-level ConvStem across Mondriaan canvas sizes.

Four 4x4 objects, 12 fixed patterns, freely positioned, one RGB token per pixel.
Records classification accuracy (overall/per case), loss, training speed, and CUDA
peak memory. Each canvas size trains a fresh model from scratch.
"""

import argparse
from collections import Counter
import gc
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator
from sparse_wsi_vit.models.conv_stem import ConvStem
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig


CASES = {
    (1, 1, 1, 1): "all_distinct",
    (1, 1, 2): "one_pair",
    (2, 2): "two_pairs",
    (1, 3): "three_matching",
    (4,): "four_matching",
}

NEGATIVE = [
    "all_distinct",
    "two_pairs",
    "three_matching",
    "four_matching",
]


def make_dataset(count, side, seed, min_sep):
    """Store only the four objects, rather than full dense images."""
    if count % 8:
        raise ValueError("Dataset sizes must be divisible by 8")

    random.seed(seed)
    np.random.seed(seed)

    quotas = {
        "one_pair": count // 2,
        **{case: count // 8 for case in NEGATIVE},
    }

    seen = Counter()
    samples = []
    attempts = 0

    while len(samples) < count:
        img, label, types, positions = MondriaanGenerator.generate_single(
            image_size=side,
            object_type="patterned",
            target_label=attempts % 2,
            min_sep=min_sep,
            random_colors=False,
            free_positions=True,
            extra_patterns=True,
        )
        attempts += 1

        case = CASES[tuple(sorted(Counter(types).values()))]

        if seen[case] >= quotas[case]:
            continue

        patches = np.stack([
            img[y:y + 4, x:x + 4]
            for y, x in positions
        ]).copy()

        samples.append((positions, patches, int(label), case))
        seen[case] += 1

    random.shuffle(samples)
    return samples


def make_coords(side):
    yy, xx = np.mgrid[:side, :side]

    return torch.from_numpy(
        np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    )


def collate(samples, indices, side, coords, device):
    b = len(indices)

    # Same background sentinel as extract_mondriaan.py.
    images = np.full(
        (b, side, side, 3),
        -1.0,
        dtype=np.float32,
    )

    labels = []
    cases = []

    for j, index in enumerate(indices):
        positions, patches, label, case = samples[int(index)]

        for (y, x), patch in zip(positions, patches):
            images[j, y:y + 4, x:x + 4] = patch

        labels.append(label)
        cases.append(case)

    features = torch.from_numpy(
        images.reshape(b, side * side, 3)
    ).to(device)

    batch_coords = (
        coords.unsqueeze(0)
        .expand(b, -1, -1)
        .to(device)
    )

    targets = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return features, batch_coords, targets, cases


def build_model(side):
    vit = LazyConfig(VitDensePreEmbedded)(
        depth=3,
        num_heads=4,
        rope_dynamic_high=side,
    )

    return ConvStem(
        net=vit,
        in_features=3,
        out_features=2,
        dim=256,
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, samples, side, coords, batch_size, device, bf16):
    model.eval()

    correct = Counter()
    totals = Counter()
    loss_sum = 0.0

    for start in range(0, len(samples), batch_size):
        inds = range(
            start,
            min(start + batch_size, len(samples)),
        )

        x, xy, y, cases = collate(
            samples, inds, side, coords, device
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            logits = model(x, xy)["logits"]
            loss = F.cross_entropy(
                logits.float(),
                y,
                reduction="sum",
            )

        predictions = logits.argmax(dim=-1)

        for j, case in enumerate(cases):
            totals[case] += 1
            correct[case] += int(
                (predictions[j] == y[j]).item()
            )

        loss_sum += loss.item()

        del x, xy, y, logits

    n = sum(totals.values())

    return {
        "accuracy": sum(correct.values()) / n,
        "loss": loss_sum / n,
        "per_case": {
            case: {
                "accuracy": correct[case] / totals[case],
                "count": totals[case],
            }
            for case in sorted(totals)
        },
    }


def run_size(args, side, device, report, save):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    record = {
        "image_size": side,
        "tokens": side * side,
        "status": "running",
        "history": [],
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
    }

    report["results"].append(record)
    save()

    model = optimizer = train = test = coords = None

    try:
        train = make_dataset(
            args.train_size,
            side,
            args.seed,
            args.min_sep,
        )

        test = make_dataset(
            args.eval_size,
            side,
            args.seed + 1,
            args.min_sep,
        )

        coords = make_coords(side)

        record["case_counts"] = dict(
            Counter(s[3] for s in test)
        )

        model = build_model(side).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

        record["parameters"] = sum(
            p.numel() for p in model.parameters()
        )

        record["gpu"] = (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pos = [
            i for i, sample in enumerate(train)
            if sample[2] == 1
        ]
        neg = [
            i for i, sample in enumerate(train)
            if sample[2] == 0
        ]

        sampler = np.random.default_rng(args.seed + 1000)

        durations = []
        start = time.perf_counter()

        bf16 = (
            args.precision == "bf16"
            and device.type == "cuda"
        )

        for step in range(1, args.steps + 1):
            model.train()

            # Balanced batches where possible.
            # Batch size 1 alternates between classes.
            npos = args.batch_size // 2

            if args.batch_size % 2:
                npos += step % 2

            indices = np.concatenate((
                sampler.choice(
                    pos, npos, replace=True
                ),
                sampler.choice(
                    neg,
                    args.batch_size - npos,
                    replace=True,
                ),
            ))

            sampler.shuffle(indices)

            x, xy, y, _ = collate(
                train,
                indices,
                side,
                coords,
                device,
            )

            sync(device)
            t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16,
            ):
                logits = model(x, xy)["logits"]
                loss = F.cross_entropy(
                    logits.float(), y
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            sync(device)
            duration = time.perf_counter() - t0

            if step > args.warmup_steps:
                durations.append(duration)

            last_loss = float(loss.item())

            del x, xy, y, logits, loss

            if step == 1 or step % args.log_every == 0:
                print(
                    f"{side}x{side} "
                    f"step {step}/{args.steps}: "
                    f"loss={last_loss:.4f}, "
                    f"step_time={duration:.3f}s",
                    flush=True,
                )

            should_evaluate = (
                step == 1
                or step % args.eval_every == 0
                or step == args.steps
            )

            if should_evaluate:
                metrics = evaluate(
                    model,
                    test,
                    side,
                    coords,
                    args.eval_batch_size,
                    device,
                    bf16,
                )

                sync(device)

                entry = {
                    "step": step,
                    "train_loss": last_loss,
                    **metrics,
                    "elapsed_seconds": (
                        time.perf_counter() - start
                    ),
                    "mean_train_step_seconds": (
                        float(np.mean(durations))
                        if durations
                        else None
                    ),
                }

                if device.type == "cuda":
                    entry["peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated(device)
                        / 1024**3
                    )
                    entry["peak_reserved_gib"] = (
                        torch.cuda.max_memory_reserved(device)
                        / 1024**3
                    )

                record["history"].append(entry)
                save()

                print(
                    f"  eval step {step}: "
                    f"accuracy={metrics['accuracy']:.4f}, "
                    f"loss={metrics['loss']:.4f}",
                    flush=True,
                )

        record["status"] = "ok"
        record["final_accuracy"] = (
            record["history"][-1]["accuracy"]
        )
        record["final_test_loss"] = (
            record["history"][-1]["loss"]
        )
        record["mean_step_seconds"] = (
            float(np.mean(durations))
            if durations
            else None
        )
        record["median_step_seconds"] = (
            float(np.median(durations))
            if durations
            else None
        )
        record["total_seconds"] = (
            time.perf_counter() - start
        )

    except Exception as exc:
        msg = str(exc)

        record["status"] = (
            "oom"
            if (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in msg.lower()
            )
            else "error"
        )

        record["error"] = traceback.format_exc()[-5000:]
        print(record["error"], flush=True)

    finally:
        if device.type == "cuda":
            record["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device)
                / 1024**3
            )
            record["peak_reserved_gib"] = (
                torch.cuda.max_memory_reserved(device)
                / 1024**3
            )

        save()

        del optimizer, model, train, test, coords
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return record["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--image-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128],
    )
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--min-sep", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    p.add_argument("--threads", type=int, default=4)

    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/mondriaan_conv_scaling/dense_vit5.json"
        ),
    )

    args = p.parse_args()

    if any(
        s < 32 or s % 4
        for s in args.image_sizes
    ):
        p.error(
            "image sizes must be >=32 and divisible by 4"
        )

    if (
        args.train_size < 8
        or args.train_size % 8
        or args.eval_size < 8
        or args.eval_size % 8
    ):
        p.error(
            "train-size and eval-size must be multiples of 8"
        )

    if (
        min(
            args.steps,
            args.batch_size,
            args.eval_batch_size,
            args.eval_every,
            args.log_every,
            args.threads,
        ) < 1
        or args.warmup_steps < 0
    ):
        p.error("invalid nonpositive argument")

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        p.error("CUDA unavailable")

    torch.set_num_threads(args.threads)

    device = torch.device(args.device)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "benchmark": (
            "mondriaan_conv_dense_vit5_accuracy_and_scaling"
        ),
        "settings": {
            **vars(args),
            "output": str(args.output),
        },
        "data": {
            "free_positions": True,
            "extra_patterns": True,
            "random_colors": False,
            "pool_size": 1,
            "background_sentinel": -1.0,
            "objects": 4,
        },
        "results": [],
    }

    def save():
        temp = args.output.with_suffix(".tmp")

        with temp.open("w") as f:
            json.dump(report, f, indent=2)

        temp.replace(args.output)

    for side in args.image_sizes:
        print(
            f"\n=== {side}x{side} | "
            f"{side * side} tokens ===",
            flush=True,
        )

        status = run_size(
            args,
            side,
            device,
            report,
            save,
        )

        if status != "ok":
            print(
                f"Stopped after {side}x{side}: {status}",
                flush=True,
            )
            break

    print(
        f"Saved results to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
