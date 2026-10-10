"""Run a bounded CPU learning probe using the current Mondriaan generator.

Defaults use fixed colors, pixel tokens, and the configured 3 x 256 ViT5 Dense.
Object mode concatenates the ordered RGB pixels of each 4x4 block into a token.
Patch mode concatenates pixels in each pool_size block without averaging.
The small training set is intentionally reused to test whether it can be fitted;
an independently generated evaluation set tests whether that learning transfers.
No experiment service, GPU, HDF5 dataset, or checkpoint is required.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.extract_mondriaan import (
    BACKGROUND_SENTINEL,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


CASES = {
    (1, 1, 1, 1): "all_distinct",
    (1, 1, 2): "one_pair",
    (2, 2): "two_pairs",
    (1, 3): "three_matching",
    (4,): "four_matching",
}


def tokenize_image(image, pool_size=1, foreground_only=False, tokenization="pixel"):
    """Represent pixels or whole 4x4 blocks, retaining RGB order in object tokens."""
    height, width, _ = image.shape
    if tokenization in ("object", "patch"):
        if tokenization == "object" and pool_size != 1:
            raise ValueError("Object tokens preserve full pixels; pool_size must be 1")
        stride = 4 if tokenization == "object" else pool_size
        rows, cols = height // stride, width // stride
        features = image.reshape(rows, stride, cols, stride, 3)
        features = features.transpose(0, 2, 1, 3, 4).reshape(rows * cols, stride * stride * 3)
    else:
        stride = pool_size
        rows, cols = height // stride, width // stride
        features = image.reshape(rows, stride, cols, stride, 3)
        features = features.mean(axis=(1, 3)).reshape(rows * cols, 3)
    ys, xs = np.mgrid[:rows, :cols]
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1) * stride
    background = (features == 0).all(axis=1)
    features = features.copy()
    features[background] = BACKGROUND_SENTINEL
    keep = ~background if foreground_only else np.ones(len(features), dtype=bool)
    return features[keep], coords[keep]


def make_data(size, seed, pool_size, random_colors, foreground_only=False,
              tokenization="pixel", *, retain_images=False, interior_only=False):
    random.seed(seed)
    np.random.seed(seed)
    quotas = {case: size // (2 if case == "one_pair" else 8) for case in CASES.values()}
    counts = Counter()
    features, labels, cases, sample_coords = [], [], [], []
    type_multisets = []
    images, object_types, positions = [], [], []
    attempts = 0
    while len(labels) < size:
        image, label, types, corners = MondriaanGenerator.generate_single(
            image_size=32, min_sep=8, object_type="patterned",
            target_label=attempts % 2, random_colors=random_colors,
        )
        attempts += 1
        if interior_only and any(y > 24 or x > 24 for y, x in corners):
            continue
        case = CASES[tuple(sorted(Counter(types).values()))]
        if counts[case] == quotas[case]:
            continue
        tokens, coords = tokenize_image(image, pool_size, foreground_only, tokenization)
        features.append(tokens)
        sample_coords.append(coords)
        labels.append(label)
        cases.append(case)
        type_multisets.append(tuple(sorted(types)))
        if retain_images:
            images.append(image)
            object_types.append(types)
            positions.append(corners)
        counts[case] += 1
    data = {
        "x": torch.from_numpy(np.stack(features)),
        "y": torch.tensor(labels),
        "coords": torch.from_numpy(np.stack(sample_coords)).float(),
        "cases": cases,
        "type_multisets": type_multisets,
        "indices": {case: torch.tensor([i for i, c in enumerate(cases) if c == case])
                    for case in quotas},
    }
    if retain_images:
        data.update(images=np.stack(images), object_types=object_types, positions=positions)
    return data


def balanced_indices(data, batch_size, generator=None):
    groups = []
    for case, indices in data["indices"].items():
        count = batch_size // (2 if case == "one_pair" else 8)
        groups.append(indices[torch.randint(len(indices), (count,), generator=generator)])
    indices = torch.cat(groups)
    return indices[torch.randperm(len(indices), generator=generator)]


def forward(model, data, indices):
    return model(data["x"][indices], data["coords"][indices])["logits"]


@torch.inference_mode()
def evaluate(model, data, batch_size):
    model.eval()
    logits = torch.cat([
        forward(model, data, torch.arange(start, min(start + batch_size, len(data["y"]))))
        for start in range(0, len(data["y"]), batch_size)
    ])
    correct = logits.argmax(dim=1) == data["y"]
    metrics = {
        "loss": F.cross_entropy(logits, data["y"]).item(),
        "accuracy": correct.float().mean().item(),
        "margin_std": (logits[:, 1] - logits[:, 0]).std().item(),
        "by_case": {
            case: {"count": len(indices), "accuracy": correct[indices].float().mean().item()}
            for case, indices in data["indices"].items()
        },
    }
    if "unseen_combinations" in data:
        unseen = data["unseen_combinations"]
        metrics["unseen_combinations"] = {
            "count": unseen.sum().item(),
            "accuracy": correct[unseen].float().mean().item() if unseen.any() else None,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--random-colors", action="store_true")
    parser.add_argument("--tokenization", choices=("pixel", "object", "patch"), default="pixel",
                        help="Object flattens 4x4 pixels; patch flattens pool-size pixels; pixel averages.")
    parser.add_argument("--foreground-only", action="store_true",
                        help="Diagnostic control: omit constant background tokens.")
    parser.add_argument("--init-scale", type=float, default=1e-4)
    parser.add_argument("--rope-high", type=float, default=32,
                        help="Coordinate divisor for dynamic RoPE (configured default: 32).")
    parser.add_argument("--color-control", action="store_true",
                        help="Toy optimization control: make foreground red/cyan according to label.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("train_size", "eval_size", "batch_size"):
        if getattr(args, name) < 8 or getattr(args, name) % 8:
            parser.error(f"{name} must be a positive multiple of 8")
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")
    if args.rope_high <= 0:
        parser.error("rope-high must be positive")
    if args.tokenization == "object" and args.pool_size != 1:
        parser.error("object tokenization requires pool-size=1 to retain the full pattern")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train = make_data(args.train_size, args.seed, args.pool_size, args.random_colors,
                      args.foreground_only, args.tokenization)
    evaluation = make_data(args.eval_size, args.seed + 1, args.pool_size, args.random_colors,
                           args.foreground_only, args.tokenization)
    if args.color_control:
        # This intentionally changes the task, and is recorded as a diagnostic control.
        for data in (train, evaluation):
            foreground = ~(data["x"] == BACKGROUND_SENTINEL).all(dim=-1)
            colors = torch.tensor([[1., 0., 0.], [0., 1., 1.]])[data["y"]]
            colors = colors.repeat(1, data["x"].shape[-1] // 3)
            data["x"][foreground] = colors[:, None].expand_as(data["x"])[foreground]
    if not args.random_colors:
        training_combinations = set(train["type_multisets"])
        evaluation["unseen_combinations"] = torch.tensor([
            types not in training_combinations for types in evaluation["type_multisets"]
        ])
    in_features = train["x"].shape[-1]
    model = VitDensePreEmbedded(
        in_features=in_features, out_features=2, downproj=256, depth=3, num_heads=4,
        rope_dynamic_high=args.rope_high, init_scale=args.init_scale,
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    report = {
        "settings": {**vars(args), "output": str(args.output), "device": "cpu",
                     "precision": "float32", "scheduler": "constant learning rate",
                     "sampling": "balanced minibatches, fixed stratified training set"},
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_per_sample": train["x"].shape[1],
        "input_features_per_token": in_features,
        "history": [],
    }
    model.eval()
    indices = balanced_indices(train, args.batch_size)
    F.cross_entropy(forward(model, train, indices), train["y"][indices]).backward()
    report["initial_gradient_norms"] = {
        name: parameter.grad.norm().item()
        for name, parameter in model.named_parameters()
        if name in ("down.weight", "vit5.head.weight", "vit5.blocks.0.attn.qkv.weight",
                    "vit5.blocks.0.gamma_1") and parameter.grad is not None
    }
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    def record(step):
        metrics = {"step": step, "seconds": time.perf_counter() - started,
                   "train": evaluate(model, train, args.batch_size),
                   "evaluation": evaluate(model, evaluation, args.batch_size)}
        report["history"].append(metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"step={step:4d} train_loss={metrics['train']['loss']:.6f} "
              f"train_acc={metrics['train']['accuracy']:.3f} "
              f"eval_loss={metrics['evaluation']['loss']:.6f} "
              f"eval_acc={metrics['evaluation']['accuracy']:.3f} "
              f"margin_std={metrics['train']['margin_std']:.3e} "
              f"seconds={time.perf_counter() - started:.1f}", flush=True)

    record(0)
    for step in range(1, args.steps + 1):
        model.train()
        indices = balanced_indices(train, args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, train, indices), train["y"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            record(step)
    report["seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
