"""Matched CPU controls for full-grid, pooling, and object-token robustness.

Uses the original six fixed-color patterns for training. Novel-pattern evaluation
uses the other 18 permutations of exactly the same four colors. Shift evaluation
uses paired interior images translated by 1 or 2 pixels, without clipping.
Neither object type IDs nor rendering positions are passed as input features.
"""

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_vit5 import (
    balanced_indices,
    evaluate,
    forward,
    make_data,
    tokenize_image,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import (
    CYAN,
    GREEN,
    MAGENTA,
    PATTERNED_VOCAB,
    RED,
)
from sparse_wsi_vit.models.abmil import ABMIL
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


def novel_vocabulary():
    """Return all 18 fixed-palette permutations absent from the training vocabulary."""
    colors = [RED, CYAN, GREEN, MAGENTA]
    seen = {np.array(pattern).tobytes() for pattern in PATTERNED_VOCAB}
    return [np.array([colors[i] for i in order]) for order in permutations(range(4))
            if np.array([colors[i] for i in order]).tobytes() not in seen]


def render_objects(patterns, positions):
    image = np.zeros((32, 32, 3), dtype=np.float32)
    occupied = np.zeros((32, 32), dtype=bool)
    for pattern, (y, x) in zip(patterns, positions, strict=True):
        if not (0 <= y <= 28 and 0 <= x <= 28):
            raise ValueError("Object would be clipped")
        if occupied[y:y + 4, x:x + 4].any():
            raise ValueError("Objects would overlap")
        occupied[y:y + 4, x:x + 4] = True
        patch = np.array(pattern).reshape(2, 2, 3).repeat(2, axis=0).repeat(2, axis=1)
        image[y:y + 4, x:x + 4] = patch
    return image


def make_variant(source, *, shift=0, novel=False, seed=0):
    """Preserve each source label, multiplicity, and palette in a paired evaluation."""
    rng = np.random.default_rng(seed)
    vocabulary = novel_vocabulary() if novel else np.array(PATTERNED_VOCAB)
    images, features, coordinates, type_multisets = [], [], [], []
    for types, corners in zip(source["object_types"], source["positions"], strict=True):
        if novel:
            unique = sorted(set(types))
            selected = rng.choice(len(vocabulary), size=len(unique), replace=False)
            mapping = dict(zip(unique, selected.tolist(), strict=True))
            mapped_types = [mapping[t] for t in types]
            # IDs 0..5 are reserved for training types; these are analysis metadata.
            type_multisets.append(tuple(sorted(t + 6 for t in mapped_types)))
        else:
            mapped_types = types
            type_multisets.append(tuple(sorted(types)))
        positions = [(y + shift, x + shift) for y, x in corners]
        image = render_objects([vocabulary[t] for t in mapped_types], positions)
        tokens, coords = tokenize_image(image, tokenization="object")
        images.append(image)
        features.append(tokens)
        coordinates.append(coords)
    return {**source, "x": torch.from_numpy(np.stack(features)),
            "coords": torch.from_numpy(np.stack(coordinates)).float(),
            "images": np.stack(images), "type_multisets": type_multisets}


def foreground_view(data):
    keep = ~(data["x"] == -1).all(dim=-1)
    counts = keep.sum(dim=1)
    if counts.unique().numel() != 1:
        raise ValueError("Foreground bags have variable length; padding would confound the test")
    n = counts[0].item()
    return {**data, "x": data["x"][keep].reshape(len(keep), n, 48),
            "coords": data["coords"][keep].reshape(len(keep), n, 2)}


class CoordinateAdapter(nn.Module):
    """Let a coordinate-free pooling baseline share the ViT probe interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, coords):
        return self.model(x)


class MeanMIL(nn.Module):
    """ABMIL's feature projection and classifier, with uniform mean aggregation."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(48, 256), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 2),
        )

    def forward(self, x):
        return {"logits": self.classifier(self.input_proj(x).mean(dim=1))}


def build_model(name):
    if name == "vit5":
        return VitDensePreEmbedded(
            in_features=48, out_features=2, downproj=256, depth=3, num_heads=4,
            rope_dynamic_high=32, init_scale=1e-4,
        )
    if name == "abmil":
        return CoordinateAdapter(ABMIL(
            in_features=48, input_proj_dim=256, hidden_dim=256, out_features=2,
            num_branches=1, attention_dropout=0.25,
        ))
    return CoordinateAdapter(MeanMIL())


def data_checks(train, views):
    train_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in train["images"]}
    checks = {"training_unique_type_multisets": len(set(train["type_multisets"])),
              "training_vocabulary_size": 6, "novel_vocabulary_size": len(novel_vocabulary()),
              "evaluations": {}}
    for name, data in views.items():
        images = data["images"]
        histogram = np.stack([(images == color).all(axis=-1).sum(axis=(1, 2))
                              for color in [RED, CYAN, GREEN, MAGENTA]], axis=1)
        if not np.all(histogram == 16):
            raise ValueError("Color/foreground amounts changed between configurations")
        checks["evaluations"][name] = {
            "samples": len(data["y"]),
            "identical_images_in_training": sum(
                hashlib.sha256(image.tobytes()).hexdigest() in train_hashes for image in images),
            "foreground_pixel_count": 64,
            "foreground_block_counts": sorted((~(data["x"] == -1).all(dim=-1))
                                                .sum(dim=1).unique().tolist()),
            "unique_type_multisets": len(set(data["type_multisets"])),
            "distinct_object_types_observed": len({
                t for types in data["type_multisets"] for t in types}),
        }
    return checks


def train_one(name, representation, seed, train, views, args):
    torch.manual_seed(seed)
    model = build_model(name).cpu()
    # Sampling is independent of model initialization, architecture, and dropout.
    sampler = torch.Generator().manual_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    report = {
        "settings": {"model": name, "representation": representation, "seed": seed,
                     "train_size": args.train_size, "eval_size": args.eval_size,
                     "batch_size": args.batch_size, "steps": args.steps, "lr": args.lr,
                     "threads": args.threads, "random_colors": False, "device": "cpu",
                     "precision": "float32", "weight_decay": 1e-4, "grad_clip": 1.0,
                     "scheduler": "constant learning rate", "sampler_seed": seed + 1000},
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_per_sample": train["x"].shape[1],
        "evaluation_tokens": {key: data["x"].shape[1] for key, data in views.items()},
        "history": [],
    }
    output = args.output_dir / f"{name}_{representation}_seed{seed}.json"
    started = time.perf_counter()

    def record(step, final=False):
        metrics = {"step": step, "seconds": time.perf_counter() - started,
                   "train": evaluate(model, train, args.batch_size),
                   "evaluation": evaluate(model, views["standard"], args.batch_size)}
        if final:
            metrics["robustness"] = {key: evaluate(model, data, args.batch_size)
                                     for key, data in views.items() if key != "standard"}
            if name == "abmil":
                with torch.inference_mode():
                    mass = []
                    for start in range(0, args.eval_size, args.batch_size):
                        x = views["standard"]["x"][start:start + args.batch_size]
                        weights = model.model(x, return_attention=True)["attention"][:, 0]
                        foreground = ~(x == -1).all(dim=-1)
                        mass.append((weights * foreground).sum(dim=1))
                    metrics["foreground_attention_mass"] = torch.cat(mass).mean().item()
                    metrics["uniform_foreground_attention_mass"] = 4 / train["x"].shape[1]
        report["history"].append(metrics)
        report["seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}/{representation}/seed{seed} step={step} "
              f"train={metrics['train']['accuracy']:.3f} "
              f"eval={metrics['evaluation']['accuracy']:.3f} "
              f"seconds={report['seconds']:.1f}", flush=True)
        if final:
            print("robustness=" + json.dumps({key: value["accuracy"]
                  for key, value in metrics["robustness"].items()}), flush=True)

    record(0)
    for step in range(1, args.steps + 1):
        model.train()
        indices = balanced_indices(train, args.batch_size, generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, train, indices), train["y"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            record(step, final=step == args.steps)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=("vit5", "abmil", "meanmlp"),
                        default=["vit5", "abmil", "meanmlp"])
    parser.add_argument("--representations", nargs="+", choices=("foreground", "grid"),
                        default=["foreground", "grid"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("train_size", "eval_size", "batch_size"):
        if getattr(args, key) < 8 or getattr(args, key) % 8:
            parser.error(f"{key} must be a positive multiple of 8")
    if min(args.steps, args.eval_every, args.threads) < 1:
        parser.error("steps, eval-every, and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    train = make_data(args.train_size, args.seed, 1, False, tokenization="object",
                      retain_images=True)
    standard = make_data(args.eval_size, args.seed + 1, 1, False, tokenization="object",
                         retain_images=True)
    interior = make_data(args.eval_size, args.seed + 2, 1, False, tokenization="object",
                         retain_images=True, interior_only=True)
    views = {"standard": standard,
             "novel_permutations": make_variant(standard, novel=True, seed=args.seed + 100),
             "shift_control": interior,
             "shift_1px": make_variant(interior, shift=1),
             "shift_2px": make_variant(interior, shift=2)}
    training_combinations = set(train["type_multisets"])
    for data in views.values():
        data["unseen_combinations"] = torch.tensor([
            types not in training_combinations for types in data["type_multisets"]])
    checks = data_checks(train, views)
    (args.output_dir / f"data_checks_seed{args.seed}.json").write_text(
        json.dumps(checks, indent=2) + "\n")
    for representation in args.representations:
        current_train = foreground_view(train) if representation == "foreground" else train
        current_views = {key: foreground_view(data) if representation == "foreground" else data
                         for key, data in views.items()}
        for name in args.models:
            train_one(name, representation, args.seed, current_train, current_views, args)


if __name__ == "__main__":
    main()
