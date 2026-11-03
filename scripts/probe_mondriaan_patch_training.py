"""Separate vocabulary and placement controls with a fixed 64-token patch grid.

Every 32x32 image is represented by 64 flattened 4x4 patches (48 RGB values).
Compare six aligned patterns, all 24 aligned permutations, and six patterns
with independently sampled integer-pixel object positions. Four objects, the
strict-two labels, fixed colors, and all background patches are retained.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_controls import novel_vocabulary, render_objects, train_one
from scripts.probe_mondriaan_vit5 import make_data, tokenize_image
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import PATTERNED_VOCAB


VOCABULARY = np.concatenate([np.array(PATTERNED_VOCAB), np.array(novel_vocabulary())])
CONDITIONS = {"six_aligned": (6, True), "all24_aligned": (24, True),
              "six_random": (6, False)}


def sample_positions(rng, aligned):
    """Independently propose corners, rejecting distances below eight pixels."""
    corners = []
    for _ in range(80_000):
        y, x = rng.integers(0, 8, size=2) * 4 if aligned else rng.integers(0, 29, size=2)
        corner = (int(y), int(x))
        if all((y - py) ** 2 + (x - px) ** 2 >= 64 for py, px in corners):
            corners.append(corner)
            if len(corners) == 4:
                return corners
    raise RuntimeError("Could not place four separated objects")


def balanced_layouts(source, *, seed, aligned):
    """Reuse each random layout once per label, eliminating background-shape cues."""
    rng = np.random.default_rng(seed)
    labels = source["y"].numpy()
    positive = rng.permutation(np.flatnonzero(labels == 1))
    negative = rng.permutation(np.flatnonzero(labels == 0))
    if len(positive) != len(negative):
        raise ValueError("Layout matching requires balanced labels")
    layouts = [None] * len(labels)
    for pos, neg in zip(positive, negative, strict=True):
        corners = sample_positions(rng, aligned)
        layouts[pos], layouts[neg] = corners, list(corners)
    return layouts


def make_patch_view(source, vocabulary_ids, positions, *, mapping_seed):
    rng = np.random.default_rng(mapping_seed)
    original_six = list(vocabulary_ids) == list(range(6))
    images, features, coords, actual_types = [], [], [], []
    for types, corners in zip(source["object_types"], positions, strict=True):
        if original_six:
            mapped = list(types)
        else:
            unique = sorted(set(types))
            selected = rng.choice(vocabulary_ids, size=len(unique), replace=False)
            mapping = dict(zip(unique, selected.tolist(), strict=True))
            mapped = [mapping[t] for t in types]
        image = render_objects([VOCABULARY[t] for t in mapped], corners)
        tokens, token_coords = tokenize_image(image, tokenization="object")
        images.append(image)
        features.append(tokens)
        coords.append(token_coords)
        actual_types.append(mapped)
    return {**source, "x": torch.from_numpy(np.stack(features)),
            "coords": torch.from_numpy(np.stack(coords)).float(),
            "images": np.stack(images), "positions": positions, "object_types": actual_types,
            "type_multisets": [tuple(sorted(types)) for types in actual_types]}


def dataset_checks(data):
    phases, fragments = [], []
    masks = {0: Counter(), 1: Counter()}
    for i, (tokens, coords) in enumerate(zip(data["x"].numpy(),
                                           data["coords"].int().numpy(), strict=True)):
        if tokens.shape != (64, 48):
            raise ValueError("Every sample must contain 64 full pixel patches")
        image = np.zeros((32, 32, 3), dtype=np.float32)
        for token, (x, y) in zip(tokens, coords, strict=True):
            if not np.all(token == -1):
                image[y:y + 4, x:x + 4] = token.reshape(4, 4, 3)
        if not np.array_equal(image, data["images"][i]):
            raise ValueError("Patch representation lost pixels")
        expected_grid = np.array([(x, y) for y in range(0, 32, 4) for x in range(0, 32, 4)])
        if not np.array_equal(coords, expected_grid):
            raise ValueError("Patch coordinates must stay fixed")
        crops = []
        corners = data["positions"][i]
        for j, (y, x) in enumerate(corners):
            if not (0 <= y <= 28 and 0 <= x <= 28):
                raise ValueError("Clipped object")
            if any((y - py) ** 2 + (x - px) ** 2 < 64 for py, px in corners[:j]):
                raise ValueError("Object separation changed")
            crops.append(image[y:y + 4, x:x + 4].tobytes())
            phases.append((y % 4, x % 4))
            fragments.append((1 + int(y % 4 != 0)) * (1 + int(x % 4 != 0)))
        counts = Counter(crops)
        label = int(max(counts.values()) == 2 and list(counts.values()).count(2) == 1)
        if label != data["y"][i].item():
            raise ValueError("Pixel-based strict-two label changed")
        if sorted(counts.values()) != sorted(Counter(data["object_types"][i]).values()):
            raise ValueError("Object identity metadata disagrees with pixels")
        for color in VOCABULARY[0]:
            if (image == color).all(axis=-1).sum() != 16:
                raise ValueError("Global color amounts changed")
        masks[label][np.any(image != 0, axis=-1).tobytes()] += 1
    if masks[0] != masks[1]:
        raise ValueError("Background layouts reveal a binary label")
    foreground_counts = (~(data["x"] == -1).all(dim=-1)).sum(dim=1).numpy()
    return {"samples": len(data["y"]), "shape": list(data["x"].shape),
            "case_counts": dict(Counter(data["cases"])),
            "observed_vocabulary_ids": sorted({t for types in data["object_types"] for t in types}),
            "unique_type_multisets": len(set(data["type_multisets"])),
            "pixel_reconstruction_exact": True, "pixel_label_oracle_accuracy": 1.0,
            "background_layouts_exactly_balanced_across_labels": True,
            "foreground_pixels": 64, "pixels_per_palette_color": 16,
            "foreground_patch_counts": dict(sorted(Counter(foreground_counts.tolist()).items())),
            "corner_phase_counts": {f"{y},{x}": count
                                    for (y, x), count in sorted(Counter(phases).items())},
            "patch_fragments_per_object": dict(sorted(Counter(fragments).items()))}


def build_datasets(train_size, eval_size, seed):
    train_source = make_data(train_size, seed, 1, False, tokenization="object", retain_images=True)
    eval_source = make_data(eval_size, seed + 1, 1, False, tokenization="object", retain_images=True)
    train_layouts = {True: balanced_layouts(train_source, seed=seed + 2000, aligned=True),
                     False: balanced_layouts(train_source, seed=seed + 2000, aligned=False)}
    eval_layouts = {True: balanced_layouts(eval_source, seed=seed + 2001, aligned=True),
                    False: balanced_layouts(eval_source, seed=seed + 2001, aligned=False)}
    training = {
        name: make_patch_view(train_source, range(vocab_size), train_layouts[aligned],
                              mapping_seed=seed + 1000)
        for name, (vocab_size, aligned) in CONDITIONS.items()
    }
    evaluations = {
        name: make_patch_view(eval_source, range(vocab_size), eval_layouts[aligned],
                              mapping_seed=seed + 1001)
        for name, (vocab_size, aligned) in {**CONDITIONS, "all24_random": (24, False)}.items()
    }
    evaluations["other18_aligned"] = make_patch_view(
        eval_source, range(6, 24), eval_layouts[True], mapping_seed=seed + 1001)
    checks = {"seed": seed, "random_colors": False, "min_separation": 8,
              "placement": "independent corner proposals; each layout reused once per binary label",
              "training": {name: dataset_checks(data) for name, data in training.items()},
              "evaluation": {name: dataset_checks(data) for name, data in evaluations.items()},
              "overlap": {}}
    for name, train in training.items():
        hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in train["images"]}
        checks["overlap"][name] = {}
        for view, data in evaluations.items():
            overlap = sum(hashlib.sha256(image.tobytes()).hexdigest() in hashes
                          for image in data["images"])
            if overlap:
                raise ValueError("Training and evaluation share identical images")
            checks["overlap"][name][view] = overlap
    return training, evaluations, checks


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
    parser.add_argument("--models", nargs="+", choices=("vit5", "abmil"), default=["vit5", "abmil"])
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS),
                        default=list(CONDITIONS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("train_size", "eval_size", "batch_size"):
        if getattr(args, key) < 8 or getattr(args, key) % 8:
            parser.error(f"{key} must be a positive multiple of 8")
    if min(args.steps, args.eval_every, args.threads, args.lr) <= 0:
        parser.error("steps, eval-every, threads, and lr must be positive")
    torch.set_num_threads(args.threads)
    training, evaluations, checks = build_datasets(args.train_size, args.eval_size, args.seed)
    if ("all24_aligned" in args.conditions and
            len(checks["training"]["all24_aligned"]["observed_vocabulary_ids"]) != 24):
        parser.error("Increase train-size: training must actually contain all 24 patterns")
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    checks_file = output_root / f"data_checks_seed{args.seed}.json"
    checks_file.write_text(json.dumps(checks, indent=2) + "\n")
    for condition in args.conditions:
        train = training[condition]
        views = {"standard": evaluations[condition],
                 **{name: data for name, data in evaluations.items() if name != condition}}
        known_bags = set(train["type_multisets"])
        views = {name: {**data, "unseen_combinations": torch.tensor([
            types not in known_bags for types in data["type_multisets"]])}
                 for name, data in views.items()}
        args.output_dir = output_root / condition
        args.output_dir.mkdir(parents=True, exist_ok=True)
        vocabulary_size, aligned = CONDITIONS[condition]
        for model in args.models:
            train_one(model, "grid", args.seed, train, views, args,
                      experiment_settings={"condition": condition,
                          "vocabulary_size": vocabulary_size,
                          "object_placement": "aligned" if aligned else "random integer pixels",
                          "rope_high": 32, "init_scale": 1e-4,
                          "data_checks_file": str(checks_file)}, save_checkpoint=True)


if __name__ == "__main__":
    main()
