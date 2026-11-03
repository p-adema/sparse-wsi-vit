"""CPU controls for exact matching and the influence of all-distinct negatives.

Keep all 24 patterns, 64 flattened RGB patches, all background, and the original
ViT5/ABMIL settings. Compare a balanced one-pair/all-distinct task with the full
strict-two task sampled with twice as many all-distinct negatives per minibatch.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_controls import render_objects, train_one
from scripts.probe_mondriaan_patch_training import (
    VOCABULARY,
    build_datasets,
    dataset_checks,
)
from scripts.probe_mondriaan_vit5 import tokenize_image


def subset_data(data, indices):
    """Select samples while keeping all rendering metadata and case indices valid."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    selected = indices.tolist()
    result = {key: data[key][indices] for key in ("x", "y", "coords")}
    result["images"] = data["images"][indices.numpy()]
    for key in ("cases", "object_types", "positions", "type_multisets"):
        result[key] = [data[key][i] for i in selected]
    result["indices"] = {
        case: torch.tensor([i for i, value in enumerate(result["cases"]) if value == case])
        for case in dict.fromkeys(result["cases"])
    }
    return result


def paired_binary_view(data):
    """Keep all-distinct negatives and the positives sharing their exact layouts."""
    positives = defaultdict(list)
    for i in data["indices"]["one_pair"].tolist():
        positives[tuple(data["positions"][i])].append(i)
    selected = []
    for i in data["indices"]["all_distinct"].tolist():
        selected.extend([i, positives[tuple(data["positions"][i])].pop()])
    return subset_data(data, selected)


def pair_training_data(full_train, seed):
    """Retain every original positive and create a matched all-distinct negative.

    Changing one member of the matching pair to a uniformly selected unused
    pattern creates four distinct objects, preserving every foreground layout.
    Thus 4,096 full-task images give 2,048 unchanged positives and 2,048 negatives.
    This isolates equality from the other negative cases but also increases the
    number of distinct-negative training images; it is not a sampling-only test.
    """
    rng = np.random.default_rng(seed + 3000)
    data = subset_data(full_train, full_train["indices"]["one_pair"])
    images, x, coords, types, positions = [], [], [], [], []
    for original, corners in zip(data["object_types"], data["positions"], strict=True):
        repeated = next(t for t, count in Counter(original).items() if count == 2)
        replace = int(rng.choice([i for i, t in enumerate(original) if t == repeated]))
        distinct = list(original)
        distinct[replace] = int(rng.choice([t for t in range(24) if t not in original]))
        image = render_objects([VOCABULARY[t] for t in distinct], corners)
        tokens, token_coords = tokenize_image(image, tokenization="object")
        images.append(image)
        x.append(tokens)
        coords.append(token_coords)
        types.append(distinct)
        positions.append(list(corners))
    n = len(images)
    data.update(
        x=torch.cat([data["x"], torch.from_numpy(np.stack(x))]),
        y=torch.cat([data["y"], torch.zeros(n, dtype=torch.long)]),
        coords=torch.cat([data["coords"], torch.from_numpy(np.stack(coords)).float()]),
        images=np.concatenate([data["images"], np.stack(images)]),
        object_types=data["object_types"] + types,
        positions=data["positions"] + positions,
        cases=["one_pair"] * n + ["all_distinct"] * n,
        type_multisets=data["type_multisets"] + [tuple(sorted(t)) for t in types],
        indices={"one_pair": torch.arange(n), "all_distinct": torch.arange(n, 2 * n)},
    )
    return data


def control_indices(data, batch_size, generator=None, *, pair_only=False):
    """Retain 50/50 binary balance while changing the negative-case mixture."""
    positive = batch_size // 2
    counts = {"one_pair": positive,
              "all_distinct": positive if pair_only else batch_size // 4}
    if not pair_only:
        remainder = batch_size // 4
        # Randomize which case receives the extra slots (e.g. 3/3/2 at batch 32).
        cases = ["two_pairs", "three_matching", "four_matching"]
        ordering = torch.randperm(3, generator=generator).tolist()
        counts.update({case: remainder // 3 for case in cases})
        for i in ordering[:remainder % 3]:
            counts[cases[i]] += 1
    selected = []
    for case, count in counts.items():
        pool = data["indices"][case]
        selected.append(pool[torch.randint(len(pool), (count,), generator=generator)])
    indices = torch.cat(selected)
    return indices[torch.randperm(len(indices), generator=generator)]


def pair_indices(data, batch_size, generator=None):
    return control_indices(data, batch_size, generator, pair_only=True)


def prepare_data(train_size, eval_size, seed):
    training, evaluations, original_checks = build_datasets(train_size, eval_size, seed)
    full_train = training["all24_aligned"]
    pair_train = pair_training_data(full_train, seed)
    full_eval = evaluations["all24_aligned"]
    pair_eval = paired_binary_view(full_eval)
    checks = {
        "seed": seed,
        "original_data": original_checks,
        "pair_training": dataset_checks(pair_train),
        "pair_evaluation": dataset_checks(pair_eval),
        "pair_training_positives_identical_to_original": torch.equal(
            pair_train["x"][:train_size // 2],
            full_train["x"][full_train["indices"]["one_pair"]]),
        "overlap": {},
    }
    for name, data in {"pair_only": pair_train, "upweighted": full_train}.items():
        hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]}
        overlap = sum(hashlib.sha256(image.tobytes()).hexdigest() in hashes
                      for image in full_eval["images"])
        if overlap:
            raise ValueError("Training and evaluation share an identical image")
        checks["overlap"][name] = overlap
    return {"pair_only": pair_train, "upweighted": full_train}, full_eval, pair_eval, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=("vit5", "abmil"),
                        default=["vit5", "abmil"])
    parser.add_argument("--controls", nargs="+", choices=("pair_only", "upweighted"),
                        default=["pair_only", "upweighted"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("train_size", "eval_size", "batch_size"):
        if getattr(args, key) < 8 or getattr(args, key) % 8:
            parser.error(f"{key} must be a positive multiple of eight")
    if min(args.steps, args.eval_every, args.threads) < 1:
        parser.error("steps, eval-every, and threads must be positive")
    torch.set_num_threads(args.threads)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    training, full_eval, pair_eval, checks = prepare_data(args.train_size, args.eval_size,
                                                        args.seed)
    checks_dir = root / (args.controls[0] if len(args.controls) == 1 else "data_checks")
    checks_dir.mkdir(parents=True, exist_ok=True)
    checks_path = checks_dir / f"data_checks_seed{args.seed}.json"
    checks_path.write_text(json.dumps(checks, indent=2) + "\n")
    for control in args.controls:
        train = training[control]
        views = {"standard": pair_eval if control == "pair_only" else full_eval,
                 "full_task": full_eval, "pair_balanced": pair_eval}
        known = set(train["type_multisets"])
        views = {name: {**data, "unseen_combinations": torch.tensor([
            types not in known for types in data["type_multisets"]])}
            for name, data in views.items()}
        args.output_dir = root / control
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            train_one(
                model, "grid", args.seed, train, views, args,
                experiment_settings={
                    "control": control, "vocabulary_size": 24,
                    "object_placement": "aligned", "rope_high": 32, "init_scale": 1e-4,
                    "eval_size": len(views["standard"]["y"]),
                    "full_eval_size": len(full_eval["y"]),
                    "data_checks_file": str(checks_path),
                    "minibatch_case_counts": ({"one_pair": args.batch_size // 2,
                                              "all_distinct": args.batch_size // 2}
                        if control == "pair_only" else
                        {"one_pair": args.batch_size // 2,
                         "all_distinct": args.batch_size // 4,
                         "other_negative_total": args.batch_size // 4,
                         "other_negative_allocation": "equal; remainder randomly rotated"}),
                }, save_checkpoint=True,
                batch_sampler=pair_indices if control == "pair_only" else control_indices,
            )


if __name__ == "__main__":
    main()
