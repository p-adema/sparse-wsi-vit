"""CPU generalization tests after the quadrant-token small-data fitting gate.

Split mode trains on 12 fixed-palette permutations and evaluates the other 12.
Augmented mode trains with shifts 0/1/2 and evaluates a held-out shift of 3 pixels.
Both retain all 256 2x2 patches. Augmented mode flattens each patch into 12 ordered
RGB values so shifted inputs preserve every pixel. Original generator labels and
case quotas are retained; vocabulary IDs and rendering corners are metadata only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_controls import novel_vocabulary, render_objects
from scripts.probe_mondriaan_vit5 import balanced_indices, evaluate, forward, make_data, tokenize_image
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import PATTERNED_VOCAB
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


VOCABULARY = np.concatenate([np.array(PATTERNED_VOCAB), np.array(novel_vocabulary())])


def vocabulary_split():
    order = np.random.default_rng(2026).permutation(24)
    return order[:12].tolist(), order[12:].tolist()


def make_view(source, vocabulary_ids, *, mapping_seed, phases=(0,), phase_seed=0,
              flattened=False):
    rng = np.random.default_rng(mapping_seed)
    phase_rng = np.random.default_rng(phase_seed)
    assigned_phases = np.empty(len(source["y"]), dtype=np.int64)
    # Exactly match phase frequencies across the two balanced labels.
    for label in (0, 1):
        indices = np.flatnonzero(source["y"].numpy() == label)
        values = np.resize(np.array(phases), len(indices))
        phase_rng.shuffle(values)
        assigned_phases[indices] = values
    images, features, coordinates, object_types, positions = [], [], [], [], []
    for types, corners, phase in zip(source["object_types"], source["positions"],
                                     assigned_phases, strict=True):
        unique = sorted(set(types))
        chosen = rng.choice(vocabulary_ids, size=len(unique), replace=False)
        mapping = dict(zip(unique, chosen.tolist(), strict=True))
        actual_types = [mapping[t] for t in types]
        actual_positions = [(y + int(phase), x + int(phase)) for y, x in corners]
        image = render_objects([VOCABULARY[t] for t in actual_types], actual_positions)
        tokens, coords = tokenize_image(image, pool_size=2,
                                        tokenization="patch" if flattened else "pixel")
        images.append(image)
        features.append(tokens)
        coordinates.append(coords)
        object_types.append(actual_types)
        positions.append(actual_positions)
    return {**source, "x": torch.from_numpy(np.stack(features)),
            "coords": torch.from_numpy(np.stack(coordinates)).float(),
            "images": np.stack(images), "object_types": object_types, "positions": positions,
            "type_multisets": [tuple(sorted(types)) for types in object_types],
            "phases": assigned_phases.tolist()}


def data_checks(train, views, training_ids, held_ids, flattened):
    hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in train["images"]}
    checks = {"training_vocabulary_ids": training_ids, "held_out_vocabulary_ids": held_ids,
              "vocabulary_split_seed": 2026, "datasets": {}}
    for name, data in {"train": train, **views}.items():
        correct = []
        for i, (tokens, coords) in enumerate(zip(data["x"].numpy(), data["coords"].int().numpy(),
                                                strict=True)):
            reconstructed = np.zeros((32, 32, 3), dtype=np.float32)
            for token, (x, y) in zip(tokens, coords, strict=True):
                if not np.all(token == -1):
                    reconstructed[y:y + 2, x:x + 2] = token.reshape(2, 2, 3) if flattened else token
            if not np.array_equal(reconstructed, data["images"][i]):
                raise ValueError("Tokenization discarded pixel information")
            patterns = [reconstructed[y:y + 4, x:x + 4].tobytes()
                        for y, x in data["positions"][i]]
            counts = Counter(patterns)
            correct.append(int(max(counts.values()) == 2 and list(counts.values()).count(2) == 1))
        if not np.array_equal(correct, data["y"].numpy()):
            raise ValueError("Rendered repetitions changed the truth-table label")
        observed = sorted({t for types in data["object_types"] for t in types})
        overlap = (0 if name == "train" else sum(
            hashlib.sha256(image.tobytes()).hexdigest() in hashes for image in data["images"]))
        if overlap:
            raise ValueError("Exact image overlap between training and evaluation")
        checks["datasets"][name] = {
            "shape": list(data["x"].shape), "observed_vocabulary_ids": observed,
            "observed_types_outside_training_vocabulary": sorted(set(observed) - set(training_ids)),
            "phase_counts": dict(Counter(data["phases"])),
            "phase_counts_by_case": {case: dict(Counter(data["phases"][i] for i in indices.tolist()))
                                     for case, indices in data["indices"].items()},
            "pixel_reconstruction_exact": True, "oracle_accuracy": 1.0,
            "identical_images_in_training": overlap,
        }
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "augmented"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rope-high", type=float, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("train_size", "eval_size", "batch_size"):
        if getattr(args, name) < 8 or getattr(args, name) % 8:
            parser.error(f"{name} must be a positive multiple of 8")
    if min(args.steps, args.eval_every, args.threads, args.rope_high) <= 0:
        parser.error("steps, eval-every, threads, and rope-high must be positive")
    torch.set_num_threads(args.threads)
    training_ids, held_ids = vocabulary_split()
    flattened = args.mode == "augmented"
    train_source = make_data(args.train_size, args.seed, 2, False,
                             retain_images=True, interior_only=True)
    eval_source = make_data(args.eval_size, args.seed + 1, 2, False,
                            retain_images=True, interior_only=True)
    train = make_view(train_source, training_ids, mapping_seed=args.seed + 1000,
                      phases=(0, 1, 2) if flattened else (0,), phase_seed=args.seed + 2000,
                      flattened=flattened)
    if args.mode == "split":
        views = {
            "known_patterns": make_view(eval_source, training_ids, mapping_seed=args.seed + 1001),
            "held_out_patterns": make_view(eval_source, held_ids, mapping_seed=args.seed + 1001),
        }
        primary = "known_patterns"
    else:
        views = {
            "mixed_training_phases": make_view(eval_source, training_ids,
                mapping_seed=args.seed + 1001, phases=(0, 1, 2), phase_seed=args.seed + 2001,
                flattened=True),
            "aligned_control": make_view(eval_source, training_ids,
                mapping_seed=args.seed + 1001, flattened=True),
            "held_out_shift3": make_view(eval_source, training_ids,
                mapping_seed=args.seed + 1001, phases=(3,), flattened=True),
        }
        primary = "mixed_training_phases"
    checks = data_checks(train, views, training_ids, held_ids, flattened)
    torch.manual_seed(args.seed)
    model = VitDensePreEmbedded(
        in_features=train["x"].shape[-1], out_features=2, downproj=256, depth=3,
        num_heads=4, rope_dynamic_high=args.rope_high, init_scale=1e-4,
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(args.seed + 1000)
    report = {"settings": {**vars(args), "output": str(args.output), "device": "cpu",
                           "precision": "float32", "random_colors": False,
                           "scheduler": "constant learning rate", "weight_decay": 1e-4,
                           "grad_clip": 1.0, "init_scale": 1e-4, "sampler_seed": args.seed + 1000},
              "parameters": sum(p.numel() for p in model.parameters()),
              "data_checks": checks, "history": []}
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(step, final=False):
        metrics = {"step": step, "seconds": time.perf_counter() - started,
                   "train": evaluate(model, train, args.batch_size),
                   "evaluation": evaluate(model, views[primary], args.batch_size)}
        if final:
            metrics["generalization"] = {name: evaluate(model, data, args.batch_size)
                                         for name, data in views.items()}
            checkpoint = args.output.with_suffix(".pt")
            torch.save({"settings": report["settings"], "model_state": model.state_dict()}, checkpoint)
            report["checkpoint"] = str(checkpoint)
        report["history"].append(metrics)
        report["seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.mode}/seed{args.seed} step={step} "
              f"train={metrics['train']['accuracy']:.3f} "
              f"eval={metrics['evaluation']['accuracy']:.3f} "
              f"seconds={report['seconds']:.1f}", flush=True)
        if final:
            print("generalization=" + json.dumps({name: value["accuracy"]
                  for name, value in metrics["generalization"].items()}), flush=True)

    record(0)
    for step in range(1, args.steps + 1):
        model.train()
        indices = balanced_indices(train, args.batch_size, generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, train, indices), train["y"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            record(step, final=step == args.steps)


if __name__ == "__main__":
    main()
