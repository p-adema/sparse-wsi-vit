"""Validation-selected Mondriaan comparisons with a fresh, separate final test.

Examples (CPU; all artifacts stay local):
  .venv/bin/python scripts/probe_mondriaan_benchmark.py --models mean sum coord_mean abmil vit5
  .venv/bin/python scripts/probe_mondriaan_benchmark.py --placement offgrid --representation objects
  .venv/bin/python scripts/probe_mondriaan_benchmark.py --task restart --steps 4000
  .venv/bin/python scripts/probe_mondriaan_benchmark.py --task composite --canvas-size 96 --scene-count 4

The protocol is written before fitting. Validation selects checkpoints; test
images are generated only after all requested models have finished fitting.
The fixed seeds here are experiment seeds, not confidence-interval bounds.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.mondriaan_metrics import CASE_NAMES, classification_metrics
from scripts.probe_mondriaan_controls import CoordinateAdapter
from sparse_wsi_vit.models.abmil import ABMIL
from sparse_wsi_vit.models.mondriaan_pooling import LearnedPooling, comparable_pooling_width
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


def tensor_data(data):
    result = dict(data)
    for key in ("x", "coords", "y", "lengths"):
        if key in result:
            result[key] = torch.as_tensor(result[key], dtype=(
                torch.long if key in ("y", "lengths") else torch.float32))
    aliases = {"triple": "three_matching", "all_matching": "four_matching"}
    counts = data.get("object_counts")
    varying = counts is not None and len(set(map(int, counts))) > 1
    result["cases"] = [
        (f"n{int(counts[i])}/{case}" if varying else
         aliases.get(case, case) if counts is not None and int(counts[i]) == 4 else case)
        for i, case in enumerate(result["cases"])]
    return result


def build_model(name, canvas_size=32, width=None, transmil_landmarks=32):
    if name == "vit5":
        return VitDensePreEmbedded(in_features=48, out_features=2, downproj=256,
                                  depth=3, num_heads=4, rope_dynamic_high=canvas_size,
                                  init_scale=1e-4)
    if name == "abmil":
        return CoordinateAdapter(ABMIL(in_features=48, input_proj_dim=256,
                                      hidden_dim=256, out_features=2,
                                      num_branches=1, attention_dropout=0.25))
    if name == "transmil":
        from sparse_wsi_vit.models.mondriaan_transmil import MondriaanTransMIL
        return MondriaanTransMIL(in_features=48, out_features=2, num_landmarks=transmil_landmarks)
    if name not in ("mean", "sum", "coord_mean", "coord_sum"):
        raise ValueError(f"Unknown model {name}")
    coordinate_aware = name.startswith("coord_")
    return LearnedPooling(width=width or comparable_pooling_width(
        use_coordinates=coordinate_aware), aggregation=name.split("_")[-1],
        use_coordinates=coordinate_aware, coordinate_scale=canvas_size)


def planned_cases(args):
    """Define the objective before sampling; missing validation cases cannot shrink it."""
    if args.task == "composite":
        return tuple(f"positive_scene_count_{n}" for n in range(args.scene_count + 1))
    from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import case_multiplicities
    if len(args.object_counts) == 1 and args.object_counts[0] == 4:
        return CASE_NAMES
    return tuple(sorted(f"n{count}/{case}" if len(args.object_counts) > 1 else case
                        for count in args.object_counts for case in case_multiplicities(count)))


def forward_batch(model, data, indices):
    x, coords = data["x"][indices], data["coords"][indices]
    lengths = data.get("lengths")
    if lengths is None:
        return model(x, coords)["logits"]
    lengths = lengths[indices]
    if isinstance(model, LearnedPooling):
        return model(x, coords, lengths=lengths)["logits"]
    if isinstance(model, CoordinateAdapter):
        return model.model(x, lengths=lengths)["logits"]
    # ViT has no padding-mask interface. Group complete-object diagnostics by
    # actual length rather than exposing fake background objects to attention.
    outputs, positions = [], []
    for length in lengths.unique().tolist():
        selected = torch.nonzero(lengths == length).flatten()
        outputs.append(model(x[selected, :length], coords[selected, :length])["logits"])
        positions.append(selected)
    order = torch.cat(positions).argsort()
    return torch.cat(outputs)[order]


def balanced_batch(data, batch_size, generator):
    """Balance both labels and cycle uniformly through each label's subgroups."""
    object_counts = data.get("object_counts")
    if object_counts is not None and len(set(map(int, object_counts))) > 1:
        # Choose identical count allocations for both labels. Uniform sampling
        # across all negative families would overweight counts with more
        # families (three objects has fewer partitions than four/six).
        counts = sorted(set(map(int, object_counts)))
        allocation = dict.fromkeys(counts, (batch_size // 2) // len(counts))
        extra = (batch_size // 2) % len(counts)
        for i in torch.randperm(len(counts), generator=generator)[:extra].tolist():
            allocation[counts[i]] += 1
        pieces = []
        for label in (0, 1):
            for count in counts:
                groups = defaultdict(list)
                for i, (target, case, observed_count) in enumerate(zip(
                        data["y"].tolist(), data["cases"], object_counts, strict=True)):
                    if target == label and int(observed_count) == count:
                        groups[case].append(i)
                if not groups:
                    raise ValueError("Every object count requires both binary labels")
                keys = sorted(groups)
                order = torch.randperm(len(keys), generator=generator).tolist()
                for j in range(allocation[count]):
                    group = groups[keys[order[j % len(keys)]]]
                    pieces.append(group[torch.randint(len(group), (), generator=generator).item()])
        indices = torch.tensor(pieces)
        return indices[torch.randperm(len(indices), generator=generator)]
    pieces = []
    for label in (0, 1):
        groups = defaultdict(list)
        for i, (target, case) in enumerate(zip(data["y"].tolist(), data["cases"], strict=True)):
            if target == label:
                groups[case].append(i)
        if not groups:
            raise ValueError("Training requires both binary labels")
        keys = sorted(groups)
        choices = torch.randperm(len(keys), generator=generator).tolist()
        for j in range(batch_size // 2):
            group = groups[keys[choices[j % len(keys)]]]
            pieces.append(group[torch.randint(len(group), (), generator=generator).item()])
    indices = torch.tensor(pieces)
    return indices[torch.randperm(len(indices), generator=generator)]


@torch.inference_mode()
def evaluate(model, data, batch_size=32, intended_cases=None, return_predictions=False):
    model.eval()
    logits = torch.cat([forward_batch(model, data, torch.arange(i, min(
        i + batch_size, len(data["y"])))) for i in range(0, len(data["y"]), batch_size)])
    predictions = logits.argmax(dim=1).numpy()
    metrics = classification_metrics(data["y"].numpy(), predictions, data["cases"],
                                     intended_cases=(intended_cases if intended_cases is not None
                                                     else data.get("intended_cases", CASE_NAMES)))
    metrics["loss"] = F.cross_entropy(logits, data["y"]).item()
    return (metrics, predictions) if return_predictions else metrics


def state_bundle(model, optimizer, sampler, step, settings):
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "sampler_state": sampler.get_state(), "torch_rng_state": torch.get_rng_state(),
            "step": step, "settings": settings}


def load_training_state(model, optimizer, sampler, state, restart_optimizer=False):
    model.load_state_dict(state["model_state"])
    if not restart_optimizer:
        optimizer.load_state_dict(state["optimizer_state"])
    sampler.set_state(state["sampler_state"])
    torch.set_rng_state(state["torch_rng_state"])
    return int(state["step"])


def checkpoint_rank(metrics):
    worst = metrics.get("worst_case_accuracy")
    if worst is None:
        raise ValueError("Validation is missing an intended subgroup; checkpoint selection is ineligible")
    return (worst, metrics["binary_macro_f1"], metrics["accuracy"])


def train(model, training, validation, *, seed, steps, batch_size=32, lr=2e-4,
          eval_every=400, output_dir, settings=None, initial_state=None,
          restart_optimizer=False, save_midpoint=None, inherited_selection=None):
    """Fit without test data; serialization preserves the sampling/dropout stream."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(seed + 1000)
    start = 0
    if initial_state is not None:
        start = load_training_state(model, optimizer, sampler, initial_state,
                                    restart_optimizer=restart_optimizer)
    else:
        torch.manual_seed(seed + 2000)
    history, best_rank, best_step = [], None, None
    if inherited_selection is not None:
        best_rank = tuple(inherited_selection["rank"])
        best_step = inherited_selection["step"]
        shutil.copyfile(inherited_selection["checkpoint"], output_dir / "best.pt")
    started = time.perf_counter()
    intended = tuple(validation.get("intended_cases", CASE_NAMES))

    def record(step):
        nonlocal best_rank, best_step
        val = evaluate(model, validation, batch_size, intended)
        rank = checkpoint_rank(val)
        entry = {"step": step, "validation": val,
                 "seconds": time.perf_counter() - started}
        history.append(entry)
        if best_rank is None or rank > best_rank:
            best_rank, best_step = rank, step
            torch.save(state_bundle(model, optimizer, sampler, step, settings),
                       output_dir / "best.pt")
        # A validation-selected checkpoint may be much older than the current
        # trajectory. Preserve the latest exact state for interrupted long fits.
        latest = output_dir / "latest.tmp"
        torch.save(state_bundle(model, optimizer, sampler, step, settings), latest)
        latest.replace(output_dir / "latest.pt")
        print(f"{settings.get('model', 'model')}/seed{seed} step={step} "
              f"val={val['accuracy']:.3f} worst={val['worst_case_accuracy']:.3f} "
              f"macro_case={val['macro_case_accuracy']:.3f} f1={val['positive_f1']:.3f}",
              flush=True)

    record(start)
    for step in range(start + 1, steps + 1):
        model.train()
        indices = balanced_batch(training, batch_size, sampler)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward_batch(model, training, indices), training["y"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if save_midpoint == step:
            torch.save(state_bundle(model, optimizer, sampler, step, settings),
                       output_dir / "midpoint.pt")
        if step % eval_every == 0 or step == steps:
            record(step)
    torch.save(state_bundle(model, optimizer, sampler, steps, settings), output_dir / "final.pt")
    report = {"settings": settings, "parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_selection": "validation worst-case, then binary macro-F1, then accuracy",
              "selected_step": best_step, "history": history,
              "selected_rank": list(best_rank),
              "selection_scope": "all scheduled validation checkpoints, including inherited prefix best",
              "train_final": evaluate(model, training, batch_size),
              "seconds": time.perf_counter() - started,
              "input_shape": list(training["x"].shape)}
    (output_dir / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def grouped_test_uncertainty(data, predictions, seed, repeats=1000):
    """Bootstrap matched layouts together; do not confuse this with seed variation."""
    correct = predictions == data["y"].numpy()
    groups = defaultdict(list)
    pair_ids = data.get("layout_pair_ids", data.get("pair_ids"))
    if pair_ids is not None:
        for i, pair in enumerate(pair_ids):
            groups[int(pair)].append(i)
    elif "positions" in data:
        for i, positions in enumerate(data["positions"]):
            groups[tuple(sorted(tuple(p) for p in positions))].append(i)
    else:
        for i in range(len(correct)):
            groups[i].append(i)
    values = list(groups.values())
    sums = np.array([correct[g].sum() for g in values])
    sizes = np.array([len(g) for g in values])
    rng = np.random.default_rng(seed + 9000)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {"method": "layout-group bootstrap", "groups": len(values), "repeats": repeats,
            "accuracy_95_percentile_interval": np.quantile(samples, [.025, .975]).tolist(),
            "interpretation": "test-sample uncertainty conditional on this checkpoint, not training-seed variation"}


def make_local_splits(args, seed):
    from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
        make_appearance_pools, make_controlled_dataset,
    )
    known = args.variant == "quadrant24" and not args.appearance_holdout
    sizes = dict.fromkeys(("train", "validation", "test"), 24 if known else args.appearance_pool_size)
    pools = make_appearance_pools(variant=args.variant, sizes=sizes,
                                  seed=seed + getattr(args, "appearance_seed_offset", 3000),
                                  min_distance=args.min_distance, disjoint=not known)
    def create(split, size, offset):
        return make_controlled_dataset(pools[split], size=size, seed=seed + offset,
                                       object_counts=tuple(args.object_counts),
                                       aligned=args.placement == "aligned",
                                       palette_mode=args.palette, split=split)
    training = create("train", args.train_size, 4000)
    validation = create("validation", args.validation_size, 5000)
    return pools, training, validation, lambda: create("test", args.test_size, getattr(args, "test_seed_offset", 6000))


def prepare_task(args, seed):
    pools, training, validation, create_test = make_local_splits(args, seed)
    from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import dataset_checks, audit_splits
    checks = {"train": dataset_checks(training), "validation": dataset_checks(validation),
              "overlap": audit_splits({"train": training, "validation": validation},
                                      require_disjoint_appearances=(args.variant != "quadrant24" or args.appearance_holdout))}
    (args.output_dir / f"development_checks_seed{seed}.json").write_text(json.dumps(checks, indent=2) + "\n")
    if args.task == "composite":
        from sparse_wsi_vit.datasets.mondriaan_dataset.composite import (
            scene_pool_from_data, build_balanced_composite, verify_split_pools, lattice_origins,
        )
        train_bank, validation_bank = scene_pool_from_data(training), scene_pool_from_data(validation)
        heldout = args.variant != "quadrant24" or args.appearance_holdout
        verify_split_pools([train_bank, validation_bank], appearance_holdout=heldout)
        fixed_origins = (lattice_origins(args.scene_layout_size, args.scene_count)[:args.scene_count]
                         if args.scene_layout_size else None)
        def compose(data, size, offset):
            bank = data if hasattr(data, "scene_ids") else scene_pool_from_data(data)
            return build_balanced_composite(bank, n_samples=size,
                                           canvas_size=args.canvas_size,
                                           scene_count=args.scene_count, rule=args.rule,
                                           exact_k=args.exact_k, seed=seed + offset,
                                           palette_mode="shared_per_canvas" if args.palette == "shared" else "fixed",
                                           fixed_origins=fixed_origins)
        training = compose(train_bank, args.train_size, 7000)
        validation = compose(validation_bank, args.validation_size, 8000)
        local_create_test = create_test
        def create_test():
            test_bank = scene_pool_from_data(local_create_test())
            overlap = verify_split_pools([train_bank, validation_bank, test_bank], appearance_holdout=heldout)
            (args.output_dir / f"constituent_checks_seed{seed}.json").write_text(json.dumps(overlap, indent=2) + "\n")
            return compose(test_bank, args.test_size, 9000)
        for data in (training, validation):
            data["case_mixes"] = data["cases"]
            data["cases"] = [f"positive_scene_count_{n}" for n in data["positive_scene_counts"]]
    elif args.representation == "objects":
        from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import complete_object_view
        training, validation = complete_object_view(training), complete_object_view(validation)
        local_create_test = create_test
        def create_test():
            return complete_object_view(local_create_test())
    training, validation = tensor_data(training), tensor_data(validation)
    training["intended_cases"] = validation["intended_cases"] = planned_cases(args)
    return pools, training, validation, create_test


def flatten_metrics(metrics):
    keys = ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "positive_precision",
            "positive_recall", "positive_f1", "binary_macro_f1", "loss")
    result = {key: metrics.get(key) for key in keys}
    for case, item in metrics["by_case"].items():
        result[f"accuracy_{case}"] = item["accuracy"]
        result[f"count_{case}"] = item["count"]
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    result.update(tn=tn, fp=fp, fn=fn, tp=tp)
    return result


def write_results(root, results):
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = []
    for run in results:
        row = {k: run[k] for k in ("model", "seed", "selected_step", "parameters")}
        row.update(flatten_metrics(run["test"]))
        row.update(seconds=run["seconds"], tokens=run["tokens"])
        if "arm" in run:
            row["arm"] = run["arm"]
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = [f"{r['model']} {r.get('arm','')}\nseed {r['seed']}" for r in results]
    for ax, metric, title in zip(axes, ("accuracy", "worst_case_accuracy"),
                                 ("Fresh test: overall", "Fresh test: worst subgroup"), strict=True):
        ax.bar(np.arange(len(results)), [100 * r["test"][metric] for r in results])
        ax.set_xticks(np.arange(len(results)), labels, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
    fig.savefig(root / "test_metrics.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("local", "restart", "composite"), default="local")
    parser.add_argument("--variant", choices=("quadrant24", "cells8", "pixels16"), default="pixels16")
    parser.add_argument("--appearance-holdout", action="store_true")
    parser.add_argument("--appearance-pool-size", type=int, default=None,
                        help="Appearances per split: defaults to 8 for quadrant24 holdout, otherwise 256")
    parser.add_argument("--appearance-seed-offset", type=int, default=3000,
                        help="Independent appearance-pool seed offset, fixed before fitting")
    parser.add_argument("--min-distance", type=int, default=0)
    parser.add_argument("--palette", choices=("fixed", "shared"), default="fixed")
    parser.add_argument("--placement", choices=("aligned", "offgrid"), default="aligned")
    parser.add_argument("--representation", choices=("scene", "objects"), default="scene")
    parser.add_argument("--object-counts", type=int, nargs="+", default=[4])
    parser.add_argument("--models", nargs="+", choices=("mean", "sum", "coord_mean", "coord_sum", "abmil", "vit5", "transmil"),
                        default=["mean", "coord_mean", "abmil", "vit5"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2026])
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--test-seed-offset", type=int, default=6000,
                        help="Independent final-test image seed offset, fixed before fitting")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restart-at", type=int, default=2000)
    parser.add_argument("--resume-prefix", type=Path, default=None,
                        help="Reuse a saved restart-control prefix directory without repeating its updates")
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=None, help="Override pooling width only for diagnostics")
    parser.add_argument("--transmil-landmarks", type=int, default=32,
                        help="Explicit Nyström landmark count for the short-sequence TransMIL baseline")
    parser.add_argument("--canvas-size", type=int, default=96)
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--scene-layout-size", type=int, default=None,
                        help="Fix scene origins to this lattice when sweeping padding")
    parser.add_argument("--coordinate-scale", type=int, default=None,
                        help="Keep coordinate normalization/RoPE fixed during a padding sweep")
    parser.add_argument("--rule", choices=("or", "majority", "exact_k", "parity"), default="or")
    parser.add_argument("--exact-k", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mondriaan_benchmark"))
    args = parser.parse_args()
    if args.appearance_pool_size is None:
        args.appearance_pool_size = 8 if args.variant == "quadrant24" and args.appearance_holdout else 256
    if args.appearance_pool_size < 1:
        parser.error("appearance-pool-size must be positive")
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("batch-size must be positive and even")
    if args.steps < 1 or args.eval_every < 1 or args.threads < 1:
        parser.error("steps, eval-every, and threads must be positive")
    if args.task == "restart" and not 0 < args.restart_at < args.steps:
        parser.error("restart-at must be strictly between zero and total steps")
    if args.resume_prefix and (args.task != "restart" or len(args.models) != 1 or len(args.seeds) != 1):
        parser.error("resume-prefix requires restart task with one model and one seed")
    if args.task == "composite" and args.representation != "scene":
        parser.error("Composite full-canvas models use scene representation")
    if args.task == "composite" and args.object_counts != [4]:
        parser.error("Composite constituents currently contain exactly four objects")
    if args.transmil_landmarks < 1:
        parser.error("transmil-landmarks must be positive")
    if "transmil" in args.models and args.representation != "scene":
        parser.error("TransMIL spatial baseline requires the complete scene patch grid")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    protocol = {**vars(args), "output_dir": str(root), "device": "cpu", "precision": "float32",
                "resume_prefix": str(args.resume_prefix) if args.resume_prefix else None,
                "test_policy": "generated after model/checkpoint selection; no test tuning",
                "selection": "validation worst-case, binary macro-F1, accuracy",
                "parameter_policy": "pooling widths nearest ViT parameter count; ABMIL reference retained",
                "local_rule": "exactly one identity occurs twice; every other identity occurs once"}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        parser.error("Output directory already has a frozen protocol; choose a new directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(args.threads)
    pending = []
    creators = {}
    for seed in args.seeds:
        pools, training, validation, create_test = prepare_task(args, seed)
        creators[seed] = (pools, training, validation, create_test)
        scene_size = args.coordinate_scale or (args.canvas_size if args.task == "composite" else 32)
        for name in args.models:
            torch.manual_seed(seed)
            model = build_model(name, scene_size, args.width, args.transmil_landmarks)
            settings = {**protocol, "seed": seed, "model": name,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "coordinate_scale": scene_size}
            directory = root / f"{name}_seed{seed}"
            if args.task == "restart":
                # One identical prefix, then branches differing only in AdamW state.
                if args.resume_prefix:
                    source = args.resume_prefix
                    saved = torch.load(source / "final.pt", weights_only=True)
                    required = ("optimizer_state", "sampler_state", "torch_rng_state", "model_state")
                    if any(key not in saved for key in required) or saved["step"] != args.restart_at:
                        raise ValueError("Prefix lacks required training state or has wrong update count")
                    comparison = ("model", "seed", "variant", "palette", "placement", "representation",
                                  "object_counts", "appearance_pool_size", "appearance_holdout", "min_distance",
                                  "train_size", "validation_size", "test_size", "batch_size", "lr", "coordinate_scale")
                    for key in comparison:
                        if saved["settings"].get(key) != settings.get(key):
                            raise ValueError(f"Incompatible resume prefix setting: {key}")
                    prefix_report = json.loads((source / "training.json").read_text())
                    (directory / "prefix").mkdir(parents=True, exist_ok=True)
                    for filename in ("final.pt", "best.pt", "training.json"):
                        shutil.copyfile(source / filename, directory / "prefix" / filename)
                else:
                    prefix_report = train(model, training, validation, seed=seed, steps=args.restart_at,
                          batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                          output_dir=directory / "prefix", settings=settings)
                state = torch.load(directory / "prefix/final.pt", weights_only=True)
                for arm in ("uninterrupted", "restart"):
                    branch = build_model(name, scene_size, args.width, args.transmil_landmarks)
                    report = train(branch, training, validation, seed=seed, steps=args.steps,
                                   batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                                   output_dir=directory / arm, settings={**settings, "arm": arm},
                                   initial_state=state, restart_optimizer=arm == "restart",
                                   inherited_selection={"rank": prefix_report["selected_rank"],
                                                        "step": prefix_report["selected_step"],
                                                        "checkpoint": directory / "prefix/best.pt"})
                    pending.append((seed, name, arm, directory / arm, report))
            else:
                report = train(model, training, validation, seed=seed, steps=args.steps,
                               batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                               output_dir=directory, settings=settings)
                pending.append((seed, name, None, directory, report))
    # This is the first construction/evaluation of fresh final test images.
    tests = {}
    for seed, (pools, training, validation, create_test) in creators.items():
        from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import audit_splits
        test = create_test()
        if args.task == "composite":
            test["case_mixes"] = test["cases"]
            test["cases"] = [f"positive_scene_count_{n}" for n in test["positive_scene_counts"]]
        tests[seed] = tensor_data(test)
        tests[seed]["intended_cases"] = planned_cases(args)
        if args.task != "composite":
            from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import dataset_checks, patchify
            # Object diagnostics retain raw images. Verify the full patch
            # representation independently rather than treating crops as grids.
            raw_test = dict(test)
            raw_test["x"] = np.stack([patchify(image)[0] for image in test["images"]])
            raw_test["coords"] = np.stack([patchify(image)[1] for image in test["images"]])
            test_checks = dataset_checks(raw_test)
            checks = audit_splits({"train": training, "validation": validation, "test": test},
                                  require_disjoint_appearances=(args.variant != "quadrant24" or args.appearance_holdout))
            (root / f"data_checks_seed{seed}.json").write_text(json.dumps(
                {"test": test_checks, "overlap": checks}, indent=2) + "\n")
        else:
            def hashes(data):
                return {hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]}
            def constituents(data):
                return set(data["scene_ids"].ravel())
            checks = {}
            for name, development in (("train", training), ("validation", validation)):
                image_overlap = len(hashes(development) & hashes(test))
                scene_overlap = len(constituents(development) & constituents(test))
                checks[name] = {"canvas_image_overlap": image_overlap, "constituent_scene_overlap": scene_overlap}
                if image_overlap or scene_overlap:
                    raise ValueError(f"Composite test leaks {name} scenes or canvases")
            (root / f"data_checks_seed{seed}.json").write_text(json.dumps(checks, indent=2) + "\n")
        image_hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in test["images"]]
        (root / f"test_manifest_seed{seed}.json").write_text(json.dumps(
            {"split": "test", "samples": len(image_hashes), "image_sha256": image_hashes}, indent=2) + "\n")
    results = []
    for seed, name, arm, directory, report in pending:
        size = args.coordinate_scale or (args.canvas_size if args.task == "composite" else 32)
        model = build_model(name, size, args.width, args.transmil_landmarks)
        checkpoint = torch.load(directory / "best.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test = tests[seed]
        inference_started = time.perf_counter()
        metrics, predictions = evaluate(model, test, args.batch_size, return_predictions=True)
        inference_seconds = time.perf_counter() - inference_started
        final_state = torch.load(directory / "final.pt", weights_only=True)
        model.load_state_dict(final_state["model_state"])
        final_metrics = evaluate(model, test, args.batch_size)
        result = {"model": name, "seed": seed, "selected_step": report["selected_step"],
                  "parameters": report["parameters"], "test": metrics,
                  "test_final_update": final_metrics, "seconds": report["seconds"],
                  "tokens": test["x"].shape[1],
                  "test_inference_seconds": inference_seconds,
                  "test_images_per_second": len(test["y"]) / inference_seconds,
                  "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                  * (1 if sys.platform == "darwin" else 1024),
                  "memory_scope": "whole process high-water RSS; includes data and earlier runs, not isolated model peak",
                  "test_sample_uncertainty": grouped_test_uncertainty(test, predictions, seed)}
        if arm:
            result["arm"] = arm
        if args.task == "composite":
            from sparse_wsi_vit.datasets.mondriaan_dataset.composite import composite_subgroup_metrics, local_label_oracle
            result["canvas_subgroups"] = composite_subgroup_metrics(predictions, {**test, "cases": test["case_mixes"]})
            result["local_label_oracle_accuracy"] = float(np.mean(local_label_oracle(test) == test["y"].numpy()))
        (directory / "test.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(f"FINAL {name}/seed{seed}/{arm or 'local'} "
              f"test={metrics['accuracy']:.3f} worst={metrics['worst_case_accuracy']:.3f}", flush=True)
    write_results(root, results)


if __name__ == "__main__":
    main()
