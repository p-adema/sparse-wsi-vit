"""Bounded CPU compositor/oracle audit; does not train full-canvas models.

Run: PYTHONPATH=src .venv/bin/python scripts/probe_mondriaan_composite.py \
    --output-dir reports/composite_smoke

The final test split here is used to verify generation contracts only. It must
not become the final learned-model test after development choices use its scores.
"""

import argparse
from collections import Counter
import csv
import gc
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.composite import (
    build_composite_splits,
    composite_subgroup_metrics,
    local_label_oracle,
    lattice_origins,
    opposite_label_global_count_fixture,
    predict_local_then_aggregate,
    scene_pool_from_data,
    verify_composite,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    make_appearance_pools,
    make_controlled_dataset,
)


def _process_highwater_mb():
    # macOS reports bytes, Linux reports KiB. This is cumulative process peak,
    # not model-memory consumption or a per-condition peak delta.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == "darwin" else 1024)


def _aligned_pixel_local_oracle(x, coords):
    """Read actual RGB patch equality; no local labels or appearance IDs supplied."""
    if coords.shape[1:] != (64, 2):
        raise ValueError("Expected the complete local 32x32 scene")
    results = []
    for sample in x:
        objects = [token.tobytes() for token in sample if not np.all(token == -1)]
        if len(objects) != 4:
            raise ValueError("This diagnostic assumes aligned complete-object patches")
        results.append(int(sorted(Counter(objects).values()) == [1, 1, 2]))
    return np.asarray(results)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sample_figure(data, output):
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    pair_indices = np.flatnonzero(data["layout_pair_ids"] == data["layout_pair_ids"][0])
    for target, ax in enumerate(axes):
        i = pair_indices[np.flatnonzero(data["y"][pair_indices] == target)[0]]
        ax.imshow(data["images"][i], interpolation="nearest")
        ax.set_title(f"OR label {target}; {data['positive_scene_counts'][i]} positive scenes")
        ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _fixture_figure(fixture, output):
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, ax in enumerate(axes):
        ax.imshow(fixture["images"][i], interpolation="nearest")
        ax.set_title(["AABB + CCDD: OR = 0", "AABC + BCDD: OR = 1"][i])
        ax.set_axis_off()
    figure.suptitle("Identical global identity counts and coordinate-free patch bags")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_audit(output, *, samples=32, scene_pool_size=64, appearance_pool_size=32, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    appearances = make_appearance_pools("pixels16", seed=seed,
        sizes={split: appearance_pool_size for split in ["train", "validation", "test"]})
    pools = {split: scene_pool_from_data(make_controlled_dataset(pool, scene_pool_size,
                 seed=seed + 100 + i, object_counts=(4,), aligned=True, palette_mode="fixed"))
             for i, (split, pool) in enumerate(appearances.items())}
    specifications = [("padding", size, 4, "or", "fixed") for size in [80, 128, 256]]
    specifications += [("scene_count", 128, count, "or", "fixed") for count in [1, 4, 9]]
    specifications += [("aggregation", 80, 4, rule, "fixed")
                       for rule in ["majority", "exact_k", "parity"]]
    specifications += [("palette", 80, 4, "or", "shared_per_canvas")]
    summary, diagnostics, checks = [], {}, {}
    for experiment, size, m, rule, palette in specifications:
        condition = f"{experiment}_{rule}_S{size}_M{m}_{palette}"
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        datasets, overlap = build_composite_splits(pools, n_samples=samples,
            canvas_size=size, scene_count=m, rule=rule, exact_k=1,
            palette_mode=palette, seed=seed,
            fixed_origins=lattice_origins(80, 4) if experiment == "padding" else None)
        generation_seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        checks[condition] = {"split_overlap": overlap, "splits": {}}
        diagnostics[condition] = {}
        if experiment == "padding" and size == 80:
            _sample_figure(datasets["validation"], output / "balanced_or_samples.png")
        for split, data in datasets.items():
            start = time.perf_counter()
            check = verify_composite(data, pools[split])
            check_seconds = time.perf_counter() - start
            checks[condition]["splits"][split] = check
            diagnostics[condition][split] = {}
            predictors = {"verified_local_label_oracle": lambda: local_label_oracle(data),
                          "local_pixel_equality_then_f": lambda: predict_local_then_aggregate(
                              data, _aligned_pixel_local_oracle)[1],
                          "always_positive": lambda: np.ones_like(data["y"]),
                          "always_negative": lambda: np.zeros_like(data["y"])}
            for method, predict in predictors.items():
                start = time.perf_counter()
                predicted = predict()
                diagnostic_seconds = time.perf_counter() - start
                metrics = composite_subgroup_metrics(predicted, data)
                diagnostics[condition][split][method] = metrics
                summary.append({"condition": condition, "sweep": experiment, "split": split,
                    "rule": rule, "palette_mode": palette, "canvas_size": size,
                    "scene_count": m, "samples": samples, "token_count": check["token_count"],
                    "foreground_fraction": check["foreground_fraction"], "method": method,
                    "accuracy": metrics["accuracy"],
                    "worst_subgroup_accuracy": metrics["worst_subgroup_accuracy"],
                    "macro_count_accuracy": metrics["macro_count_accuracy"],
                    "generation_and_initial_audit_seconds_all_splits": generation_seconds,
                    "verification_seconds_this_split": check_seconds,
                    "diagnostic_seconds_this_split": diagnostic_seconds,
                    "composite_arrays_mb_this_split": sum(v.nbytes for v in data.values()
                                                          if isinstance(v, np.ndarray)) / 1024 ** 2,
                    "tracemalloc_generation_peak_mb_all_splits": peak / 1024 ** 2,
                    "process_cumulative_highwater_mb": _process_highwater_mb()})
        del datasets
    fixture = opposite_label_global_count_fixture()
    _fixture_figure(fixture, output / "global_count_counterexample.png")
    checks["global_count_counterexample"] = {
        "canvas_labels": fixture["y"].tolist(), "local_labels": fixture["local_labels"].tolist(),
        "identical_global_identity_counts": True, "identical_coordinate_free_patch_bags": True,
        "identical_foreground_layout": True, "different_spatial_images": True,
        "scope": "Analytic verified fixture, not learned-model performance"}
    (output / "generation_checks.json").write_text(json.dumps(checks, indent=2,
        default=_json_default) + "\n")
    (output / "subgroup_diagnostics.json").write_text(json.dumps(diagnostics, indent=2,
        default=_json_default) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _write_report(output, summary, samples, scene_pool_size, appearance_pool_size, seed)
    return summary


def _write_report(output, rows, samples, scene_pool_size, appearance_pool_size, seed):
    selected = [row for row in rows if row["split"] == "validation"
                and row["method"] == "local_pixel_equality_then_f"
                and row["sweep"] in {"padding", "scene_count"}]
    table = "\n".join(f"| {r['sweep']} | {r['canvas_size']} | {r['scene_count']} | "
                       f"{r['token_count']} | {r['foreground_fraction']:.4%} | "
                       f"{r['accuracy']:.1%} | {r['worst_subgroup_accuracy']:.1%} | "
                       f"{r['composite_arrays_mb_this_split']:.2f} | "
                       f"{r['generation_and_initial_audit_seconds_all_splits']:.3f} |"
                       for r in selected)
    report = f"""# Mondriaan composite canvas: bounded generation and oracle audit

This run verifies composition, labels, split isolation and diagnostics. It does
not train or evaluate a learned full-canvas model. Seed {seed}; {samples} canvases
per split and condition, built from {scene_pool_size} distinct constituent
scenes per split and {appearance_pool_size} explicitly disjoint pixel-arrangement
identities per split. Every object has four pixels of each shared palette color.

The primary task is balanced OR: a canvas is positive iff at least one local
32×32 scene contains exactly one pair (AABC). Negatives have zero positive
scenes. Positives cycle evenly through counts 1..M, then choose positive regions
uniformly conditional on count. The generator therefore supplies 50/50 canvas
labels instead of the 1 − 2^(-M) prevalence from independent fair local labels.
This sampling deliberately changes the local-label distribution; the exact
observed counts and negative-case mixtures are saved in generation_checks.json.
Local negative cases are inherited from the verified layout-matched scene bank,
not forced to independent uniform case mixtures in each canvas.

Scene grouping uses a regular 32×32 spatial lattice with a four-pixel gutter.
Origins are chosen without reference to labels, and every opposite-label pair
has exactly the same origins, local foreground masks and palette. There are no
boundary-marker pixels or extra region-ID model features. Coordinates allow a
model to recover the prescribed grouping. One palette is global to the canvas;
identities may recur across regions. Shared-per-canvas palettes and majority,
exactly-one-positive, and parity rules are also exercised at S=80, M=4.

Constituent scenes are split before composition. The high-level split builder
rejects shared canonical scenes (even after recoloring), shared RGB images and
shared arrangements across splits before rendering canvases. It then rechecks
constituent and exact canvas-image overlaps. All observed overlaps are zero.
Every local label is recomputed from rendered object pixels, and the canvas
label is recomputed with f. Patch tokenization preserves all pixels and retains
all background patches with the existing -1 sentinel.

| Sweep | Canvas S | Scenes M | Tokens | Foreground | Pixel-local→f | Worst observed subgroup | Stored arrays MiB/split | Build+audit seconds/3 splits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

Canvas size is swept at fixed M=4 with identical scene identities and placements
at each S: the only image change is adding black padding. Scene count is swept
separately at S=128. Canvas arrays expose layout_pair_ids after the final shuffle
so uncertainty estimates can resample complete positive/negative matched pairs.
The coordinate grid spans 0..S−4 in four-pixel steps. Recommended RoPE high is
S, with normalized neighboring-coordinate difference 4/S. This checks the
coordinate contract; it is not a claim about learned RoPE behavior or robustness.
Stored arrays include full canvas images/tokens and local inputs, not model
activations. CSV generation times include initial verification and all three
splits. Python allocation peaks and cumulative process RSS high-water marks are
saved separately; the latter cannot be interpreted as a condition-specific
model-memory peak. All timings are CPU generation/diagnostic measurements.

The verified local-label oracle and actual RGB local-pixel equality→f diagnostic
both achieve 100% on every observed label/count/case-mixture subgroup. Always
positive and always negative score 50% overall and 0% on a label subgroup. These
results establish that the controlled data and aggregation hooks agree. They
say nothing about whether a neural model can learn the rule.

![Balanced OR canvases](balanced_or_samples.png)

![Counterexample to global identity counts](global_count_counterexample.png)

AABB+CCDD has local labels (0,0); AABC+BCDD has (1,1). Both contain two objects
of each identity A,B,C,D and have identical foreground layouts and identical
coordinate-free patch bags, but opposite OR labels. Thus spatial grouping is
necessary for this example. A coordinate-free patch pooling model has lost
necessary input information. Coordinate-aware or explicitly hierarchical pooling
can solve the grouped task: OR does not require full-canvas attention.

Next measured comparisons must include learned local-model→f and full-canvas
models under comparable capacity/update budgets, with validation selection and
a newly locked learned-model test. Off-grid composite objects, spatial relations
beyond count rules and learned scaling remain unmeasured. This small audit's
test split verifies contracts; it should not be reused as a final test after
development choices consume learned-model results on it.

Artifacts: [CSV](summary.csv), [generation checks](generation_checks.json),
[subgroup diagnostics](subgroup_diagnostics.json).
"""
    (output / "report.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--scene-pool-size", type=int, default=64)
    parser.add_argument("--appearance-pool-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2 or args.scene_pool_size < 8 or args.scene_pool_size % 8:
        parser.error("samples must be even and scene-pool-size a positive multiple of eight")
    rows = run_audit(args.output_dir, samples=args.samples, scene_pool_size=args.scene_pool_size,
                     appearance_pool_size=args.appearance_pool_size, seed=args.seed)
    print(json.dumps({"conditions": len({r["condition"] for r in rows}), "rows": len(rows),
                      "output_dir": str(args.output_dir), "learned_models_run": False}))


if __name__ == "__main__":
    main()
