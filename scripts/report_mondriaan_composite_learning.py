"""Render the measured frozen composite/learned-local diagnostic report."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pct(value):
    return f"{100 * value:.1f}%"


def _range(values):
    return f"{_pct(min(values))}–{_pct(max(values))}"


def create_report(benchmark, diagnostics, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    full = json.loads((benchmark / "results.json").read_text())
    frozen = json.loads((diagnostics / "diagnostics.json").read_text())
    local = [row for row in frozen if row["method"] == "local_mean_then_or"]
    base = [row for row in frozen if row["sweep"] == "padding" and row["canvas_size"] == 80]
    model_order = ["mean", "coord_mean"]
    base.sort(key=lambda row: (model_order.index(row["method"]), [42, 7, 2026].index(row["seed"])))
    cases = [f"positive_scene_count_{k}" for k in range(5)]
    values = np.array([[row["test"]["by_case"][case]["accuracy"] for case in cases] for row in base])
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, _pct(values[i, j]), ha="center", va="center", color="white")
    ax.set_xticks(range(5), range(5))
    ax.set_yticks(range(len(base)), [f"{r['method']} / seed {r['seed']}" for r in base])
    ax.set_xlabel("True positive-scene count (zero is the negative canvas class)")
    ax.set_title("Selected S=80, M=4 models: held-out count subgroup accuracy")
    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.savefig(report_dir / "mondriaan_composite_learning_counts.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7), constrained_layout=True)
    colors = {"mean": "#2366a4", "coord_mean": "#c66c28"}
    for column, (sweep, key, title) in enumerate([
            ("padding", "canvas_size", "Same canvases with added black padding"),
            ("scene_count", "scene_count", "Changed scene count at canvas S=128")]):
        for method in model_order:
            selected = [r for r in frozen if r["sweep"] == sweep and r["method"] == method]
            xs = sorted({r[key] for r in selected})
            for row, metric in enumerate(["accuracy", "worst_case_accuracy"]):
                by_x = [[r["test"][metric] for r in selected if r[key] == x] for x in xs]
                average = [100 * np.mean(values) for values in by_x]
                low, high = [100 * min(v) for v in by_x], [100 * max(v) for v in by_x]
                axes[row, column].plot(xs, average, "o-", label=method, color=colors[method])
                axes[row, column].fill_between(xs, low, high, alpha=.15, color=colors[method])
                axes[row, column].set_ylim(-2, 102)
                axes[row, column].set_xticks(xs)
                axes[row, column].set_xlabel("Canvas size S" if column == 0 else "Scenes M")
                axes[row, column].set_ylabel("Overall accuracy (%)" if row == 0 else "Worst count accuracy (%)")
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(title)
        axes[0, column].legend()
    fig.suptitle("Frozen checkpoints trained only at S=80, M=4; shading = range across three seeds")
    fig.savefig(report_dir / "mondriaan_composite_learning_scaling.png", dpi=160)
    plt.close(fig)
    full_table = "\n".join(f"| {r['method']} | {r['seed']} | {r['selected_step']} | "
        f"{_pct(r['test']['accuracy'])} | {_pct(r['test']['worst_case_accuracy'])} | "
        f"{_pct(r['test']['macro_case_accuracy'])} | {_pct(r['test']['positive_f1'])} |"
        for r in base)
    local_table = "\n".join(f"| {r['seed']} | {r['selected_step']} | {_pct(r['local_test']['accuracy'])} | "
        f"{_pct(r['local_test']['worst_case_accuracy'])} | {_pct(r['test']['accuracy'])} | "
        f"{_pct(r['test']['worst_case_accuracy'])} | {r['test']['confusion_matrix'][0][1]} | "
        f"{r['test']['confusion_matrix'][1][0]} |" for r in local)
    scale_table = []
    for sweep, xs, key in [("padding", [80, 128, 256], "canvas_size"),
                           ("scene_count", [1, 4, 9], "scene_count")]:
        for x in xs:
            for method in model_order:
                selected = [r for r in frozen if r["sweep"] == sweep and r[key] == x and r["method"] == method]
                scale_table.append(f"| {sweep} | {selected[0]['canvas_size']} | {selected[0]['scene_count']} | "
                    f"{method} | {selected[0]['tokens']} | {_range([r['test']['accuracy'] for r in selected])} | "
                    f"{_range([r['test']['worst_case_accuracy'] for r in selected])} | "
                    f"{min(r['inference_seconds'] for r in selected):.3f}–{max(r['inference_seconds'] for r in selected):.3f} |")
    scale_table = "\n".join(scale_table)
    parameters = {r["model"]: r["parameters"] for r in full}
    max_rss = max(r["process_cumulative_peak_rss_bytes"] for r in frozen) / 1024 ** 3
    report = f"""# Mondriaan composite learning: early pooling and frozen scaling probes

At the tested budget, neither learned pooling model solves the composite task.
Coordinate-free mean pooling scores 48.0–51.2% on new S=80/M=4 canvases;
coordinate-aware mean pooling scores 43.0–50.0%. Adding padding makes every
selected checkpoint a constant predictor at S=256. A learned local-model→OR
diagnostic also remains near chance because local false positives propagate
through OR. These are early learning probes, not evidence that longer training,
stronger local encoders or hierarchical pooling cannot work.

## Protocol and controls

The frozen parent protocol trains 512 canvases, selects on 128 validation
canvases and evaluates 256 separate test canvases, independently for seeds
42, 7 and 2026. Each split has its own 256-identity pixel-pattern pool, with
four pixels of each of four colors in every 4×4 object. Appearance pools,
canonical constituent scenes, local rendered images and final canvas images
are disjoint across training/validation/test; all overlap audits passed.
Small scenes are split before composition, and final test canvases are generated
after all six full-canvas fits finish. The source-scene bank sizes match the
canvas sizes: 512/128/256 scenes in the three splits.

The canvas label is OR of local strict-one-pair labels. Canvas classes are
50/50; negative canvases contain zero positive scenes, while positive-scene
counts 1,2,3,4 occur equally often. Local scene layouts and palettes are exactly
matched within opposite-label canvas pairs. A regular 32×32 spatial lattice
with four-pixel gutters defines membership; scene identities can recur across
regions, and no corner, identity or region-ID metadata enters full-canvas models.
This is a grouped MIL detection control, not a demonstration that global
self-attention is necessary. All scene objects in this probe align to patches.

Both models use an arrangement-sensitive per-token MLP and nonlinear classifier.
The coordinate-aware version appends (x/80,y/80). Parameter counts are
{parameters['mean']:,} (mean) and {parameters['coord_mean']:,} (coordinate mean),
close to the existing 2,381,954-parameter ViT. Full-canvas fitting uses 400 AdamW
updates, learning rate 2e-4, batch size 32, CPU float32 and one PyTorch thread.
Validation selects by worst positive-scene-count accuracy, then binary macro-F1,
then overall accuracy, from updates 0,100,200,300,400. No checkpoint or threshold
is selected using test data. This short schedule is not a training-limit study.

## Full-canvas result at the trained scale

| Model | Seed | Selected update | Overall | Worst count subgroup | Macro count | Positive F1 |
|---|---:|---:|---:|---:|---:|---:|
{full_table}

![Per-count test accuracy](mondriaan_composite_learning_counts.png)

The test has 128 count-zero canvases and 32 in each positive-count subgroup.
Case-mixture diagnostics and both binary-class metrics are retained in JSON and
CSV. Small case-mixture cells sometimes contain only one canvas: their observed
minimum is diagnostic, not a stable population estimate. Test uncertainty uses
128 matched layout-pair bootstrap groups, following explicit shuffled
layout_pair_ids. These intervals condition on a fitted checkpoint. The three
seed ranges shown here are training/data-seed variation, not confidence intervals;
constituent scenes also recur within a split, adding dependence beyond pair grouping.

## Frozen padding and scene-count diagnostics

The selected S=80/M=4 checkpoints are frozen. Padding evaluations use exactly
the same test pixels, scene identities, palettes, nuisance pairs and origins at
S=80,128,256; only additional black space changes. Base images were verified
against the original test SHA256 manifest. Coordinate normalization stays fixed
at 80, so added background coordinates extend beyond the training range, and
mean pooling also changes the foreground/background proportion. This tests
robustness to additional context; it does not retune positional scales.

The separate count sweep creates M=1,4,9 canvases at S=128 from the same held-out
scene bank. The M=4 version equals the padded baseline pixel-for-pixel. The
aggregation rule remains balanced OR, but the count distribution and number of
local decisions change. M=9 also occupies extra foreground regions outside the
four spatial slots observed in training. M=1/M=9 scores describe transfer to a changed task
demand, not separately trained learning curves. No scale result tunes a model.

| Sweep | Canvas S | Scenes M | Model | Tokens | Overall across seeds | Worst count across seeds | Inference seconds/256 canvases |
|---|---:|---:|---|---:|---:|---:|---:|
{scale_table}

![Frozen scaling](mondriaan_composite_learning_scaling.png)

At S=256 all mean checkpoints predict negative on every canvas; coordinate
mean predicts one constant class in each seed. The resulting 50% accuracy
conceals 0% accuracy on at least one count subgroup. The trained-scale scores
are already weak, so these probes show padding sensitivity of these checkpoints
rather than a scaling limit of otherwise successful models.

Inference timings use CPU, batches of 32 and include the model forward pass and
metric calculation, excluding scene generation. They are one measured pass,
without a statistical timing benchmark. Process cumulative peak RSS reaches
{max_rss:.2f} GiB while retaining multiple data views and evaluating successive
models; this is not isolated model/activation memory. The pooling implementation
encodes identical background values efficiently while preserving their exact
multiplicity. Timings therefore do not predict full-attention runtime.

## Learned local-model → OR diagnostic

A separate mean pooling model is fitted for 2,000 updates per seed on exactly
the same 512 constituent training scenes and 128 constituent validation scenes,
verified by their complete image-hash lists. It selects using the five local
repetition-case metrics at updates 0,400,800,1200,1600,2000. After selection, it
reads each prescribed region in the same locked composite test canvases, then
the exact OR rule aggregates its binary predictions. The grouping/cropping is
supplied in this diagnostic; it does not test learned region discovery. The
five-times-longer local budget is intentionally diagnostic, not a fair overall
training-budget comparison with full-canvas models.

| Seed | Local selected update | Local accuracy | Local worst case | Canvas accuracy | Canvas worst count | Canvas FP / 128 negatives | Canvas FN / 128 positives |
|---|---:|---:|---:|---:|---:|---:|---:|
{local_table}

Local predictions are evaluated over 1,024 scene occurrences: 320 positives
and 704 negatives, reflecting controlled OR sampling rather than a balanced
local test. A negative canvas becomes positive as soon as any of its four local
predictions is a false positive. The diagnostic falsely accepts 116–122 of 128
negative canvases; its positive recall remains 90.6–95.3%. Its near-50% canvas
score therefore does not show useful global aggregation despite high positive
recall. Every wrong canvas has at least one wrong local label; applying f to
fully correct local predictions introduces zero errors.

There is a further controlled-input limitation to this first diagnostic: local
training palettes are sampled in [.1,.9] with minimum RGB-color separation .2,
while composite global palettes use [.1,.95] without that separation restriction.
The same constituent patterns/layouts are recolored for global palette sharing,
so local inference has a mild palette-distribution shift. A follow-up should
match the two palette distributions before attributing the complete local
failure to matching alone. Current results do not isolate that contribution.

## Grouping counterexample and interpretation

The verified AABB+CCDD versus AABC+BCDD fixture has opposite OR labels but
identical global identity counts and coordinate-free patch bags. All three
learned coordinate-free mean checkpoints produce equal positive probabilities
for the two canvases, to numerical precision, and predict negative on both.
Coordinate-aware mean gives slightly different logits but the same class to both
at this budget. The fixture establishes an information loss when coordinates
and scene grouping are removed; its coordinate-aware result is a learning
failure at a short budget. A correctly functioning local classifier followed by
OR is still a complete solution, as the 100% oracle audit demonstrates.

Next comparisons should first establish a robust local appearance matcher,
align palette distributions, then compare hierarchical and full-canvas models
under explicit budgets. Off-grid scenes, learned attention models, broader
optimization and adequate-training scaling are not measured here.

Artifacts: [full-canvas CSV](mondriaan_composite_learning_summary.csv),
[frozen/local CSV](mondriaan_composite_learning_diagnostics.csv),
[complete frozen diagnostics](../runs/mondriaan_benchmark_composite_diagnostics/diagnostics.json),
[scaling/split checks](../runs/mondriaan_benchmark_composite_diagnostics/scaling_checks.json),
[fixture predictions](../runs/mondriaan_benchmark_composite_diagnostics/fixture_predictions.json),
[parent protocol](../runs/mondriaan_benchmark_composite/protocol.json),
[local protocol](../runs/mondriaan_benchmark_composite_diagnostics/local_protocol.json).
"""
    (report_dir / "mondriaan_composite_learning.md").write_text(report)
    shutil.copyfile(benchmark / "summary.csv", report_dir / "mondriaan_composite_learning_summary.csv")
    shutil.copyfile(diagnostics / "summary.csv", report_dir / "mondriaan_composite_learning_diagnostics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("runs/mondriaan_benchmark_composite"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("runs/mondriaan_benchmark_composite_diagnostics"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    create_report(args.benchmark_dir, args.diagnostics_dir, args.report_dir)


if __name__ == "__main__":
    main()
