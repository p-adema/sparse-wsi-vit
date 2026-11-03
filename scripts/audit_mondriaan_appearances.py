"""Bounded CPU-only data and scalar-shortcut audit for new Mondriaan variants.

Thresholds/intervals are fitted on training data only. Validation and fresh test
data use separate seeds and explicitly separated appearance dictionaries where
specified. This audit does not train or claim performance of learned models.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    audit_splits,
    dataset_checks,
    distance_diagnostics,
    make_appearance_pools,
    make_controlled_dataset,
)


def rgb_diversity(data: dict, *, normalize_palette: bool = False) -> np.ndarray:
    """Mean pairwise squared RGB distance, averaged across all 16 pixel sites."""
    result = []
    for patterns, palette in zip(data["patterns"], data["palettes"], strict=True):
        objects = palette[patterns].reshape(len(patterns), 16, 3).astype(np.float64)
        distances = [np.square(objects[i] - objects[j]).sum(axis=-1).mean()
                     for i in range(len(objects)) for j in range(i)]
        value = np.mean(distances)
        if normalize_palette:
            colours = palette.astype(np.float64)
            # Expected squared distance of two independent uniform palette
            # colours; computable from any complete object's histogram.
            scale = 2 * np.square(colours - colours.mean(axis=0)).sum(axis=1).mean()
            value /= scale
        result.append(value)
    return np.asarray(result)


def hamming_diversity(data: dict) -> np.ndarray:
    """Mean fraction of different colour-index pixels, using complete crops."""
    return np.asarray([np.mean([np.mean(patterns[i] != patterns[j])
                                for i in range(len(patterns)) for j in range(i)])
                       for patterns in data["patterns"]])


def _groups(scores, labels):
    unique, inverse = np.unique(scores, return_inverse=True)
    positive = np.bincount(inverse, weights=labels, minlength=len(unique)).astype(int)
    total = np.bincount(inverse, minlength=len(unique))
    return unique, positive, total - positive


def fit_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Fit both directions, including constant predictions; no heldout data."""
    unique, positive, negative = _groups(scores, labels)
    thresholds = np.r_[-np.inf, (unique[:-1] + unique[1:]) / 2, np.inf]
    cumulative_pos = np.r_[0, np.cumsum(positive)]
    cumulative_neg = np.r_[0, np.cumsum(negative)]
    lower_correct = cumulative_pos + negative.sum() - cumulative_neg
    upper_correct = positive.sum() - cumulative_pos + cumulative_neg
    combined = np.r_[lower_correct, upper_correct]
    best = int(np.argmax(combined))
    lower = best < len(thresholds)
    index = best if lower else best - len(thresholds)
    return {"kind": "threshold", "direction": "le" if lower else "gt",
            "threshold": float(thresholds[index]),
            "training_accuracy": float(combined[best] / len(labels))}


def fit_interval(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Optimal contiguous positive score interval, found with a linear scan.

    Each score group has gain positives - negatives over an always-negative
    predictor. The maximum-sum contiguous segment gives the best interval.
    """
    unique, positive, negative = _groups(scores, labels)
    gains = positive - negative
    best_gain, current_gain, start = 0, 0, 0
    best_start, best_stop = 0, -1
    for stop, gain in enumerate(gains):
        current_gain += int(gain)
        if current_gain > best_gain:
            best_gain, best_start, best_stop = current_gain, start, stop
        if current_gain < 0:
            current_gain, start = 0, stop + 1
    if best_stop < best_start:
        low, high = np.inf, -np.inf
    else:
        low = (-np.inf if best_start == 0 else
               (unique[best_start - 1] + unique[best_start]) / 2)
        high = (np.inf if best_stop == len(unique) - 1 else
                (unique[best_stop] + unique[best_stop + 1]) / 2)
    return {"kind": "interval", "low": float(low), "high": float(high),
            "training_accuracy": float((negative.sum() + best_gain) / len(labels))}


def predict(rule: dict, scores: np.ndarray) -> np.ndarray:
    if rule["kind"] == "interval":
        return ((scores >= rule["low"]) & (scores <= rule["high"])).astype(int)
    return ((scores <= rule["threshold"]) if rule["direction"] == "le" else
            (scores > rule["threshold"])).astype(int)


def metrics(labels, predictions, cases) -> dict:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    cases = np.asarray(cases)
    by_case = {str(case): {"accuracy": float(np.mean(predictions[cases == case] ==
                                                    labels[cases == case])),
                          "samples": int(np.sum(cases == case))}
               for case in sorted(set(cases))}
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    def divide(n, d):
        return n / d if d else 0.0
    return {"accuracy": float(np.mean(labels == predictions)), "by_case": by_case,
            "worst_case_accuracy": min(v["accuracy"] for v in by_case.values()),
            "macro_case_accuracy": float(np.mean([v["accuracy"] for v in by_case.values()])),
            "positive_precision": divide(tp, tp + fp), "positive_recall": divide(tp, tp + fn),
            "positive_f1": divide(2 * tp, 2 * tp + fp + fn),
            "macro_f1": (divide(2 * tp, 2 * tp + fp + fn) +
                         divide(2 * tn, 2 * tn + fp + fn)) / 2,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def matched_pair_indices(data: dict) -> np.ndarray:
    """One-pair/all-distinct subset, keeping each exact label-paired nuisance."""
    def key(i):
        if "pair_ids" in data:
            return int(data["pair_ids"][i])
        return tuple(data["positions"][i]), data["palettes"][i].tobytes()
    positives = {key(i): i for i, y in enumerate(data["y"]) if y == 1}
    indices = []
    for i, case in enumerate(data["cases"]):
        if case == "all_distinct":
            indices.extend([positives[key(i)], i])
    return np.asarray(indices)


def distribution_matching_diagnostic(data: dict, scores: np.ndarray, *, bins: int = 8) -> dict:
    """Training-only feasibility audit for nuisance-preserving rejection sampling.

    Bins are frozen from positive *training* scores. The exact-zero atom has a
    separate bin, so coarse bins cannot conceal the all-matching impossibility.
    Whole paired layouts/palettes must be retained or rejected together.
    """
    if data["split"] != "train":
        raise ValueError("Matching-policy diagnostics must use training data only")
    labels = np.asarray(data["y"])
    boundaries = np.unique(np.quantile(scores[labels == 1], np.arange(1, bins) / bins))
    bin_ids = 1 + np.searchsorted(boundaries, scores, side="right")
    bin_ids[scores == 0] = 0
    def pair_key(i):
        if "pair_ids" in data:
            return int(data["pair_ids"][i])
        return tuple(data["positions"][i]), data["palettes"][i].tobytes()
    positives = {pair_key(i): i
                 for i, label in enumerate(labels) if label == 1}
    results = {}
    for case in sorted(set(data["cases"]) - {"one_pair"}):
        negative = np.flatnonzero(np.array(data["cases"]) == case)
        positive = np.array([positives[pair_key(i)] for i in negative])
        pos_hist = np.bincount(bin_ids[positive], minlength=len(boundaries) + 2) / len(positive)
        neg_hist = np.bincount(bin_ids[negative], minlength=len(boundaries) + 2) / len(negative)
        results[case] = {"paired_samples": len(negative),
                         "exact_same_score_pairs": int(np.sum(scores[positive] == scores[negative])),
                         "same_bin_pairs": int(np.sum(bin_ids[positive] == bin_ids[negative])),
                         "paired_bin_retention": float(np.mean(bin_ids[positive] == bin_ids[negative])),
                         "unpaired_binned_common_mass": float(np.minimum(pos_hist, neg_hist).sum()),
                         "negative_exact_zero_count": int(np.sum(scores[negative] == 0))}
    return {"fitted_on": "training only", "positive_quantile_bin_boundaries": boundaries.tolist(),
            "zero_atom_separate": True,
            "positive_exact_zero_count": int(np.sum(scores[labels == 1] == 0)),
            "by_negative_case": results,
            "proposal": "Freeze training bins, generate fresh balanced case-labelled nuisance pairs; "
                        "retain the whole pair only when its scores land in the same bin. "
                        "This balances binned scores and exact nuisances, not within-bin continuous scores. "
                        "Cannot preserve all-matching negatives under exact score matching."}


def run_condition(name: str, variant: str, disjoint: bool, palette_mode: str, seed: int,
                  train_size: int, validation_size: int, test_size: int) -> dict:
    sizes = ({"train": 12, "validation": 6, "test": 6} if variant == "quadrant24" and disjoint
             else {"train": 24, "validation": 24, "test": 24} if variant == "quadrant24"
             else {"train": 256, "validation": 64, "test": 64})
    pools = make_appearance_pools(variant, sizes=sizes, seed=seed, disjoint=disjoint)
    sample_sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    datasets = {split: make_controlled_dataset(pool, sample_sizes[split], seed=seed + 1000 + i,
                                              palette_mode=palette_mode)
                for i, (split, pool) in enumerate(pools.items())}
    scores = {split: rgb_diversity(data) for split, data in datasets.items()}
    normalized_scores = {split: rgb_diversity(data, normalize_palette=True)
                         for split, data in datasets.items()}
    hamming_scores = {split: hamming_diversity(data) for split, data in datasets.items()}
    pair_indices = {split: matched_pair_indices(data) for split, data in datasets.items()}
    train, train_scores = datasets["train"], scores["train"]
    rules = {"full_threshold": fit_threshold(train_scores, train["y"]),
             "full_interval": fit_interval(train_scores, train["y"]),
             "pair_threshold": fit_threshold(train_scores[pair_indices["train"]],
                                              train["y"][pair_indices["train"]]),
             "full_normalized_interval": fit_interval(normalized_scores["train"], train["y"]),
             "pair_normalized_threshold": fit_threshold(
                 normalized_scores["train"][pair_indices["train"]], train["y"][pair_indices["train"]]),
             "full_hamming_interval": fit_interval(hamming_scores["train"], train["y"]),
             "pair_hamming_threshold": fit_threshold(
                 hamming_scores["train"][pair_indices["train"]], train["y"][pair_indices["train"]])}
    evaluations = {}
    for rule_name, rule in rules.items():
        evaluations[rule_name] = {}
        for split, data in datasets.items():
            selection = (pair_indices[split] if rule_name.startswith("pair_")
                         else np.arange(len(data["y"])))
            statistic = (normalized_scores if "normalized" in rule_name else
                         hamming_scores if "hamming" in rule_name else scores)
            predictions = predict(rule, statistic[split][selection])
            evaluations[rule_name][split] = metrics(
                data["y"][selection], predictions, np.asarray(data["cases"])[selection])
    return {"condition": name, "variant": variant, "disjoint_appearances": disjoint,
            "palette_mode": palette_mode, "seed": seed,
            "statistic": "mean pairwise squared RGB distance per corresponding object pixel",
            "fitting": "rules fitted on training scores and labels only; validation/test untouched",
            "rules": rules, "evaluations": evaluations,
            "checks": {split: dataset_checks(data) for split, data in datasets.items()},
            "overlap": audit_splits(datasets, require_disjoint_appearances=disjoint),
            "pool_distances": {split: distance_diagnostics(pool.patterns, seed=seed)
                               for split, pool in pools.items()},
            "training_score_matching": {
                "raw": distribution_matching_diagnostic(train, train_scores),
                "normalized": distribution_matching_diagnostic(train, normalized_scores["train"]),
                "hamming": distribution_matching_diagnostic(train, hamming_scores["train"])},
            "score_by_test_case": {case: {"mean": float(scores["test"][np.array(
                datasets["test"]["cases"]) == case].mean()),
                "std": float(scores["test"][np.array(datasets["test"]["cases"]) == case].std())}
                                   for case in Counter(datasets["test"]["cases"])}}


def write_figure(results: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ["one_pair", "all_distinct", "two_pairs", "triple", "all_matching"]
    conditions = ["quadrant24_known", "cells8_heldout", "pixels16_heldout"]
    colours = {"fixed": "#3563a5", "shared": "#c2662a"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, baseline in enumerate(["full_interval", "full_normalized_interval", "full_hamming_interval"]):
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            for palette in ("fixed", "shared"):
                matching = [r for r in results if r["condition"] == condition and r["palette_mode"] == palette]
                values = np.array([[100 * r["evaluations"][baseline]["test"]["by_case"][case]["accuracy"]
                                    for case in cases] for r in matching])
                ax.plot(range(5), values.mean(axis=0), "o-", color=colours[palette], label=palette)
                ax.fill_between(range(5), values.min(axis=0), values.max(axis=0),
                                color=colours[palette], alpha=0.13)
            ax.set_title(condition.replace("_", " "))
            ax.set_ylim(-3, 103)
            ax.set_xticks(range(5), ["One pair", "All distinct", "Two pairs", "Triple", "All match"],
                          rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.2)
            if column == 0:
                ax.set_ylabel(["Raw RGB diversity", "Palette-normalized diversity", "Pixel-Hamming diversity"][row]
                              + "\nTest accuracy (%)")
            if row == 0 and column == 2:
                ax.legend(title="Palette", loc="lower left")
    fig.suptitle("Scalar interval fitted on training only: richer arrangements retain shortcut signals\n"
                 "Lines: seed means; shaded ranges: seeds 42 and 7, not confidence intervals", fontsize=12)
    fig.savefig(output / "mondriaan_richer_appearances_audit_cases.png", dpi=160)
    plt.close(fig)


def write_report(results: list[dict], output: Path, train_size: int,
                 validation_size: int, test_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "mondriaan_richer_appearances_audit.json").write_text(
        json.dumps(results, indent=2) + "\n")
    rows = []
    for result in results:
        for baseline, evaluations in result["evaluations"].items():
            for split, score in evaluations.items():
                rows.append({"condition": result["condition"], "palette": result["palette_mode"],
                             "seed": result["seed"], "baseline": baseline, "split": split,
                             **{k: v for k, v in score.items() if k not in ("by_case", "confusion_matrix")},
                             **{f"accuracy_{k}": v["accuracy"] for k, v in score["by_case"].items()},
                             **{f"count_{k}": v["samples"] for k, v in score["by_case"].items()}})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "mondriaan_richer_appearances_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Richer Mondriaan appearances: generator and scalar-shortcut audit",
             "", "October 11, 2026. New bounded CPU controls; historical reports are unchanged.",
             "", "All objects have a 4×4 footprint and exactly four pixels of each colour. "
             "Quadrant, eight-cell and freely arranged pixel variants have 24, 2,520 and "
             "63,063,000 possible arrangements. The full image is preserved as 64 ordered "
             "48-value patch tokens.", "",
             f"Each run uses {train_size:,} training, {validation_size:,} validation and "
             f"{test_size:,} fresh test images. Separate generation seeds define each split. "
             "Binary labels, object counts, exact foreground layouts and palettes are balanced "
             "by pairing one positive and one negative scene. Four objects per scene; negatives "
             "cycle equally through all distinct, two pairs, triple and all matching. "
             "No model is trained and no optimizer/checkpoint comparison is made here.", "",
             "The scalar statistic is mean pairwise squared RGB distance at corresponding "
             "object pixels. A single threshold and a contiguous positive interval are fitted "
             "on full-task training data; a separate threshold is fitted on a layout-matched "
             "one-pair/all-distinct training subset. Both threshold directions are considered. "
             "Every rule uses training data only; neither validation nor test selects a rule.", "",
             "| Appearance control | Palette | Full-task interval test accuracy | "
             "Full-task interval worst-case | Pair threshold test accuracy |",
             "|---|---|---:|---:|---:|"]
    keys = list(dict.fromkeys((r["condition"], r["palette_mode"]) for r in results))
    for condition, palette in keys:
        selected = [r for r in results if (r["condition"], r["palette_mode"]) == (condition, palette)]
        def span(baseline, metric):
            values = [100 * r["evaluations"][baseline]["test"][metric] for r in selected]
            return f"{min(values):.1f}–{max(values):.1f}%" if len(values) > 1 else f"{values[0]:.1f}%"
        lines.append(f"| {condition} | {palette} | {span('full_interval', 'accuracy')} | "
                     f"{span('full_interval', 'worst_case_accuracy')} | "
                     f"{span('pair_threshold', 'accuracy')} |")
    lines.extend(["", "Palette normalization divides RGB diversity by twice the RGB variance "
                  "of the shared palette, which is available from any complete object's histogram. "
                  "This is a constant multiple of mean pairwise palette-colour distance. "
                  "A separate pixel-Hamming statistic measures the average fraction of "
                  "different colour-index pixels across object pairs. It is palette invariant. "
                  "Both use complete object crops; neither uses identity labels to compute "
                  "the statistic. Their fitted rules still use training labels only.", "",
                  "| Appearance control | Palette | Normalized RGB full-task accuracy | "
                  "Hamming full-task accuracy | Hamming worst-case | Hamming pair accuracy |",
                  "|---|---|---:|---:|---:|---:|"])
    for condition, palette in keys:
        selected = [r for r in results if (r["condition"], r["palette_mode"]) == (condition, palette)]
        def controlled_span(baseline, metric):
            values = [100 * r["evaluations"][baseline]["test"][metric] for r in selected]
            return f"{min(values):.1f}–{max(values):.1f}%" if len(values) > 1 else f"{values[0]:.1f}%"
        lines.append(f"| {condition} | {palette} | "
                     f"{controlled_span('full_normalized_interval', 'accuracy')} | "
                     f"{controlled_span('full_hamming_interval', 'accuracy')} | "
                     f"{controlled_span('full_hamming_interval', 'worst_case_accuracy')} | "
                     f"{controlled_span('pair_hamming_threshold', 'accuracy')} |")
    lines.extend(["", "Ranges are minimum–maximum results across the listed seeds, "
                  "not confidence intervals. Per-case counts, binary F1, confusion matrices, "
                  "training/validation performance and threshold values are in the linked artifacts.",
                  "", "![Per-case scalar performance](mondriaan_richer_appearances_audit_cases.png)",
                  "", "The known-quadrant control uses the same 24 appearances in all splits. "
                  "The heldout-quadrant control splits them 12/6/6. Eight-cell and pixel controls "
                  "use disjoint 256/64/64 appearance pools. Pixel-based checks verify exact "
                  "rendered strict-pair labels, equal within-image means/histograms, exact patch "
                  "reconstruction, nuisance balance, no identical images between splits, and "
                  "no shared canonical arrangements in heldout conditions. Shared-palette "
                  "runs draw four distinct nonblack RGB colours once per paired image layout "
                  "and reuse them for every object; there is no independent per-type recolouring.",
                  "", "Equal histograms remove mean/histogram identity shortcuts, but pairwise "
                  "arrangement diversity still changes with repetition. Richer vocabularies "
                  "therefore require a measured statistical baseline; they do not automatically "
                  "eliminate every shortcut or prove that a learned pooling solution is impossible. "
                  "The scalar audit is a diagnostic with complete object crops; its performance "
                  "cannot be compared as an equal-input model against off-grid full scenes.",
                  "", "The generator also supports 3/4/6 or other object counts, shared palettes, "
                  "independent integer-pixel placements, complete-object crops, and minimum "
                  "pixel-Hamming-distance codebooks. These implementations have tests, but this "
                  "table measures only four-object aligned scenes and unrestricted distance. "
                  "With more than four objects, the sampled canonical negative families are "
                  "representative, not exhaustive integer partitions. Distance diagnostics "
                  "state whether pair enumeration is exhaustive or sampled.",
                  "", "**Can scalar score distributions be matched?** Exact matching over "
                  "the complete five-case task is impossible while retaining all-matching "
                  "negatives: their diversity is exactly zero, whereas every strict-one-pair "
                  "positive has positive diversity. The negative distribution has a zero atom "
                  "absent from positives. Bins must keep that atom separate.",
                  "", "For ambiguous subcases, a training-only feasibility diagnostic fixes "
                  "eight positive-score quantile bins. A prospective sampling control would "
                  "generate each opposite-label pair with the same count, layout and palette, "
                  "then retain the whole pair only if its scores lie in the same frozen bin. "
                  "That preserves exact nuisance balance and equal binned score counts. "
                  "It does not equalize continuous scores within bins. Independent per-label "
                  "rejection would break layout/palette balance. The predeclared generator "
                  "may use known case labels for conditional sampling; bins and thresholds "
                  "must not change in response to test outcomes.", "",
                  "| Pixel-pattern control | Palette | All-distinct paired-bin retention | "
                  "Two-pair retention | Triple retention | All-matching retention |",
                  "|---|---|---:|---:|---:|---:|"])
    for palette in ("fixed", "shared"):
        selected = [r for r in results if r["condition"] == "pixels16_heldout" and r["palette_mode"] == palette]
        retained = []
        for case in ("all_distinct", "two_pairs", "triple", "all_matching"):
            values = [100 * r["training_score_matching"]["hamming"]["by_negative_case"][case]["paired_bin_retention"]
                      for r in selected]
            retained.append(f"{min(values):.1f}–{max(values):.1f}%")
        lines.append(f"| pixels16_heldout | {palette} | " + " | ".join(retained) + " |")
    lines.extend(["", "Retention rates use Hamming scores and training data only. "
                  "They are feasibility diagnostics, not a score-matched test or new training "
                  "experiment. Low overlap can require substantial rejection and changes "
                  "case-conditioned difficulty. Preserve the full five-case benchmark and "
                  "add a separately named matched-subcase challenge. Repeatedly used audit "
                  "tests become development diagnostics; a new final test is required after "
                  "choosing any revised sampling protocol.",
                  "", "Run: `PYTHONPATH=src .venv/bin/python scripts/audit_mondriaan_appearances.py`.",
                  "", "Artifacts: [CSV](mondriaan_richer_appearances_audit.csv), "
                  "[JSON with all diagnostics](mondriaan_richer_appearances_audit.json)."])
    write_figure(results, output)
    (output / "mondriaan_richer_appearances_audit.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if any(n <= 0 or n % 8 for n in (args.train_size, args.validation_size, args.test_size)):
        parser.error("Sample sizes must be positive multiples of eight")
    conditions = [("quadrant24_known", "quadrant24", False),
                  ("quadrant24_heldout", "quadrant24", True),
                  ("cells8_heldout", "cells8", True),
                  ("pixels16_heldout", "pixels16", True)]
    results = []
    for name, variant, disjoint in conditions:
        for palette in ("fixed", "shared"):
            for seed in args.seeds:
                result = run_condition(name, variant, disjoint, palette, seed,
                                       args.train_size, args.validation_size, args.test_size)
                results.append(result)
                print(name, palette, seed,
                      {k: round(v["test"]["accuracy"], 4)
                       for k, v in result["evaluations"].items()}, flush=True)
    write_report(results, args.output_dir, args.train_size, args.validation_size, args.test_size)


if __name__ == "__main__":
    main()
