"""Backfill case-sensitive metrics without modifying historical October runs.

Binary ground truth is constant within each repetition case, so the saved case
counts/accuracies determine exact confusion counts. All recovered counts and
overall scores are checked before companion CSVs, diagnostics, plots, and a
Markdown report are written. This performs no training, checkpoint selection,
or checkpoint inference. Smoke runs are excluded by default.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mondriaan_metrics import CASE_NAMES, metrics_from_case_accuracies


FAMILIES = ("mondriaan_patch_training", "mondriaan_matching_controls", "mondriaan_curriculum")
SCORES = ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "positive_precision",
          "positive_recall", "positive_f1", "binary_macro_f1",
          "observed_worst_case_accuracy", "observed_macro_case_accuracy")


def audit_reports(run_root, *, include_smoke=False):
    """Return recovered history/final rows, main full-task rows, and provenance."""
    history, final, primary, sources = [], [], [], []
    for family in FAMILIES:
        for path in sorted((run_root / family).rglob("*.json")):
            relative = path.relative_to(run_root)
            if not include_smoke and "smoke" in relative.parts:
                continue
            raw = path.read_bytes()
            report = json.loads(raw)
            if "settings" not in report or not report.get("history"):
                continue
            settings = report["settings"]
            if report["history"][-1]["step"] != settings["steps"]:
                raise ValueError(f"Incomplete historical run: {path}")
            experiment = "/".join(relative.parts[1:-1])
            provenance = {
                "source": str(relative), "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint": report.get("checkpoint"),
                "checkpoint_policy": "final update; historical evaluation not used for selection",
            }
            base = {"family": family, "experiment": experiment, "model": settings["model"],
                    "seed": settings["seed"], "train_size": settings["train_size"],
                    "parameters": report["parameters"], "source": str(relative),
                    "checkpoint_policy": provenance["checkpoint_policy"]}
            last_rows = {}
            recovered_records = []
            for record in report["history"]:
                offset = settings.get("parent_steps", 0)
                views = {"train": record["train"], "evaluation": record["evaluation"],
                         **{f"robustness/{view}": metrics
                            for view, metrics in record.get("robustness", {}).items()}}
                recovered_views = {}
                for view, original in views.items():
                    metrics = metrics_from_case_accuracies(original["by_case"])
                    if not math.isclose(metrics["accuracy"], original["accuracy"], abs_tol=1e-7):
                        raise ValueError(f"Saved overall accuracy disagrees with case counts: {path}/{view}")
                    row = {**base, "view": view, "step": record["step"],
                           "total_steps": record["step"] + offset,
                           "seconds": record["seconds"], "count": metrics["count"],
                           "case_coverage": metrics["case_coverage"],
                           "full_task_coverage": metrics["selection_eligible"],
                           **{score: metrics[score] for score in SCORES},
                           **metrics["confusion_counts"],
                           **{f"{case}_{key}": metrics["by_case"][case][key]
                              for case in CASE_NAMES for key in ("count", "correct", "accuracy")}}
                    history.append(row)
                    recovered_views[view] = metrics
                    if record is report["history"][-1]:
                        final.append(row)
                        last_rows[view] = row
                recovered_records.append({"step": record["step"], "total_steps": record["step"] + offset,
                                          "metrics": recovered_views})
            # The balanced pair evaluation is not a full-task result. Its
            # full-task robustness view is the appropriate five-case summary.
            main = last_rows.get("robustness/full_task", last_rows["evaluation"])
            if main["case_coverage"] != len(CASE_NAMES):
                raise ValueError(f"No complete five-case final evaluation for {path}")
            primary.append(main)
            sources.append({**provenance, "settings": settings, "recovered_history": recovered_records})
    if not sources:
        raise ValueError(f"No completed historical reports under {run_root}")
    return history, final, primary, sources


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _label(row):
    return f"{row['experiment']} · {row['model']} · s{row['seed']} · n={row['train_size']}"


def plot_results(path, primary, *, case_scores=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [[row for row in primary if row["family"] == family] for family in FAMILIES]
    heights = [max(len(group), 1) for group in groups]
    fig, axes = plt.subplots(3, 1, figsize=(14, sum(heights) * 0.29 + 3),
                             gridspec_kw={"height_ratios": heights}, layout="constrained")
    if case_scores:
        series = [(f"{case}_accuracy", case.replace("_", " ")) for case in CASE_NAMES]
    else:
        series = [("accuracy", "overall"), ("macro_case_accuracy", "macro case"),
                  ("worst_case_accuracy", "worst case"), ("positive_f1", "positive F1"),
                  ("binary_macro_f1", "binary macro F1")]
    markers = ("o", "s", "^", "D", "x")
    for ax, family, group in zip(axes, FAMILIES, groups, strict=True):
        y = np.arange(len(group))
        for (key, label), marker in zip(series, markers, strict=True):
            ax.scatter([row[key] * 100 for row in group], y, label=label, marker=marker, s=38)
        ax.set_yticks(y, [_label(row) for row in group], fontsize=8)
        # Reserve one empty row above the measurements for the legend.
        ax.set_ylim(len(group) - 0.5, -1.5)
        ax.set_xlim(-2, 102)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(family.replace("mondriaan_", "").replace("_", " "), loc="left", fontsize=11)
        ax.set_xlabel("Accuracy / score (%)" if not case_scores else "Case accuracy (%)")
        ax.legend(loc="upper right", fontsize=8, ncols=5)
    fig.suptitle("October checkpoints: complete five-case evaluation; final-update reporting", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path, primary, sources, prefix):
    lines = [
        "# October Mondriaan metrics audit",
        "",
        "This companion audit adds case-sensitive and binary metrics to the three October reports. "
        "Historical run files, checkpoints, summaries, and conclusions are retained. "
        "All values are recovered from saved case counts and accuracies; no new training, "
        "inference, or checkpoint selection was performed.",
        "",
        "Each repetition case has one binary label, so case counts and correct counts determine "
        "TP, FP, FN, and TN exactly. The audit rejects non-integer reconstructions beyond float32 "
        "rounding tolerance and checks reconstructed overall accuracy against every saved evaluation. "
        f"It covers **{len(sources)} completed runs**; smoke runs are excluded.",
        "",
        "Worst-case accuracy is the minimum of the five case scores; macro-case accuracy is their "
        "unweighted mean. Binary macro-F1 averages F1 over the two binary labels. "
        "Undefined precision, recall, and F1 use zero. Confusion matrices have true labels as rows, "
        "predictions as columns, both ordered `[0, 1]`.",
        "",
        "Incomplete objectives, such as the balanced pair test, show unobserved cases with count zero "
        "and accuracy null. Full-task worst-case and macro-case scores are null there, with explicit "
        "observed-case summaries and `full_task_coverage=False`. The table and figures below use "
        "the **complete five-case evaluation** for every checkpoint, including pair-trained models.",
        "",
        "All complete five-case evaluations contain 1,024 examples: 512 one-pair positives and "
        "128 of each negative case. Historical checkpoints are reported at their final update. "
        "Coverage eligibility does not authorize choosing checkpoints on these development images. "
        "Each table row is a separate seed; two-seed ranges are not confidence intervals.",
        "",
        f"[Final-view CSV]({prefix}_summary.csv), [all logged updates CSV]({prefix}_history.csv), "
        f"[counts, confusion matrices, and source hashes]({prefix}_diagnostics.json).",
        "",
        f"![Overall, case, and F1 scores]({prefix}_scores.png)",
        "",
        f"![All five repetition-case scores]({prefix}_cases.png)",
        "",
        "| Experiment | Model | Seed | Training n | Overall | Worst case | Macro case | Positive P / R / F1 | Binary macro-F1 | TN / FP / FN / TP |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in primary:
        experiment = row["family"].replace("mondriaan_", "") + "/" + row["experiment"]
        def pct(key):
            return f"{row[key] * 100:.1f}%"
        prf = " / ".join(pct(key) for key in ("positive_precision", "positive_recall", "positive_f1"))
        confusion = " / ".join(str(row[key]) for key in ("tn", "fp", "fn", "tp"))
        lines.append(f"| {experiment} | {row['model']} | {row['seed']} | {row['train_size']} | "
                     f"{pct('accuracy')} | {pct('worst_case_accuracy')} | {pct('macro_case_accuracy')} | "
                     f"{prf} | {pct('binary_macro_f1')} | {confusion} |")
    lines.extend([
        "",
        "The full → full ViT checkpoints reach 93.7–95.4% overall, but their worst case is "
        "71.1–73.4%, always all-distinct. Positive F1 is 93.8–95.5% and binary macro-F1 is "
        "93.6–95.4%; these high binary scores still coexist with substantially weaker all-distinct rejection.",
        "",
        "Pair → full ViT has 91.8–93.0% overall and 60.2–87.5% worst-case accuracy. "
        "In seed 42, two-pair rejection (87.5%) is slightly weaker than all-distinct rejection (88.3%); "
        "in seed 7, all-distinct is the worst case (60.2%). Thus the curriculum's worst case varies by seed.",
        "",
        "Full → full has higher overall accuracy in both seeds, but the worst-case ordering reverses "
        "between seeds: pair → full is stronger on the worst case in seed 42, and weaker in seed 7. "
        "The present two seeds therefore establish no consistent curriculum advantage on worst-case accuracy.",
        "",
        "The pair-trained ViT checkpoints have **0% full-task worst-case accuracy** because "
        "they fail every two-pair, triple, and four-matching example. Their balanced pair-test scores "
        "must be interpreted as a different objective. The larger-data 2,000-update full-task ViTs "
        "also have only 1.6–3.9% worst-case accuracy despite 73.6–78.4% overall accuracy.",
        "",
        "An always-negative predictor scores 50% overall and 80% macro-case accuracy on this "
        "balanced full task, while its positive F1 and worst-case accuracy are zero. "
        "An always-positive predictor has positive F1 66.7% and worst-case accuracy zero. "
        "A predictor solving everything except all-distinct scores 87.5% overall, 88.9% positive F1, "
        "and 87.3% binary macro-F1, yet 0% worst-case accuracy. These are checked in automated tests.",
        "",
        "The audit sharpens the existing conclusions: ViT learns much of the aligned task after "
        "continuation, but the case distribution can hide important failures. A fresh held-out "
        "test and validation-based protocol remain necessary for final model claims.",
        "",
        "Reproduce with:",
        "",
        "```sh",
        f".venv/bin/python scripts/audit_mondriaan_metrics.py --run-root runs --output-prefix reports/{prefix}",
        "```",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/mondriaan_metrics_audit"))
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    history, final, primary, sources = audit_reports(args.run_root, include_smoke=args.include_smoke)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), final)
    write_csv(prefix.with_name(prefix.name + "_history.csv"), history)
    diagnostics = {"method": "exact confusion counts reconstructed from constant-label case summaries",
                   "historical_files_modified": False, "checkpoint_policy": "final-update reporting",
                   "run_count": len(sources), "metric_rows": len(history), "sources": sources}
    prefix.with_name(prefix.name + "_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    plot_results(prefix.with_name(prefix.name + "_scores.png"), primary)
    plot_results(prefix.with_name(prefix.name + "_cases.png"), primary, case_scores=True)
    write_report(prefix.with_suffix(".md"), primary, sources, prefix.name)
    # A final source hash check also guards accidental writes during the audit.
    for source in sources:
        raw = (args.run_root / source["source"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise RuntimeError(f"Historical source changed during audit: {source['source']}")
    print(json.dumps({"runs": len(sources), "history_rows": len(history), "final_rows": len(final),
                      "report": str(prefix.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
