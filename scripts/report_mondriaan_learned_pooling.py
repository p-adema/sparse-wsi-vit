"""Summarize frozen learned-pooling runs and fit scalar diagnostics on exact splits.

This command performs no training or checkpoint selection. It verifies the fresh
test manifests before computing the prespecified scalar diagnostic families.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_mondriaan_appearances import fit_interval, hamming_diversity, predict, rgb_diversity
from scripts.mondriaan_metrics import CASE_NAMES, classification_metrics
from scripts.probe_mondriaan_benchmark import make_local_splits


def collect(directory: Path) -> tuple[list[dict], dict]:
    protocol = json.loads((directory / "protocol.json").read_text())
    learned = json.loads((directory / "results.json").read_text())
    records = [{"method": r["model"], "kind": "learned", "palette": protocol["palette"],
                "seed": r["seed"], "parameters": r["parameters"],
                "selected_step": r["selected_step"], "seconds": r["seconds"], "test": r["test"]}
               for r in learned]
    args = argparse.Namespace(**protocol)
    alias = {"triple": "three_matching", "all_matching": "four_matching"}
    for seed in protocol["seeds"]:
        _, train, _, create_test = make_local_splits(args, seed)
        test = create_test()
        manifest = json.loads((directory / f"test_manifest_seed{seed}.json").read_text())
        hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in test["images"]]
        if hashes != manifest["image_sha256"]:
            raise ValueError("Regenerated scalar diagnostic test does not match learned-model test")
        cases = [alias.get(case, case) for case in test["cases"]]
        statistics = {"rgb_diversity": lambda data: rgb_diversity(data),
                      "normalized_rgb_diversity": lambda data: rgb_diversity(data, normalize_palette=True),
                      "hamming_diversity": hamming_diversity}
        for name, statistic in statistics.items():
            rule = fit_interval(statistic(train), train["y"])
            metrics = classification_metrics(test["y"], predict(rule, statistic(test)), cases,
                                             intended_cases=CASE_NAMES)
            records.append({"method": name, "kind": "complete_object_scalar", "palette": protocol["palette"],
                            "seed": seed, "parameters": 0, "selected_step": None,
                            "seconds": None, "training_rule": rule, "test": metrics,
                            "test_manifest_exactly_matched": True})
    return records, protocol


def write_figure(records: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["mean", "sum", "abmil", "rgb_diversity", "normalized_rgb_diversity", "hamming_diversity"]
    labels = ["Mean", "Sum", "ABMIL", "RGB scalar", "Norm. RGB", "Hamming"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for row, palette in enumerate(("fixed", "shared")):
        for col, metric in enumerate(("accuracy", "worst_case_accuracy")):
            ax = axes[row, col]
            values = [[100 * r["test"][metric] for r in records
                       if r["palette"] == palette and r["method"] == name] for name in names]
            means = np.array([np.mean(v) for v in values])
            minimum = np.array([min(v) for v in values])
            maximum = np.array([max(v) for v in values])
            bars = ax.bar(range(len(names)), means, color=["#3563a5"] * 3 + ["#c2662a"] * 3,
                          yerr=np.stack([means - minimum, maximum - means]), capsize=3)
            for bar in bars[3:]:
                bar.set_hatch("//")
            ax.set_xticks(range(len(names)), labels, rotation=25, ha="right")
            ax.set_ylim(0, 103)
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"{palette.capitalize()} palette · " + ("overall" if col == 0 else "worst case"))
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Richer appearances: learned pooling and complete-object scalar diagnostics\n"
                 "Bars: three-seed means; whiskers: observed ranges. Hatched bars use object crops.", fontsize=12)
    fig.savefig(output / "mondriaan_learned_pooling.png", dpi=160)
    plt.close(fig)


def write_report(records: list[dict], protocols: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "mondriaan_learned_pooling.json").write_text(json.dumps(
        {"protocols": protocols, "results": records}, indent=2) + "\n")
    rows = []
    for record in records:
        metrics = record["test"]
        row = {k: record[k] for k in ("method", "kind", "palette", "seed", "parameters", "selected_step", "seconds")}
        row.update({k: metrics[k] for k in ("accuracy", "worst_case_accuracy", "macro_case_accuracy",
                                          "positive_precision", "positive_recall", "positive_f1", "binary_macro_f1")})
        row.update({f"accuracy_{case}": metrics["by_case"][case]["accuracy"] for case in CASE_NAMES})
        row.update({f"count_{case}": metrics["by_case"][case]["count"] for case in CASE_NAMES})
        row.update({"tn": metrics["confusion_matrix"][0][0], "fp": metrics["confusion_matrix"][0][1],
                    "fn": metrics["confusion_matrix"][1][0], "tp": metrics["confusion_matrix"][1][1]})
        rows.append(row)
    with (output / "mondriaan_learned_pooling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = protocols[0]
    lines = ["# Richer appearances: learned pooling and scalar diagnostics", "", "October 11, 2026.", "",
             f"Three independent seeds ({', '.join(map(str, protocol['seeds']))}) compare deeper "
             "arrangement-sensitive mean and sum pooling with the original ABMIL reference. "
             f"All runs use {protocol['steps']:,} CPU float32 AdamW updates, batch size {protocol['batch_size']}, "
             f"learning rate {protocol['lr']:g}, {protocol['train_size']:,} training images, "
             f"{protocol['validation_size']:,} validation images and {protocol['test_size']:,} fresh test images. "
             "Every scene has four freely arranged 4×4 pixel patterns, each with four pixels per colour, "
             "placed on aligned patches in a 32×32 canvas. Models receive all 64 lossless 48-value patches. "
             "There are separate, disjoint 256-appearance train/validation/test pools. Counts, exact "
             "foreground layouts and palettes are paired across labels.", "",
             "Validation selects from scheduled checkpoints by worst-case accuracy, then binary macro-F1, "
             "then overall accuracy. Model training finishes before the fresh test is generated. "
             "A separate report command regenerates the exact frozen datasets, verifies every test pixel "
             "hash against the original manifest, and fits three scalar interval families on training only. "
             "No test result changes the protocol or selected checkpoint.", "",
             "The learned pooling widths are chosen near ViT's 2.38 million parameters; ABMIL retains "
             "its original 0.178 million parameters. The deeper pooling encoders have no attention between "
             "tokens and use no coordinates in this aligned experiment. Constant background patches are "
             "encoded once with their exact multiplicity, an equivalent computation retaining the full "
             "input. Mean and sum have the same representational information at fixed token count, "
             "but their scaling changes optimization. They are separate fitted models.", "",
             "| Palette | Method | Parameters | Test accuracy | Worst case | Macro-case | Positive F1 | Binary macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    names = ["mean", "sum", "abmil", "rgb_diversity", "normalized_rgb_diversity", "hamming_diversity"]
    for palette in ("fixed", "shared"):
        for name in names:
            selected = [r for r in records if r["palette"] == palette and r["method"] == name]
            def span(metric):
                values = [100 * r["test"][metric] for r in selected]
                return f"{min(values):.1f}–{max(values):.1f}%"
            lines.append(f"| {palette} | {name} | {selected[0]['parameters']:,} | {span('accuracy')} | "
                         f"{span('worst_case_accuracy')} | {span('macro_case_accuracy')} | "
                         f"{span('positive_f1')} | {span('binary_macro_f1')} |")
    lines.extend(["", "Ranges are minima–maxima across three training seeds, not confidence intervals. "
                  "Per-run positive precision/recall, binary confusion matrices, selected update, "
                  "runtime and all case denominators are in CSV/JSON. Each test has 512 one-pair "
                  "positives and 128 examples of each negative case.", "",
                  "![Learned and scalar scores](mondriaan_learned_pooling.png)", "",
                  "| Palette | Learned model | One pair | All distinct | Two pairs | Triple | All matching |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for palette in ("fixed", "shared"):
        for name in names[:3]:
            selected = [r for r in records if r["palette"] == palette and r["method"] == name]
            spans = []
            for case in ("one_pair", "all_distinct", "two_pairs", "three_matching", "four_matching"):
                values = [100 * r["test"]["by_case"][case]["accuracy"] for r in selected]
                spans.append(f"{min(values):.1f}–{max(values):.1f}%")
            lines.append(f"| {palette} | {name} | " + " | ".join(spans) + " |")
    lines.extend(["", "These bounded runs assess this architecture and optimization budget. "
                  "Low learned-pooling scores do not establish that pooling cannot express the rule, "
                  "nor do they isolate reconstruction difficulties: the present objects are aligned. "
                  "The current comparison has no simultaneously trained ViT, so it cannot support "
                  "a richer-appearance attention-versus-pooling performance claim.", "",
                  "The scalar controls use complete-object crops and their shared colour histogram. "
                  "They are diagnostics with supplied object grouping, not equal-input learned models. "
                  "Their training objective maximizes overall accuracy of a single score interval, "
                  "whereas model checkpoint selection prioritizes worst-case accuracy. Strong scalar "
                  "performance nevertheless demonstrates an aggregate similarity signal that must be "
                  "accounted for before interpreting an overall score as general exact matching. "
                  "Palette normalization removes overall colour-scale variation; pixel-Hamming "
                  "diversity is palette invariant. Shared palettes alone do not remove this signal.", "",
                  "Related: [generator and statistical audit](mondriaan_richer_appearances_audit.md). "
                  "Artifacts: [CSV](mondriaan_learned_pooling.csv), [JSON](mondriaan_learned_pooling.json). "
                  "Frozen model protocols and all checkpoints remain in "
                  "`runs/mondriaan_benchmark_pixels_fixed` and `runs/mondriaan_benchmark_pixels_shared`."])
    write_figure(records, output)
    (output / "mondriaan_learned_pooling.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=[Path("runs/mondriaan_benchmark_pixels_fixed"),
                                                                  Path("runs/mondriaan_benchmark_pixels_shared")])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    records, protocols = [], []
    for directory in args.run_dirs:
        collected, protocol = collect(directory)
        records.extend(collected)
        protocols.append(protocol)
    write_report(records, protocols, args.output_dir)


if __name__ == "__main__":
    main()
