"""Reproduce off-grid diagnostics and evaluate the frozen variable-count final test."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_benchmark import (
    build_model,
    evaluate,
    flatten_metrics,
    grouped_test_uncertainty,
    make_local_splits,
    planned_cases,
    tensor_data,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    audit_splits,
    complete_object_view,
    dataset_checks,
)


def image_hashes(data):
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]]


def collect(root, *, fresh=False):
    protocol = json.loads((root / "protocol.json").read_text())
    if fresh:
        frozen = json.loads((root / "final_test_protocol.json").read_text())
        protocol.update(test_seed_offset=frozen["test_seed_offset"])
    args = argparse.Namespace(**protocol)
    args.output_dir = root
    saved = json.loads((root / "results.json").read_text())
    rows, diagnostics, predictions = [], {}, {}
    for seed in args.seeds:
        _, training, validation, create_test = make_local_splits(args, seed)
        raw_test = create_test()
        hashes = image_hashes(raw_test)
        original = json.loads((root / f"test_manifest_seed{seed}.json").read_text())
        if fresh:
            if set(hashes) & set(original["image_sha256"]):
                raise ValueError("New final test overlaps the previously consumed development test")
        elif hashes != original["image_sha256"]:
            raise ValueError("Regenerated test differs from the frozen run")
        overlap = audit_splits({"train": training, "validation": validation, "test": raw_test})
        test = tensor_data(complete_object_view(raw_test)
                           if args.representation == "objects" else raw_test)
        test["intended_cases"] = planned_cases(args)
        seed_checks = {"dataset": dataset_checks(raw_test), "overlap": overlap,
                       "test_image_sha256": hashes,
                       "previous_test_image_overlap": len(set(hashes) & set(original["image_sha256"])),
                       "checkpoints": {}}
        for name in args.models:
            directory = root / f"{name}_seed{seed}"
            training_report = json.loads((directory / "training.json").read_text())
            prior = next(row for row in saved if row["model"] == name and row["seed"] == seed)
            model = build_model(name, args.coordinate_scale or 32, args.width)
            for mode, filename in (("selected", "best.pt"), ("final", "final.pt")):
                checkpoint = torch.load(directory / filename, weights_only=True)
                model.load_state_dict(checkpoint["model_state"])
                metrics, predicted = evaluate(model, test, args.batch_size, return_predictions=True)
                if not fresh:
                    reference = prior["test" if mode == "selected" else "test_final_update"]
                    for metric in ("accuracy", "worst_case_accuracy", "binary_macro_f1"):
                        if not np.isclose(metrics[metric], reference[metric], atol=1e-12, rtol=0):
                            raise ValueError(f"Saved test score not reproduced: {seed}/{name}/{mode}/{metric}")
                rows.append({"model": name, "seed": seed, "checkpoint": mode,
                             "step": checkpoint["step"], "parameters": prior["parameters"],
                             "tokens": test["x"].shape[1], "seconds": training_report["seconds"],
                             **flatten_metrics(metrics)})
                key = f"{name}_seed{seed}_{mode}"
                predictions[key] = predicted
                seed_checks["checkpoints"][key] = {
                    "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                    "metrics": metrics,
                    "conditional_test_uncertainty": grouped_test_uncertainty(test, predicted, seed),
                }
            predictions[f"seed{seed}_targets"] = test["y"].numpy()
            predictions[f"seed{seed}_cases"] = np.array(test["cases"])
        diagnostics[str(seed)] = seed_checks
    if fresh:
        (root / "final_test_results.json").write_text(json.dumps(
            {"protocol": protocol, "rows": rows, "audit": diagnostics}, indent=2) + "\n")
    return rows, diagnostics, predictions


def write_artifacts(prefix, rows, checks, predictions):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
    np.savez_compressed(prefix.with_suffix(".npz"), **predictions)


def score_range(rows, key):
    values = [100 * row[key] for row in rows]
    return f"{min(values):.1f}–{max(values):.1f}%"


def plot(prefix, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("accuracy", "worst_case_accuracy"),
                                 ("Overall", "Worst planned subgroup"), strict=True):
        for index, (label, rows) in enumerate(groups):
            values = [100 * row[metric] for row in rows]
            ax.scatter(index + np.linspace(-.06, .06, len(values)), values, s=40)
            ax.plot([index - .15, index + .15], [np.mean(values)] * 2, color="black")
        ax.set_xticks(range(len(groups)), [label for label, _ in groups], rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    figure.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(figure)


def write_report(prefix, title, description, groups, caveats):
    lines = [f"# {title}", "", description, "",
             "Each range spans three independent seeds (42, 7, 2026), using validation-selected "
             "checkpoints. These ranges describe training/data variation; they are not confidence intervals.", "",
             "| Input/model | Overall | Worst planned subgroup | Binary macro-F1 |",
             "|---|---:|---:|---:|"]
    for label, rows in groups:
        lines.append(f"| {label.replace(chr(10), ' / ')} | {score_range(rows, 'accuracy')} | "
                     f"{score_range(rows, 'worst_case_accuracy')} | {score_range(rows, 'binary_macro_f1')} |")
    lines.extend(["", f"![Three-seed selected scores]({prefix.name}.png)", "", caveats, "",
                  "Checkpoint selection uses validation worst-case accuracy, then binary macro-F1, then "
                  "overall accuracy. Each fit uses 2,000 AdamW updates, batch size 32, learning rate 2e-4, "
                  "and CPU float32. Test metrics never select a checkpoint. Learned pooling has about "
                  "2.37 million parameters; ABMIL has 177,539. Equal updates and similar pooling capacity "
                  "do not equalize compute or establish that each architecture has been optimally tuned.", "",
                  f"[Full selected and final-update metrics]({prefix.name}.csv), "
                  f"[split/checkpoint audit and conditional test uncertainty]({prefix.name}.json), "
                  f"[saved predictions]({prefix.name}.npz). "
                  "The audit retains every case count and accuracy, confusion matrix, checkpoint hash, "
                  "and paired-layout bootstrap interval. Those intervals condition on fixed checkpoints "
                  "and exclude training and validation-selection uncertainty.", "",
                  "Reproduce both reports without refitting:", "", "```sh",
                  "MPLCONFIGDIR=/private/tmp/mondriaan-report-cache .venv/bin/python scripts/report_mondriaan_local_controls.py",
                  "```"])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    torch.set_num_threads(1)
    scene_root = Path("runs/mondriaan_benchmark_offgrid_scene")
    objects_root = Path("runs/mondriaan_benchmark_offgrid_objects")
    variable_root = Path("runs/mondriaan_benchmark_variable_counts_balanced")
    offgrid_rows, offgrid_checks, offgrid_predictions = [], {}, {}
    groups = []
    for label, root in (("Full scene", scene_root), ("Complete objects", objects_root)):
        rows, checks, predictions = collect(root)
        offgrid_rows.extend({"representation": label, **row} for row in rows)
        offgrid_checks[label] = checks
        offgrid_predictions.update({f"{label.replace(' ', '_')}_{key}": value for key, value in predictions.items()})
        for name in ("mean", "coord_mean", "abmil"):
            groups.append((f"{label}\n{name}", [r for r in rows if r["model"] == name and r["checkpoint"] == "selected"]))
    for seed in (42, 7, 2026):
        if offgrid_checks["Full scene"][str(seed)]["test_image_sha256"] != offgrid_checks["Complete objects"][str(seed)]["test_image_sha256"]:
            raise ValueError("Scene/object comparison must use identical source test images")
    prefix = Path("reports/mondriaan_offgrid_diagnostics")
    write_artifacts(prefix, offgrid_rows, offgrid_checks, offgrid_predictions)
    plot(prefix, groups)
    write_report(prefix, "Off-grid full scenes versus complete-object diagnostics",
                 "Shared-palette, fixed-histogram 4×4 pixel patterns with disjoint 256-appearance pools. "
                 "Both representations use identical off-grid training, validation and test source scenes: "
                 "1,024/512/1,024 images per seed. Source test hashes and saved scores are reproduced exactly. "
                 "The full scene has 64 lossless 4×4 patch tokens; the diagnostic has four complete-object tokens.",
                 groups,
                 "Full-scene models remain near 50% overall. Supplying complete objects helps learned mean "
                 "and coordinate-aware pooling modestly, while worst-case scores remain low. The crop "
                 "diagnostic supplies true segmentation, removes 60 background/fragment tokens, and changes "
                 "optimization difficulty; it cannot isolate reconstruction as the sole cause. Coordinates "
                 "are supplied to coordinate-aware pooling, but these runs include no off-grid ViT comparison.")
    rows, checks, predictions = collect(variable_root, fresh=True)
    groups = [(name, [r for r in rows if r["model"] == name and r["checkpoint"] == "selected"])
              for name in ("mean", "coord_mean")]
    prefix = Path("reports/mondriaan_variable_counts")
    write_artifacts(prefix, rows, checks, predictions)
    plot(prefix, groups)
    write_report(prefix, "Balanced 3/4/6-object local task",
                 "Shared-palette, aligned pixel patterns with disjoint 256-appearance pools; "
                 "1,536 training, 384 validation and 768 final-test images per seed. The strict positive "
                 "rule is exactly one repeated pair and all other identities occur once. Both the datasets "
                 "and every minibatch match object-count allocations across labels.", groups,
                 "The initial variable-count run is excluded: uniform negative-family sampling overrepresented "
                 "four/six objects relative to three among negative minibatches. Its consumed test images are "
                 "development data. The corrected run's final test uses separately frozen seed offset 166000 "
                 "and the already validation-selected checkpoints, with zero overlap against consumed tests. "
                 "No model, hyperparameter or checkpoint was selected on this replacement test. There are "
                 "13 planned count/case subgroups; higher-count negatives cover representative families "
                 "(all-distinct, two pairs, triple, all-matching), rather than every integer partition. "
                 "This is mixed-count training and evaluation, not transfer to unseen object counts.")
    print(json.dumps({"offgrid_rows": len(offgrid_rows), "variable_rows": len(rows),
                      "variable_selected": {name: score_range(selected, "accuracy") for name, selected in groups}}))


if __name__ == "__main__":
    main()
