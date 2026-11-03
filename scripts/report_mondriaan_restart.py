"""Report the optimizer restart control after its frozen experiment completes.

Rebuilds only the already-used test images, verifies their saved SHA-256 manifest,
and obtains selected/final-checkpoint predictions for paired layout bootstrap.
No fitting, tuning, or checkpoint selection occurs here. The paired intervals
describe uncertainty in these test samples, not variation across training seeds.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mondriaan_metrics import CASE_NAMES
from scripts.probe_mondriaan_benchmark import (
    build_model, evaluate, flatten_metrics, make_local_splits, tensor_data,
)


def layout_groups(positions):
    groups = defaultdict(list)
    for i, corners in enumerate(positions):
        groups[tuple(sorted(tuple(corner) for corner in corners))].append(i)
    return list(groups.values())


def paired_layout_bootstrap(labels, cases, predictions, positions, *, seed, repeats=2000, pair_ids=None):
    """Resample shared layout clusters once for all arm/checkpoint comparisons."""
    labels, cases = np.asarray(labels), np.asarray(cases)
    if pair_ids is None:
        groups = layout_groups(positions)
    else:
        pairs = defaultdict(list)
        for i, pair in enumerate(pair_ids):
            pairs[int(pair)].append(i)
        groups = list(pairs.values())
    group_ids = np.empty(len(labels), dtype=np.int64)
    counts = np.zeros((len(groups), len(CASE_NAMES)), dtype=np.int64)
    for index, members in enumerate(groups):
        group_ids[members] = index
        for case_index, case in enumerate(CASE_NAMES):
            counts[index, case_index] = np.sum(cases[members] == case)
    if not counts.sum(axis=0).all():
        raise ValueError("Bootstrap needs all five observed test cases")
    rng = np.random.default_rng(seed + 12000)
    draws = rng.integers(len(groups), size=(repeats, len(groups)))
    sampled_counts = counts[draws].sum(axis=1)
    if (sampled_counts == 0).any():
        raise ValueError("A bootstrap draw omitted a case; use a larger complete evaluation")
    samples, intervals = {}, {}
    for key, predicted in predictions.items():
        predicted = np.asarray(predicted)
        if predicted.shape != labels.shape:
            raise ValueError("Prediction arrays must match the test labels")
        correct = predicted == labels
        group_correct = np.zeros_like(counts)
        confusion = np.zeros((len(groups), 4), dtype=np.int64)
        for index, members in enumerate(groups):
            for case_index, case in enumerate(CASE_NAMES):
                group_correct[index, case_index] = np.sum(correct[members] & (cases[members] == case))
            y, p = labels[members], predicted[members]
            confusion[index] = [np.sum((y == 0) & (p == 0)), np.sum((y == 0) & (p == 1)),
                                np.sum((y == 1) & (p == 0)), np.sum((y == 1) & (p == 1))]
        sampled_correct = group_correct[draws].sum(axis=1)
        case_accuracy = sampled_correct / sampled_counts
        tn, fp, fn, tp = confusion[draws].sum(axis=1).T
        positive_denominator, negative_denominator = 2 * tp + fp + fn, 2 * tn + fp + fn
        positive_f1 = np.divide(2 * tp, positive_denominator, out=np.zeros(repeats),
                                where=positive_denominator != 0)
        negative_f1 = np.divide(2 * tn, negative_denominator, out=np.zeros(repeats),
                                where=negative_denominator != 0)
        values = {"accuracy": sampled_correct.sum(axis=1) / sampled_counts.sum(axis=1),
                  "worst_case_accuracy": case_accuracy.min(axis=1),
                  "macro_case_accuracy": case_accuracy.mean(axis=1),
                  "positive_f1": positive_f1, "binary_macro_f1": (positive_f1 + negative_f1) / 2}
        samples[key] = values
        intervals[key] = {metric: np.quantile(values_, [.025, .975]).tolist()
                          for metric, values_ in values.items()}
    deltas = {}
    for mode in ("selected", "final"):
        restart, uninterrupted = f"restart/{mode}", f"uninterrupted/{mode}"
        if restart not in samples or uninterrupted not in samples:
            continue
        deltas[mode] = {metric: np.quantile(samples[restart][metric] - samples[uninterrupted][metric],
                                          [.025, .975]).tolist()
                        for metric in samples[restart]}
    return {"groups": len(groups), "repeats": repeats, "intervals": intervals,
            "paired_restart_minus_uninterrupted_intervals": deltas,
            "interpretation": "95% percentile layout-cluster bootstrap conditional on fixed checkpoints; not seed uncertainty"}, group_ids


def reconstruct_test(protocol, seed, root):
    args = argparse.Namespace(**protocol)
    _, _, _, create_test = make_local_splits(args, seed)
    data = create_test()
    manifest = json.loads((root / f"test_manifest_seed{seed}.json").read_text())
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]]
    if hashes != manifest["image_sha256"]:
        raise RuntimeError("Reconstructed test differs from the frozen test manifest")
    data = tensor_data(data)
    data["intended_cases"] = CASE_NAMES
    return data


def collect(root, *, repeats=2000, seed_run_dirs=None):
    protocol = json.loads((root / "protocol.json").read_text())
    if (protocol["task"] != "restart" or protocol["variant"] != "quadrant24"
            or protocol["models"] != ["vit5"]):
        raise ValueError("This report is for the quadrant24 local ViT restart control")
    run_roots = {seed: root for seed in protocol["seeds"]}
    if seed_run_dirs:
        run_roots = {}
        comparable = ("task", "variant", "models", "palette", "placement", "representation",
                      "object_counts", "train_size", "validation_size", "test_size",
                      "appearance_pool_size", "appearance_holdout", "min_distance", "steps",
                      "restart_at", "batch_size", "lr", "threads", "eval_every", "width")
        for directory in seed_run_dirs:
            child_protocol = json.loads((directory / "protocol.json").read_text())
            if len(child_protocol["seeds"]) != 1:
                raise ValueError("Independent seed directories must each contain one experiment seed")
            for key in comparable:
                if child_protocol[key] != protocol[key]:
                    raise ValueError(f"Independent seed protocol changed {key}: {directory}")
            seed = child_protocol["seeds"][0]
            if seed in run_roots:
                raise ValueError("Duplicate independent seed directory")
            run_roots[seed] = directory
        if set(run_roots) != set(protocol["seeds"]):
            raise ValueError("Independent directories do not cover the frozen experiment seeds")
    original_results = [row for directory in dict.fromkeys(run_roots.values())
                        for row in json.loads((directory / "results.json").read_text())]
    if len(original_results) != 2 * len(protocol["models"]) * len(protocol["seeds"]):
        raise ValueError("The frozen experiment is incomplete")
    torch.set_num_threads(protocol["threads"])
    rows, diagnostics, arrays = [], {}, {}
    for seed in protocol["seeds"]:
        run_root = run_roots[seed]
        data = reconstruct_test(protocol, seed, run_root)
        arrays[f"labels_seed{seed}"] = data["y"].numpy()
        arrays[f"cases_seed{seed}"] = np.asarray(data["cases"])
        for name in protocol["models"]:
            predictions = {}
            branch_final_states = {}
            for arm in ("uninterrupted", "restart"):
                directory = run_root / f"{name}_seed{seed}" / arm
                result = next(row for row in original_results
                              if row["seed"] == seed and row["model"] == name and row["arm"] == arm)
                report = json.loads((directory / "training.json").read_text())
                for mode, checkpoint_name, result_key in (("selected", "best.pt", "test"),
                                                         ("final", "final.pt", "test_final_update")):
                    model = build_model(name, canvas_size=32, width=protocol["width"])
                    state = torch.load(directory / checkpoint_name, weights_only=True)
                    if mode == "final":
                        branch_final_states[arm] = state
                    model.load_state_dict(state["model_state"])
                    metrics, predicted = evaluate(model, data, protocol["batch_size"],
                                                  return_predictions=True)
                    for key in ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "positive_f1",
                                "binary_macro_f1", "confusion_matrix"):
                        if metrics[key] != result[result_key][key]:
                            raise RuntimeError(f"Checkpoint re-evaluation changed {key}: {directory}/{mode}")
                    if mode == "selected":
                        # The selected checkpoint may belong to the common prefix.
                        records = report["history"]
                        if state["step"] < protocol["restart_at"]:
                            records = json.loads((directory.parent / "prefix/training.json").read_text())["history"]
                        validation = next(entry["validation"] for entry in records if entry["step"] == state["step"])
                    else:
                        validation = report["history"][-1]["validation"]
                    row = {"model": name, "seed": seed, "arm": arm, "checkpoint": mode,
                           "step": state["step"], "parameters": result["parameters"],
                           "recorded_training_seconds": report["seconds"],
                           "timing_scope": report.get("seconds_scope", "branch suffix only; common prefix excluded"),
                           **{f"test_{key}": value for key, value in flatten_metrics(metrics).items()},
                           **{f"validation_{key}": validation[key] for key in (
                               "accuracy", "worst_case_accuracy", "macro_case_accuracy", "binary_macro_f1")}}
                    rows.append(row)
                    predictions[f"{arm}/{mode}"] = predicted
                    arrays[f"predictions_{name}_seed{seed}_{arm}_{mode}"] = predicted
            bootstrap, ids = paired_layout_bootstrap(data["y"].numpy(), data["cases"], predictions,
                                                    data["positions"], seed=seed, repeats=repeats,
                                                    pair_ids=data.get("pair_ids"))
            for key in ("sampler_state", "torch_rng_state"):
                if not torch.equal(branch_final_states["uninterrupted"][key], branch_final_states["restart"][key]):
                    raise RuntimeError(f"Restart branches diverged in their {key}: {name}/seed{seed}")
            optimizer_steps = {
                arm: sorted({int(item["step"].item()) for item in state["optimizer_state"]["state"].values()})
                for arm, state in branch_final_states.items()
            }
            if (optimizer_steps["uninterrupted"] != [protocol["steps"]]
                    or optimizer_steps["restart"] != [protocol["steps"] - protocol["restart_at"]]):
                raise RuntimeError("Saved optimizer counters do not match the restart protocol")
            bootstrap["trajectory_controls"] = {
                "matching_final_sampler_state": True, "matching_final_torch_rng_state": True,
                "optimizer_step_counters": optimizer_steps,
                "common_prefix_sha256": hashlib.sha256(
                    (run_root / f"{name}_seed{seed}" / "prefix/final.pt").read_bytes()).hexdigest(),
            }
            bootstrap["source_run_directory"] = str(run_root)
            diagnostics[f"{name}/seed{seed}"] = bootstrap
            arrays[f"layout_group_seed{seed}"] = ids
    return protocol, rows, diagnostics, arrays


def write_outputs(prefix, protocol, rows, diagnostics, arrays):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_name(prefix.name + "_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_name(prefix.name + "_uncertainty.json").write_text(json.dumps({
        "protocol": protocol, "conditional_test_sample_uncertainty": diagnostics,
        "seed_uncertainty": "three independent experiment seeds; ranges and paired differences are descriptive, not confidence intervals",
    }, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(prefix.with_name(prefix.name + "_predictions.npz"), **arrays)
    plot(prefix.with_name(prefix.name + "_scores.png"), rows, diagnostics, protocol)
    write_report(prefix.with_suffix(".md"), protocol, rows, diagnostics, prefix.name)


def plot(path, rows, diagnostics, protocol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    seeds = list(dict.fromkeys(row["seed"] for row in rows))
    for column, mode in enumerate(("selected", "final")):
        for row_index, metric in enumerate(("accuracy", "worst_case_accuracy")):
            ax = axes[row_index, column]
            for offset, arm, marker in ((-.08, "uninterrupted", "o"), (.08, "restart", "s")):
                values = [row for row in rows if row["checkpoint"] == mode and row["arm"] == arm]
                x = np.array([seeds.index(row["seed"]) for row in values]) + offset
                y = np.array([row[f"test_{metric}"] for row in values]) * 100
                bounds = np.array([diagnostics[f"{row['model']}/seed{row['seed']}"]["intervals"][f"{arm}/{mode}"][metric]
                                   for row in values]) * 100
                colour = "C0" if arm == "uninterrupted" else "C1"
                # Percentile intervals need not contain the point estimate,
                # particularly for the minimum of tied case accuracies.
                ax.vlines(x, bounds[:, 0], bounds[:, 1], color=colour)
                ax.scatter(x, y, marker=marker, label=arm, color=colour, zorder=3)
                ax.scatter(np.repeat(x, 2), bounds.reshape(-1), marker="_", color=colour)
            ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
            ax.set_xlabel("Experiment seed")
            ax.set_ylabel("Fresh-test accuracy (%)")
            ax.set_ylim(0, 102)
            ax.set_title(f"{mode.capitalize()} checkpoint · {metric.replace('_', ' ')}")
            ax.grid(axis="y", alpha=.25)
            ax.legend()
    fig.suptitle(f"Optimizer restart at {protocol['restart_at']:,} updates; {protocol['steps']:,}-update matched budget\n"
                 "Error bars: conditional test-sample layout bootstrap, not seed uncertainty")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(path, protocol, rows, diagnostics, prefix):
    def pct(value):
        return f"{100 * value:.1f}%"

    differences = {}
    for mode in ("selected", "final"):
        differences[mode] = []
        for seed in protocol["seeds"]:
            paired = {row["arm"]: row for row in rows if row["seed"] == seed and row["checkpoint"] == mode}
            differences[mode].append(100 * (paired["restart"]["test_accuracy"] - paired["uninterrupted"]["test_accuracy"]))

    lines = ["# Mondriaan optimizer restart control", "",
             "Restart effects depend on seed and checkpoint policy. At validation-selected checkpoints, "
             f"restart changes overall accuracy by {min(differences['selected']):+.1f} to "
             f"{max(differences['selected']):+.1f} percentage points. At the fixed final checkpoint, "
             f"the changes are {min(differences['final']):+.1f} to {max(differences['final']):+.1f} points. "
             "The final seed-42 checkpoint improves substantially in worst-case accuracy, while the "
             "other seeds have smaller, uncertain changes. Restarting can affect the trajectory, but "
             "these three seeds do not establish a reliable overall improvement under the chosen "
             "validation-selection policy.", "",
             f"This experiment isolates restarting AdamW after {protocol['restart_at']:,} updates from continuing with its "
             "saved state. Both arms start from one identical learned prefix and restore the same "
             "minibatch-sampler and dropout random states. The uninterrupted arm preserves optimizer "
             f"moments and step counters; the restarted arm starts fresh AdamW state. Both reach {protocol['steps']:,} updates.", "",
             f"The fixed protocol uses {protocol['train_size']:,} training images, "
             f"{protocol['validation_size']:,} validation images, and {protocol['test_size']:,} fresh test images "
             f"per seed, with seeds {', '.join(map(str, protocol['seeds']))}. Images use all 24 quadrant "
             "arrangements, aligned lossless 4×4 patches, four objects, a fixed palette, and the strict "
             "one-pair label. ViT has 2,381,954 parameters, 64 tokens per image, three blocks, and width 256. "
             "AdamW uses constant learning rate 2e-4, weight decay 1e-4, batches of 32, clipping at 1, CPU float32.", "",
             "Checkpoints are selected using validation worst-case accuracy, then binary macro-F1, then "
             "overall accuracy, over all scheduled checkpoints including the common prefix. Test images "
             "were first generated after fitting and selection finished. Both the validation-selected "
             f"checkpoint and the matched {protocol['steps']:,}-update final checkpoint are reported; test performance "
             "does not select either. The reporter verifies regenerated test images against saved image "
             "hashes and reproduces the saved test metrics before computing uncertainty.", "",
             f"[Summary CSV]({prefix}_summary.csv), [conditional uncertainty]({prefix}_uncertainty.json), "
             f"[saved test predictions]({prefix}_predictions.npz).", "",
             f"![Selected and final checkpoint scores]({prefix}_scores.png)", "",
             "| Seed | Arm | Checkpoint | Update | Validation worst | Test overall | Test worst | Test macro case | Positive F1 | Binary macro-F1 |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        keys = ("validation_worst_case_accuracy", "test_accuracy", "test_worst_case_accuracy",
                "test_macro_case_accuracy", "test_positive_f1", "test_binary_macro_f1")
        lines.append(f"| {row['seed']} | {row['arm']} | {row['checkpoint']} | {row['step']} | "
                     + " | ".join(pct(row[key]) for key in keys) + " |")
    lines.extend(["", "| Seed | Checkpoint | Restart − uninterrupted overall | Conditional 95% interval | Restart − uninterrupted worst case | Conditional 95% interval |",
                  "|---:|---|---:|---|---:|---|"])
    for seed in protocol["seeds"]:
        for mode in ("selected", "final"):
            selected = [row for row in rows if row["seed"] == seed and row["checkpoint"] == mode]
            uninterrupted = next(row for row in selected if row["arm"] == "uninterrupted")
            restarted = next(row for row in selected if row["arm"] == "restart")
            intervals = diagnostics[f"{uninterrupted['model']}/seed{seed}"]["paired_restart_minus_uninterrupted_intervals"][mode]
            fields = []
            for metric in ("accuracy", "worst_case_accuracy"):
                delta = 100 * (restarted[f"test_{metric}"] - uninterrupted[f"test_{metric}"])
                low, high = (100 * value for value in intervals[metric])
                fields.extend([f"{delta:+.1f} pp", f"[{low:+.1f}, {high:+.1f}] pp"])
            lines.append(f"| {seed} | {mode} | " + " | ".join(fields) + " |")
    lines.extend(["", "Uncertainty has two different sources. The conditional intervals resample matching "
                  "layout clusters, keeping the arms paired on the same images. They describe finite-test-sample "
                  "uncertainty for fixed fitted checkpoints. They do not include training variation or uncertainty "
                  "from selecting checkpoints on a finite validation set. Each seed changes initialization, "
                  "training/validation/test sampling, and stochastic training; between-seed differences and "
                  "three-seed ranges are descriptive evidence, not confidence intervals.", "",
                  "These runs share the finite 24-pattern vocabulary across splits and differ from the October "
                  "development pools. They test restart effects on new image combinations and layouts, without "
                  "establishing transfer to unseen appearances. Comparisons with the historical curriculum "
                  "should account for the new data generator and validation selection rule.", "",
                  "The independent seed jobs reuse saved common prefixes. Interrupted restart branches for "
                  "seeds 7 and 2026 were recovered from their serialized best checkpoints, retaining the "
                  "prior validation selection and restoring AdamW, sampler and dropout states. Updates after "
                  "those checkpoints were replayed to the frozen final budget. Final sampler/dropout states "
                  "and optimizer counters are verified across both arms. Recorded wall times exclude the "
                  "common prefix; recovered branches record only the recovered segment. These timings "
                  "are not full-budget runtime comparisons.", "",
                  "Reproduce the completed report without refitting:", "", "```sh",
                  f"MPLCONFIGDIR=/private/tmp/mondriaan-report-cache .venv/bin/python scripts/report_mondriaan_restart.py --run-dir {protocol['output_dir']} --output-prefix reports/{prefix}"
                  + (" --seed-run-dirs " + " ".join(dict.fromkeys(
                      value["source_run_directory"] for value in diagnostics.values()))
                     if any(value["source_run_directory"] != protocol["output_dir"] for value in diagnostics.values()) else ""),
                  "```"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mondriaan_benchmark_restart"))
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/mondriaan_restart_control"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed-run-dirs", type=Path, nargs="+", default=None,
                        help="Assemble compatible independent seed jobs under the original frozen protocol")
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("bootstrap-repeats must be at least 100")
    protocol, rows, diagnostics, arrays = collect(args.run_dir, repeats=args.bootstrap_repeats,
                                                  seed_run_dirs=args.seed_run_dirs)
    write_outputs(args.output_prefix, protocol, rows, diagnostics, arrays)
    print(json.dumps({"rows": len(rows), "report": str(args.output_prefix.with_suffix('.md'))}))


if __name__ == "__main__":
    main()
