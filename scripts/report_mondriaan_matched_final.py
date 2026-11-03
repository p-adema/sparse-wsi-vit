"""Add fixed-final endpoint diagnostics without changing frozen challenge code.

This report-only evaluation reuses exact saved challenge NPZ inputs. It tests
every fixed 4000-update endpoint, never chooses between selected/final models,
and verifies selected predictions against the original saved evaluation.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.probe_mondriaan_benchmark import (
    build_model, evaluate, flatten_metrics, grouped_test_uncertainty, tensor_data,
)
from scripts.probe_mondriaan_matched_challenge import (
    INTENDED_CASES, challenge_source_hashes, selection_gate, sha256_file,
)


def verify_receipt(root):
    receipt = json.loads((root / "challenge_source_receipt.json").read_text())
    for filename, key in (("protocol.json", "protocol_sha256"),
                          ("training_policies.json", "training_policies_sha256")):
        if sha256_file(root / filename) != receipt[key]:
            raise ValueError(f"Frozen challenge receipt mismatch: {filename}")
    if challenge_source_hashes() != receipt["challenge_source_sha256"]:
        raise ValueError("Frozen challenge/scalar source receipt mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-root", type=Path,
                        default=Path("runs/mondriaan_oct11_matched_challenge"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    root, report_dir = args.challenge_root, args.report_dir
    if (root / "final_checkpoint_results.json").exists():
        raise ValueError("Supplemental endpoint evaluation already exists")
    receipt = verify_receipt(root)
    protocol = json.loads((root / "protocol.json").read_text())
    sources = json.loads((root / "training_policies.json").read_text())
    audits = json.loads((root / "challenge_audits.json").read_text())
    selected_results = json.loads((root / "results.json").read_text())
    selections = selection_gate(protocol)
    torch.set_num_threads(protocol["threads"])
    results, score_audits, loaded_arrays = [], {}, {}
    for seed, run_dir in zip(protocol["seeds"], protocol["run_dirs"], strict=True):
        npz_path = root / f"challenge_seed{seed}.npz"
        if sha256_file(npz_path) != audits[str(seed)]["npz_sha256"]:
            raise ValueError("Saved challenge arrays changed")
        with np.load(npz_path) as npz:
            arrays = {key: npz[key] for key in npz.files}
        loaded_arrays[seed] = arrays
        exact_hashes = np.asarray(audits[str(seed)]["challenge_image_hashes"])
        np.testing.assert_array_equal(arrays["image_sha256"], exact_hashes)
        import hashlib
        actual_hashes = np.asarray([hashlib.sha256(image.tobytes()).hexdigest()
                                    for image in arrays["images"]])
        np.testing.assert_array_equal(actual_hashes, exact_hashes)
        data = tensor_data({"x": arrays["x"], "coords": arrays["coords"],
                            "y": arrays["labels"], "cases": arrays["cases"].tolist(),
                            "pair_ids": arrays["pair_ids"], "intended_cases": INTENDED_CASES})
        saved = {"labels": arrays["labels"], "cases": arrays["cases"],
                 "pair_ids": arrays["pair_ids"], "image_sha256": exact_hashes}
        for name in protocol["models"]:
            item = selections[f"{name}/seed{seed}"]
            selected_path = Path(item["checkpoint"])
            final_path = Path(run_dir) / f"{name}_seed{seed}" / "final.pt"
            if sha256_file(final_path) != item["final_checkpoint_sha256"]:
                raise ValueError("Fixed-final checkpoint changed after selection gate")
            selected_state = torch.load(selected_path, map_location="cpu", weights_only=True)
            settings = selected_state["settings"]
            model = build_model(name, settings["coordinate_scale"], settings.get("width"),
                                settings["transmil_landmarks"])
            model.load_state_dict(selected_state["model_state"])
            selected_metrics, predictions = evaluate(model, data, protocol["batch_size"],
                                                      INTENDED_CASES, return_predictions=True)
            np.testing.assert_array_equal(predictions, arrays[f"predictions_{name}"])
            original = next(run for run in selected_results
                            if run["model"] == name and run["seed"] == seed)
            if selected_metrics != original["test"]:
                raise ValueError("Reevaluated selected metrics differ from frozen challenge results")
            saved[f"selected_predictions_{name}"] = predictions
            final_state = torch.load(final_path, map_location="cpu", weights_only=True)
            if final_state["step"] != protocol["steps"]:
                raise ValueError("Final checkpoint is not the fixed budget endpoint")
            model.load_state_dict(final_state["model_state"])
            metrics, predictions = evaluate(model, data, protocol["batch_size"],
                                            INTENDED_CASES, return_predictions=True)
            saved[f"final_predictions_{name}"] = predictions
            results.append({"model": name, "seed": seed, "checkpoint_phase": "fixed_final",
                            "step": final_state["step"], "test": metrics,
                            "uncertainty": grouped_test_uncertainty(data, predictions, seed),
                            "checkpoint": str(final_path), "checkpoint_sha256": sha256_file(final_path),
                            "challenge_npz_sha256": sha256_file(npz_path),
                            "selected_prediction_reproduction": "exact",
                            "selected_checkpoint_sha256": item["checkpoint_sha256"]})
            del model, selected_state, final_state
        np.savez_compressed(root / f"final_checkpoint_predictions_seed{seed}.npz", **saved)
        bins = arrays["matching_bins"]
        requested = protocol["bins"]
        policy_edges = sources[str(seed)]["policy"]["boundaries"]["4"]
        labels = arrays["labels"]
        score_audits[str(seed)] = {
            "top_bin_id": requested, "top_bin_lower_boundary": policy_edges[-1],
            "top_bin_upper_boundary": None, "top_bin_is_unbounded": True,
            "top_bin_pair_fraction": float(np.mean(bins == requested)),
            "hamming_mean_by_label": {
                str(label): float(np.mean(arrays["score_hamming"][labels == label])) for label in (0, 1)},
            "hamming_median_by_label": {
                str(label): float(np.median(arrays["score_hamming"][labels == label])) for label in (0, 1)},
        }
    metadata = {"scope": "Requested supplemental fixed-final endpoint diagnostic; no selection or tuning",
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "report_script_sha256": sha256_file(__file__),
                "frozen_source_receipt": receipt, "score_audits": score_audits,
                "results": results}
    verify_receipt(root)
    (root / "final_checkpoint_results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (report_dir / "mondriaan_matched_challenge_checkpoints.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for run in selected_results:
        if run["kind"] == "learned":
            row = {"model": run["model"], "seed": run["seed"], "checkpoint_phase": "validation_selected",
                   "step": run["selected_step"]}
            row.update(flatten_metrics(run["test"]))
            row["checkpoint_sha256"] = run["checkpoint_sha256"]
            low, high = run["uncertainty"]["accuracy_95_percentile_interval"]
            row.update(test_accuracy_ci_low=low, test_accuracy_ci_high=high)
            rows.append(row)
    for run in results:
        row = {key: run[key] for key in ("model", "seed", "checkpoint_phase", "step", "checkpoint_sha256")}
        row.update(flatten_metrics(run["test"]))
        low, high = run["uncertainty"]["accuracy_95_percentile_interval"]
        row.update(test_accuracy_ci_low=low, test_accuracy_ci_high=high)
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (report_dir / "mondriaan_matched_challenge_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    append_interpretation(report_dir, selected_results, results, score_audits, audits)
    supplemental_figure(report_dir, selected_results, results, loaded_arrays, score_audits)
    print(json.dumps({"fixed_final": [{"model": run["model"], "seed": run["seed"],
                       "accuracy": run["test"]["accuracy"], "worst": run["test"]["worst_case_accuracy"]}
                      for run in results], "score_audits": score_audits}, indent=2))


def append_interpretation(report_dir, selected, final, score_audits, audits):
    lines = ["", "## Measured verdict and fixed-final endpoints", "",
             "**Matching frozen bins removed the bin-only label signal, but did not remove "
             "the continuous diversity shortcut.** The original full-training Hamming interval "
             "still achieves 75.4–80.9% accuracy (68.4–75.4% worst-case), and normalized RGB "
             "intervals achieve 66.4–70.9%. In contrast, every frozen-bin lookup is exactly "
             "50% by the paired design. The matched challenge is a valid audit of coarse-bin "
             "matching, but it is not a credible shortcut-removing benchmark.", "",
             "Most accepted pairs lie in the unbounded highest bin. Each pair is matched "
             "at this coarse level while positives and negatives can occupy different "
             "continuous-score ranges inside that bin. This is an observed limitation, "
             "not a sampling-policy adjustment; bins and quotas remain frozen.", "",
             "| Seed | Pairs in highest bin | Highest-bin lower edge | Bin mismatch rejections | Duplicate rejections | Quota skips |",
             "|---|---:|---:|---:|---:|---:|"]
    for seed, item in score_audits.items():
        rejection = audits[seed]["matching_audit"]
        lines.append(f"| {seed} | {100 * item['top_bin_pair_fraction']:.1f}% | "
                     f"{item['top_bin_lower_boundary']:.5f} | "
                     f"{sum(rejection['rejected_bin_mismatch_by_stratum'].values())} | "
                     f"{sum(rejection['rejected_duplicate_images_by_stratum'].values())} | "
                     f"{sum(rejection['skipped_completed_quota_by_stratum'].values())} |")
    lines.extend(["", "The additional fixed-final evaluation below reports every 4000-update "
                  "checkpoint on the **same saved challenge inputs**. It was requested as an "
                  "additive endpoint diagnostic after the primary fits and validation selections "
                  "were frozen; it does not select a better checkpoint or tune any model. "
                  "Reevaluating all nine selected checkpoints exactly reproduced the original "
                  "predictions and metrics. Pixel and checkpoint hashes were verified against "
                  "the saved challenge and primary selection manifests.", "",
                  "| Model | Seed | Selected accuracy | Final accuracy | Selected worst-case | Final worst-case | Final paired 95% interval |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for run in final:
        original = next(item for item in selected if item["model"] == run["model"] and item["seed"] == run["seed"])
        low, high = run["uncertainty"]["accuracy_95_percentile_interval"]
        lines.append(f"| {run['model']} | {run['seed']} | {100 * original['test']['accuracy']:.1f}% | "
                     f"{100 * run['test']['accuracy']:.1f}% | {100 * original['test']['worst_case_accuracy']:.1f}% | "
                     f"{100 * run['test']['worst_case_accuracy']:.1f}% | {100 * low:.1f}–{100 * high:.1f}% |")
    lines.extend(["", "Training-seed ranges summarize variation across these three fitted "
                  "solutions; the intervals above describe conditional paired test uncertainty. "
                  "Neither learned endpoint provides evidence of solving the diagnostic. "
                  "The scalar counterexample makes architectural-necessity claims especially "
                  "premature, despite exact labels and nuisance matching.", "",
                  "Additive artifacts: [selected/final CSV](mondriaan_matched_challenge_checkpoints.csv), "
                  "[endpoint provenance JSON](mondriaan_matched_challenge_checkpoints.json), "
                  "[continuous-score figure](mondriaan_matched_challenge_scores.png). "
                  "Endpoint prediction NPZs accompany the original challenge NPZs in the run directory."])
    report = report_dir / "mondriaan_matched_challenge.md"
    if "## Measured verdict and fixed-final endpoints" in report.read_text():
        raise ValueError("Endpoint discussion already exists")
    with report.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def supplemental_figure(report_dir, selected, final, arrays, score_audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    labels = ["ViT", "TransMIL", "Learned mean", "Continuous Hamming\nfull-train interval", "Frozen-bin\nlookup"]
    model_names = ["vit5", "transmil", "mean", "full_training/hamming_interval",
                   "full_training/frozen_hamming_bin"]
    seeds = list(arrays)
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for i, model in enumerate(model_names):
        for j, seed in enumerate(seeds):
            original = next(item for item in selected if item["model"] == model and item["seed"] == seed)
            x = i + (j - 1) * .12
            axes[0].scatter(x, 100 * original["test"]["accuracy"], color=colors[j], marker="o", s=36)
            if i < 3:
                endpoint = next(item for item in final if item["model"] == model and item["seed"] == seed)
                axes[0].scatter(x, 100 * endpoint["test"]["accuracy"], color=colors[j], marker="x", s=48)
    axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(labels)), labels, fontsize=9)
    axes[0].set_ylim(25, 100)
    axes[0].set_ylabel("Matched-challenge accuracy (%)")
    axes[0].set_title("Continuous shortcut survives bin matching")
    for j, seed in enumerate(seeds):
        axes[0].scatter([], [], color=colors[j], label=f"Seed {seed}")
    axes[0].scatter([], [], color="black", marker="o", label="Selected / scalar")
    axes[0].scatter([], [], color="black", marker="x", label="Fixed final")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    scores = np.concatenate([item["score_hamming"] for item in arrays.values()])
    targets = np.concatenate([item["labels"] for item in arrays.values()])
    edges = np.arange(0, 1 + 1 / 96, 1 / 96)
    for label, name, color in ((1, "One pair", "#377eb8"), (0, "All distinct", "#ff7f00")):
        values = scores[targets == label]
        axes[1].hist(values, bins=edges, weights=np.full(len(values), 1 / len(values)),
                     histtype="step", linewidth=2, color=color, label=name)
    boundaries = [item["top_bin_lower_boundary"] for item in score_audits.values()]
    axes[1].axvspan(min(boundaries), max(boundaries), alpha=.18, color="gray", label="Top-bin lower-edge range")
    axes[1].set_xlim(min(scores) - .02, max(scores) + .02)
    axes[1].set_xlabel("Continuous complete-object Hamming diversity")
    axes[1].set_ylabel("Fraction of label examples per score bin")
    axes[1].set_title("90–97% of pairs fall in the unbounded top bin")
    axes[1].legend(fontsize=8)
    fig.savefig(report_dir / "mondriaan_matched_challenge_scores.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
