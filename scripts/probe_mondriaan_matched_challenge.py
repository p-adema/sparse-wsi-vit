"""Freeze train-only matching, then test frozen models on a separate challenge.

The ``freeze`` phase must precede all learned fits. ``evaluate`` refuses to
generate challenge images until every requested model/seed has finished its
update budget and saved its validation selection. Matching is a diagnostic
two-case distribution; it leaves the primary five-case benchmark unchanged.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_mondriaan_appearances import (
    fit_interval, fit_threshold, matched_pair_indices, predict,
)
from scripts.mondriaan_metrics import classification_metrics
from scripts.probe_mondriaan_benchmark import (
    build_model, evaluate, flatten_metrics, grouped_test_uncertainty,
    make_local_splits, tensor_data,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import audit_splits
from sparse_wsi_vit.datasets.mondriaan_dataset.matched import (
    MatchingPolicy, build_matched_subcase_challenge, challenge_checks,
    diversity_scores, fit_matching_policy,
)


INTENDED_CASES = ("one_pair", "all_distinct")
STATISTICS = ("rgb", "normalized_rgb", "hamming")
COMPARE_SETTINGS = (
    "variant", "appearance_pool_size", "appearance_holdout", "min_distance",
    "palette", "placement", "representation", "object_counts", "train_size",
    "validation_size", "test_size", "test_seed_offset", "steps", "batch_size", "lr",
    "appearance_seed_offset", "transmil_landmarks",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_hashes(data):
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]]


def challenge_source_hashes():
    repo = Path(__file__).resolve().parents[1]
    paths = (Path(__file__), repo / "scripts/audit_mondriaan_appearances.py",
             repo / "src/sparse_wsi_vit/datasets/mondriaan_dataset/matched.py")
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def load_primary_batch(path):
    from scripts.probe_mondriaan_attention_batch import load_batch
    return load_batch(Path(path).resolve().parent)


def preflight_primary(path):
    from scripts.probe_mondriaan_attention_batch import preflight
    return preflight(Path(path).resolve().parent)


def config_from_primary(path):
    """The primary batch is the single authority for all source/model settings."""
    batch = load_primary_batch(path)
    return {**batch, "batch_protocol": str(Path(path).resolve()),
            "run_dirs": [str(Path(batch["output_dir"]) / f"seed{seed}") for seed in batch["seeds"]],
            "bins": 8, "challenge_size": 512, "challenge_seed_offset": 266000,
            "proposal_pairs": 512, "max_rounds": 100}


def restore_policy(value):
    return MatchingPolicy(value["statistic"], tuple(
        (int(count), tuple(edges)) for count, edges in value["boundaries"].items()),
        value["training_fingerprint"], value["requested_bins"])


def fit_bin_lookup(training, policy, indices):
    """Training-only majority lookup; ties/unseen bins predict negative."""
    scores = diversity_scores(training, policy.statistic)
    lookup = {}
    for index in indices:
        key = str(policy.score_bin(float(scores[index]), int(training["object_counts"][index])))
        lookup.setdefault(key, [0, 0])[int(training["y"][index])] += 1
    return {key: int(counts[1] > counts[0]) for key, counts in lookup.items()}


def fit_controls(training, policy):
    """Freeze both full-task and nuisance-paired ambiguous-training controls."""
    populations = {"full_training": np.arange(len(training["y"])),
                   "ambiguous_training": matched_pair_indices(training)}
    controls = {}
    for population, indices in populations.items():
        labels = training["y"][indices]
        for statistic in STATISTICS:
            scores = diversity_scores(training, statistic)[indices]
            for kind, fit in (("threshold", fit_threshold), ("interval", fit_interval)):
                controls[f"{population}/{statistic}_{kind}"] = {
                    "population": population, "statistic": statistic,
                    "rule": fit(scores, labels), "training_samples": len(indices),
                }
        controls[f"{population}/frozen_hamming_bin"] = {
            "population": population, "statistic": "hamming_bin",
            "lookup": fit_bin_lookup(training, policy, indices),
            "training_samples": len(indices),
        }
    return controls


def freeze(output_dir, config):
    """Write a concrete protocol and train-only rules; generate no test images."""
    output_dir = Path(output_dir)
    if (output_dir / "protocol.json").exists():
        raise ValueError("Challenge protocol already frozen; choose a new directory")
    if not Path(config["batch_protocol"]).exists():
        raise ValueError("Primary batch protocol must be frozen first")
    batch = load_primary_batch(config["batch_protocol"])
    for key in (*COMPARE_SETTINGS, "seeds", "models", "threads"):
        if config.get(key) != batch.get(key):
            raise ValueError(f"Challenge/primary setting differs before freeze: {key}")
    expected_dirs = [Path(batch["output_dir"]) / f"seed{seed}" for seed in batch["seeds"]]
    if [Path(path).resolve() for path in config["run_dirs"]] != [path.resolve() for path in expected_dirs]:
        raise ValueError("Challenge/primary seed directories differ before freeze")
    for run_dir in config["run_dirs"]:
        if Path(run_dir).exists() and any(Path(run_dir).glob("*/best.pt")):
            raise ValueError("Challenge protocol must be frozen before learned fitting")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {**config,
                "batch_protocol_sha256": sha256_file(config["batch_protocol"]),
                "challenge_source_sha256": challenge_source_hashes(),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "diagnostic_only": True, "intended_cases": list(INTENDED_CASES),
                "fitting": "original training positives only for Hamming bins; original training only for scalar rules",
                "test_gate": "all requested models/seeds finish and freeze validation selections before challenge generation",
                "rejection": "whole opposite-label nuisance pairs; fixed bins/quotas; never relaxed",
                "uncertainty": "paired test bootstrap conditional on checkpoints; training seeds reported separately"}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sources = {}
    for seed in config["seeds"]:
        pools, training, validation, _ = make_local_splits(argparse.Namespace(**config), seed)
        policy = fit_matching_policy(training, bins=config["bins"])
        sources[str(seed)] = {
            "policy": policy.as_dict(), "controls": fit_controls(training, policy),
            "training_image_hashes": image_hashes(training),
            "validation_image_hashes": image_hashes(validation),
            "training_pool_ids": pools["train"].ids.tolist(),
            "validation_pool_ids": pools["validation"].ids.tolist(),
            "test_pool_ids": pools["test"].ids.tolist(),
            "development_overlap": audit_splits({"train": training, "validation": validation}),
        }
    (output_dir / "training_policies.json").write_text(json.dumps(sources, indent=2) + "\n")
    return protocol, sources


def selection_gate(protocol):
    """Read every selection before any challenge generation/evaluation."""
    if ("challenge_source_sha256" in protocol and
            challenge_source_hashes() != protocol["challenge_source_sha256"]):
        raise ValueError("Matching/scalar/challenge source changed after freeze")
    if ("batch_protocol_sha256" in protocol and
            sha256_file(protocol["batch_protocol"]) != protocol["batch_protocol_sha256"]):
        raise ValueError("Primary batch protocol changed after challenge freeze")
    preflight_primary(protocol["batch_protocol"])
    selections = {}
    for seed, root in zip(protocol["seeds"], protocol["run_dirs"], strict=True):
        root = Path(root)
        run_protocol_path = root / "protocol.json"
        if not run_protocol_path.exists():
            raise ValueError(f"Primary protocol not ready: {root}")
        run_protocol = json.loads(run_protocol_path.read_text())
        for key in COMPARE_SETTINGS:
            if run_protocol.get(key) != protocol[key]:
                raise ValueError(f"Primary/challenge source setting differs: {root}/{key}")
        if run_protocol["seeds"] != [seed] or run_protocol["models"] != protocol["models"]:
            raise ValueError(f"Primary seed/model plan differs: {root}")
        for name in protocol["models"]:
            directory = root / f"{name}_seed{seed}"
            paths = {key: directory / filename for key, filename in (
                ("training", "training.json"), ("selected", "best.pt"), ("final", "final.pt"))}
            if not all(path.exists() for path in paths.values()):
                raise ValueError(f"All validation selections are not ready: {directory}")
            report = json.loads(paths["training"].read_text())
            if not report["history"] or report["history"][-1]["step"] != protocol["steps"]:
                raise ValueError(f"Update budget unfinished: {directory}")
            selected = torch.load(paths["selected"], weights_only=True, map_location="cpu")
            final = torch.load(paths["final"], weights_only=True, map_location="cpu")
            if final["step"] != protocol["steps"] or selected["step"] != report["selected_step"]:
                raise ValueError(f"Checkpoint selection/budget mismatch: {directory}")
            if selected["settings"]["model"] != name or selected["settings"]["seed"] != seed:
                raise ValueError(f"Wrong checkpoint identity: {directory}")
            selections[f"{name}/seed{seed}"] = {
                "model": name, "seed": seed, "selected_step": report["selected_step"],
                "selected_validation_rank": report["selected_rank"],
                "parameters": report["parameters"], "checkpoint": str(paths["selected"]),
                "checkpoint_sha256": sha256_file(paths["selected"]),
                "final_checkpoint_sha256": sha256_file(paths["final"]),
                "training_report_sha256": sha256_file(paths["training"]),
                "primary_protocol_sha256": sha256_file(run_protocol_path),
            }
    return selections


def scalar_predictions(control, challenge):
    if control["statistic"] == "hamming_bin":
        return np.array([control["lookup"].get(str(int(value)), 0)
                         for value in challenge["matching_bins"]])
    return predict(control["rule"], diversity_scores(challenge, control["statistic"]))


def primary_manifest_gate(protocol):
    manifests = {}
    for seed, root in zip(protocol["seeds"], protocol["run_dirs"], strict=True):
        path = Path(root) / f"test_manifest_seed{seed}.json"
        if not path.exists():
            raise ValueError(f"Primary final-test manifest not ready: {path}")
        manifest = json.loads(path.read_text())
        if (manifest["samples"] != protocol["test_size"] or
                len(manifest["image_sha256"]) != protocol["test_size"]):
            raise ValueError(f"Primary final-test manifest size differs: {path}")
        manifests[str(seed)] = {"path": str(path), "sha256": sha256_file(path),
                                "image_sha256": manifest["image_sha256"]}
    return manifests


def make_challenge(protocol, seed, frozen_source):
    pools, training, validation, create_primary_test = make_local_splits(argparse.Namespace(**protocol), seed)
    if (image_hashes(training) != frozen_source["training_image_hashes"]
            or image_hashes(validation) != frozen_source["validation_image_hashes"]):
        raise ValueError("Training/validation pixels changed after challenge freeze")
    policy = restore_policy(frozen_source["policy"])
    if fit_matching_policy(training, bins=protocol["bins"]).as_dict() != policy.as_dict():
        raise ValueError("Frozen matching policy changed")
    challenge = build_matched_subcase_challenge(
        pools["test"], policy, protocol["challenge_size"],
        seed=seed + protocol["challenge_seed_offset"], negative_cases=("all_distinct",),
        object_counts=(4,), aligned=True, palette_mode="shared",
        proposal_pairs=protocol["proposal_pairs"], max_rounds=protocol["max_rounds"],
    )
    checks = challenge_checks(challenge, policy)
    checks["matching_audit"] = challenge["matching_audit"]
    checks["training_validation_separation"] = audit_splits(
        {"train": training, "validation": validation, "challenge": challenge})
    primary_test = create_primary_test()
    checks["primary_test_overlap"] = audit_splits(
        {"primary_test": primary_test, "challenge": challenge},
        require_disjoint_appearances=False)
    checks["primary_test_image_hashes"] = image_hashes(primary_test)
    checks["case_counts"] = dict(Counter(challenge["cases"]))
    checks["binned_counts_by_label"] = {
        str(label): dict(Counter(map(int, challenge["matching_bins"][challenge["y"] == label])))
        for label in (0, 1)}
    if checks["binned_counts_by_label"]["0"] != checks["binned_counts_by_label"]["1"]:
        raise ValueError("Matched bins retain label imbalance")
    return challenge, checks


def evaluate_frozen(output_dir, report_dir):
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    protocol = json.loads((output_dir / "protocol.json").read_text())
    sources = json.loads((output_dir / "training_policies.json").read_text())
    if (output_dir / "results.json").exists():
        raise ValueError("Challenge already evaluated; preserve the original results")
    selections = selection_gate(protocol)
    primary_manifests = primary_manifest_gate(protocol)
    # This boundary is reached only after every model/seed selection is frozen.
    (output_dir / "selection_manifest.json").write_text(json.dumps(selections, indent=2) + "\n")
    torch.set_num_threads(protocol["threads"])
    results, audits = [], {}
    for seed in protocol["seeds"]:
        challenge, audits[str(seed)] = make_challenge(protocol, seed, sources[str(seed)])
        if audits[str(seed)]["primary_test_image_hashes"] != primary_manifests[str(seed)]["image_sha256"]:
            raise ValueError("Reconstructed primary final test differs from frozen manifest")
        audits[str(seed)]["primary_test_manifest"] = primary_manifests[str(seed)]
        tensors = tensor_data(challenge)
        tensors["intended_cases"] = INTENDED_CASES
        hashes = image_hashes(challenge)
        arrays = {"labels": challenge["y"], "cases": np.array(challenge["cases"]),
                  "pair_ids": challenge["pair_ids"], "image_sha256": np.array(hashes),
                  "matching_bins": challenge["matching_bins"],
                  "x": challenge["x"], "coords": challenge["coords"],
                  "palettes": challenge["palettes"]}
        arrays.update(images=challenge["images"], positions=np.asarray(challenge["positions"]),
                      object_types=np.asarray(challenge["object_types"]),
                      patterns=np.asarray(challenge["patterns"]))
        for statistic in STATISTICS:
            arrays[f"score_{statistic}"] = diversity_scores(challenge, statistic)
        for name, control in sources[str(seed)]["controls"].items():
            predictions = scalar_predictions(control, challenge)
            metrics = classification_metrics(challenge["y"], predictions, challenge["cases"],
                                             intended_cases=INTENDED_CASES)
            if control["statistic"] == "hamming_bin" and metrics["accuracy"] != .5:
                raise ValueError("Any deterministic bin-only lookup must score exactly 50%")
            arrays[f"predictions_{name.replace('/', '__')}"] = predictions
            results.append({"model": name, "kind": "scalar", "seed": seed,
                            "selected_step": None, "test": metrics,
                            "uncertainty": grouped_test_uncertainty(tensors, predictions, seed),
                            "training_rule": control})
        for name in protocol["models"]:
            item = selections[f"{name}/seed{seed}"]
            path = Path(item["checkpoint"])
            if sha256_file(path) != item["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after gate")
            state = torch.load(path, weights_only=True, map_location="cpu")
            settings = state["settings"]
            model = build_model(name, settings["coordinate_scale"], settings.get("width"),
                                transmil_landmarks=settings.get("transmil_landmarks", 32))
            model.load_state_dict(state["model_state"])
            metrics, predictions = evaluate(model, tensors, protocol["batch_size"],
                                            intended_cases=INTENDED_CASES, return_predictions=True)
            arrays[f"predictions_{name}"] = predictions
            results.append({**item, "kind": "learned", "test": metrics,
                            "uncertainty": grouped_test_uncertainty(tensors, predictions, seed)})
            del model, state
        np.savez_compressed(output_dir / f"challenge_seed{seed}.npz", **arrays)
        audits[str(seed)]["challenge_image_hashes"] = hashes
        audits[str(seed)]["npz_sha256"] = sha256_file(output_dir / f"challenge_seed{seed}.npz")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "challenge_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    write_report(results, audits, protocol, report_dir, output_dir)
    return results, audits


def write_report(results, audits, protocol, report_dir, output_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = "mondriaan_matched_challenge"
    rows = []
    for run in results:
        row = {key: run[key] for key in ("model", "kind", "seed", "selected_step")}
        row.update(flatten_metrics(run["test"]))
        low, high = run["uncertainty"]["accuracy_95_percentile_interval"]
        row.update(test_accuracy_ci_low=low, test_accuracy_ci_high=high)
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (report_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / f"{stem}.json").write_text(json.dumps(
        {"protocol": protocol, "results": results, "audits": audits}, indent=2) + "\n")
    lines = ["# Training-defined matched-bin challenge", "",
             "This separate diagnostic retains one-pair positives and all-distinct negatives "
             "only. It uses fresh images from the primary held-out appearance pool. Whole "
             "opposite-label pairs share layout, palette, object count and foreground mask; "
             "both images are retained only when their Hamming diversity scores fall in the "
             "same frozen training-positive quantile bin. No bins or quotas were relaxed.", "",
             f"The policy uses {protocol['bins']} requested bins, {protocol['challenge_size']} "
             f"images per seed, proposal batches of {protocol['proposal_pairs']} pairs and "
             f"at most {protocol['max_rounds']} rounds. All requested validation selections "
             "were frozen before any challenge generation. The challenge never selects "
             "checkpoints, thresholds, or hyperparameters.", "",
             "## Frozen checkpoint results", "",
             "| Model | Seed | Selected update | Accuracy | Worst of two cases | Macro F1 | Paired test 95% interval |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for run in results:
        if run["kind"] != "learned":
            continue
        low, high = run["uncertainty"]["accuracy_95_percentile_interval"]
        metrics = run["test"]
        lines.append(f"| {run['model']} | {run['seed']} | {run['selected_step']} | "
                     f"{100 * metrics['accuracy']:.1f}% | {100 * metrics['worst_case_accuracy']:.1f}% | "
                     f"{metrics['binary_macro_f1']:.3f} | {100 * low:.1f}–{100 * high:.1f}% |")
    lines.extend(["", "## Training-only scalar controls", "",
                  "Full-training controls use all five original cases. Ambiguous-training "
                  "controls fit only original nuisance-paired one-pair/all-distinct training "
                  "examples. Each statistic fits a threshold and a contiguous positive "
                  "interval by training accuracy; neither uses challenge outcomes. These "
                  "controls receive complete-object grouping to calculate their scores.", "",
                  "| Training population / scalar rule | Seed accuracy range | Worst-case range |", "|---|---:|---:|"])
    names = list(dict.fromkeys(run["model"] for run in results if run["kind"] == "scalar"))
    for name in names:
        chosen = [run["test"] for run in results if run["model"] == name]
        def span(key):
            values = [100 * item[key] for item in chosen]
            return f"{min(values):.1f}–{max(values):.1f}%"
        lines.append(f"| {name} | {span('accuracy')} | {span('worst_case_accuracy')} |")
    lines.extend(["", "## Rejection and separation", "",
                  "| Seed | Rounds | All proposed pairs | Eligible all-distinct pairs | Retained pairs | Eligible-pair retention |",
                  "|---|---:|---:|---:|---:|---:|"])
    for seed in protocol["seeds"]:
        audit = audits[str(seed)]
        # matching_audit is copied below from the generated challenge.
        rejection = audit["matching_audit"]
        eligible = sum(rejection["candidate_pairs_by_stratum"].values())
        retained = rejection["retained_pairs"]
        lines.append(f"| {seed} | {rejection['rounds']} | {rejection['proposed_pairs']} | "
                     f"{eligible} | {retained} | {100 * retained / eligible:.1f}% |")
    lines.extend(["", f"Every challenge contains {protocol['challenge_size'] // 2} examples of each case. "
                  "Pixel hashes verify no overlap with training, validation, "
                  "or the primary final test; canonical appearance pools are disjoint from "
                  "training/validation. Challenge and primary test deliberately share the "
                  "held-out appearance pool. NPZ artifacts preserve exact inputs, predictions, "
                  "scores, bins and pair identifiers; selection manifests record checkpoint hashes.", "",
                  "## Interpretation and limits", "",
                  "A deterministic classifier using only the frozen bin scores exactly 50% "
                  "by construction: every bin contains equal numbers of both labels. This "
                  "does not remove continuous score information within bins, other diversity "
                  "statistics, or all possible shortcuts. Low rejection retention reshapes "
                  "difficulty and excludes three negative families, so this is not a replacement "
                  "for the primary five-case test. All-matching negatives have a unique zero "
                  "diversity atom and cannot be matched to positive diversity under this rule.", "",
                  "Paired bootstrap intervals quantify test-image uncertainty conditional on "
                  "one checkpoint. Three training seeds are reported separately; their range "
                  "is not a confidence interval. Results compare frozen trained solutions, "
                  "not model expressivity or a proof that attention is required.", "",
                  f"Artifacts: [CSV]({stem}.csv), [JSON]({stem}.json), "
                  f"[figure]({stem}.png). Run artifacts: `{output_dir}`."])
    (report_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
    write_figure(results, audits, protocol, report_dir / f"{stem}.png")


def write_figure(results, audits, protocol, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    plotted = [run for run in results if run["kind"] == "learned" or run["model"] in (
        "ambiguous_training/hamming_interval", "ambiguous_training/frozen_hamming_bin")]
    positions = np.arange(len(plotted))
    labels = [f"{run['model'].replace('ambiguous_training/', '')}\nseed {run['seed']}"
              for run in plotted]
    for ax, key, title in zip(axes, ("accuracy", "worst_case_accuracy"),
                              ("Matched challenge: overall", "Matched challenge: worst of two cases"), strict=True):
        ax.bar(positions, [100 * run["test"][key] for run in plotted],
               color=["#1b9e77" if run["kind"] == "learned" else "#7570b3" for run in plotted])
        ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        if key == "accuracy":
            ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    fig.suptitle(f"Train-defined Hamming bins; {protocol['challenge_size']} images per seed")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mondriaan_oct11_matched_challenge"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-protocol", type=Path,
                        default=Path("runs/mondriaan_oct11_attention/batch_protocol.json"))
    args = parser.parse_args()
    if args.phase == "evaluate":
        evaluate_frozen(args.output_dir, args.report_dir)
        return
    config = config_from_primary(args.batch_protocol)
    freeze(args.output_dir, config)
    print(f"Challenge protocol and training-only policies frozen in {args.output_dir}")


if __name__ == "__main__":
    main()
