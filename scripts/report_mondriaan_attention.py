"""Audit and summarize a frozen, equal-update ViT/TransMIL comparison.

This command fits only prespecified scalar diagnostics on primary training
images. Learned checkpoints are re-evaluated without any selection or tuning.
Bootstrap intervals condition on fitted checkpoints and resample complete
nuisance pairs; the observed seed ranges describe a different uncertainty.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.audit_mondriaan_appearances import fit_interval, hamming_diversity, predict, rgb_diversity
from scripts.mondriaan_metrics import CASE_NAMES, classification_metrics
from scripts.probe_mondriaan_benchmark import (
    build_model, checkpoint_rank, evaluate, flatten_metrics, make_local_splits, tensor_data,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import audit_splits, dataset_checks


MODELS = ("mean", "vit5", "transmil")
SCALARS = ("rgb_diversity", "normalized_rgb_diversity", "hamming_diversity")
IGNORED_PROTOCOL_KEYS = {"output_dir", "seeds", "resume_prefix"}


def aligned_patch_equality_oracle(x):
    """Apply strict-one-pair equality using only complete aligned patch tokens.

    The -1 sentinel marks background tokens. No coordinates, identities,
    segmentation metadata, palettes, or case labels enter this computation.
    This information control applies only when each complete object is exactly
    one patch; splitting an object across patches changes its premise.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] != 48 or not np.isfinite(x).all():
        raise ValueError("Oracle requires finite lossless (images, tokens, 48) patches")
    result = []
    for patches in x:
        foreground = patches[~np.all(patches == -1, axis=1)]
        if len(foreground) != 4:
            raise ValueError("Aligned oracle requires exactly four non-background object patches")
        multiplicities = Counter(np.ascontiguousarray(patch).tobytes() for patch in foreground)
        result.append(int(sorted(multiplicities.values()) == [1, 1, 2]))
    return np.asarray(result, dtype=np.int64)


def appearance_orbit_checks(pools):
    """Audit appearance geometry modulo a shared permutation of colour names."""
    def canonical(pattern):
        names = {}
        normalized = []
        for value in pattern.ravel():
            names.setdefault(int(value), len(names))
            normalized.append(names[int(value)])
        return tuple(normalized)
    orbits = {split: {canonical(pattern) for pattern in pool.patterns}
              for split, pool in pools.items()}
    splits = list(orbits)
    return {"unique_geometry_orbits": {split: len(values) for split, values in orbits.items()},
            "overlaps": {f"{left}:{right}": len(orbits[left] & orbits[right])
                         for index, left in enumerate(splits) for right in splits[index + 1:]}}


def compatible_protocols(protocols):
    """Reject a changed budget/input or incomplete/duplicate seed set."""
    if not protocols:
        raise ValueError("No frozen protocols supplied")
    reference = {key: value for key, value in protocols[0].items()
                 if key not in IGNORED_PROTOCOL_KEYS}
    seeds = []
    for protocol in protocols:
        normalized = {key: value for key, value in protocol.items()
                      if key not in IGNORED_PROTOCOL_KEYS}
        if normalized != reference:
            changes = sorted(key for key in set(normalized) | set(reference)
                             if normalized.get(key) != reference.get(key))
            raise ValueError(f"Independent seed protocols differ: {changes}")
        seeds.extend(protocol["seeds"])
    if len(seeds) != len(set(seeds)) or set(seeds) != {42, 7, 2026}:
        raise ValueError("Exactly three independent seeds 42, 7, 2026 are required")
    required = {"task": "local", "variant": "pixels16", "palette": "shared",
                "placement": "aligned", "representation": "scene", "object_counts": [4],
                "appearance_pool_size": 256, "train_size": 4096,
                "validation_size": 512, "test_size": 1024, "steps": 4000,
                "eval_every": 400, "batch_size": 32, "lr": 2e-4}
    for key, value in required.items():
        if reference.get(key) != value:
            raise ValueError(f"Unexpected frozen comparison setting: {key}")
    if (len(reference.get("models", [])) != len(MODELS)
            or set(reference.get("models", [])) != set(MODELS)):
        raise ValueError("Frozen comparison must include exactly mean pooling, ViT and TransMIL")
    return {**protocols[0], "seeds": seeds}


def verify_manifest(data, manifest):
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in data["images"]]
    if (manifest.get("samples") != len(hashes)
            or manifest.get("split") != "test" or manifest.get("image_sha256") != hashes):
        raise ValueError("Regenerated final test differs from the frozen image manifest")
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def verify_selection(training_report, protocol):
    history = training_report["history"]
    expected_steps = list(range(0, protocol["steps"] + 1, protocol["eval_every"]))
    if expected_steps[-1] != protocol["steps"]:
        expected_steps.append(protocol["steps"])
    if [record["step"] for record in history] != expected_steps:
        raise ValueError("Training did not complete the frozen validation schedule")
    # max preserves the earliest checkpoint on an exact rank tie, as training does.
    selected = max(history, key=lambda record: checkpoint_rank(record["validation"]))
    if (training_report["selected_step"] != selected["step"]
            or tuple(training_report["selected_rank"]) != checkpoint_rank(selected["validation"])):
        raise ValueError("Saved selection violates the frozen validation checkpoint policy")
    return selected


def verify_checkpoint(state, protocol, seed, model_name, *, expected_step):
    if state["step"] != expected_step:
        raise ValueError("Checkpoint update does not match the saved checkpoint policy")
    settings = state["settings"]
    for key, value in protocol.items():
        if key in IGNORED_PROTOCOL_KEYS or key == "coordinate_scale":
            continue
        if settings.get(key) != value:
            raise ValueError(f"Checkpoint construction/training setting changed: {key}")
    if settings.get("seed") != seed or settings.get("model") != model_name:
        raise ValueError("Checkpoint seed/model does not match the experiment")
    if settings.get("coordinate_scale") != (protocol.get("coordinate_scale") or 32):
        raise ValueError("Checkpoint coordinate scale changed")


def verify_metrics(actual, recorded):
    keys = ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "positive_precision",
            "positive_recall", "positive_f1", "binary_macro_f1", "by_case", "confusion_matrix")
    for key in keys:
        if actual[key] != recorded[key]:
            raise ValueError(f"Checkpoint re-evaluation changed recorded metric: {key}")


def paired_nuisance_bootstrap(labels, cases, predictions, pair_ids, *, seed, repeats=2000):
    """Use one cluster resampling stream for every method and checkpoint."""
    labels, cases, pair_ids = map(np.asarray, (labels, cases, pair_ids))
    if not (labels.ndim == cases.ndim == pair_ids.ndim == 1
            and len(labels) == len(cases) == len(pair_ids)):
        raise ValueError("Labels, cases and nuisance pair IDs must be equal-length vectors")
    groups = defaultdict(list)
    for index, pair in enumerate(pair_ids):
        groups[int(pair)].append(index)
    members = list(groups.values())
    if any(len(group) != 2 or set(labels[group]) != {0, 1} for group in members):
        raise ValueError("Each nuisance cluster must retain both opposite-label images")
    counts = np.array([[np.sum(cases[group] == case) for case in CASE_NAMES]
                       for group in members])
    if not counts.sum(axis=0).all():
        raise ValueError("Full-task uncertainty requires all five observed cases")
    rng = np.random.default_rng(seed + 18000)
    draws = rng.integers(len(members), size=(repeats, len(members)))
    totals = counts[draws].sum(axis=1)
    if (totals == 0).any():
        raise ValueError("A bootstrap sample omitted an intended subgroup")
    samples, intervals = {}, {}
    for method, predicted in predictions.items():
        predicted = np.asarray(predicted)
        classification_metrics(labels, predicted, cases, intended_cases=CASE_NAMES)
        correct = labels == predicted
        correct_counts = np.array([[np.sum(correct[group] & (cases[group] == case))
                                    for case in CASE_NAMES] for group in members])
        correct_totals = correct_counts[draws].sum(axis=1)
        case_accuracy = correct_totals / totals
        confusion = np.array([[np.sum((labels[group] == y) & (predicted[group] == p))
                               for y, p in ((0, 0), (0, 1), (1, 0), (1, 1))] for group in members])
        tn, fp, fn, tp = confusion[draws].sum(axis=1).T
        def divide(numerator, denominator):
            return np.divide(numerator, denominator, out=np.zeros(repeats), where=denominator != 0)
        pos_f1, neg_f1 = divide(2 * tp, 2 * tp + fp + fn), divide(2 * tn, 2 * tn + fp + fn)
        values = {"accuracy": correct_totals.sum(axis=1) / totals.sum(axis=1),
                  "worst_case_accuracy": case_accuracy.min(axis=1),
                  "macro_case_accuracy": case_accuracy.mean(axis=1),
                  "positive_precision": divide(tp, tp + fp), "positive_recall": divide(tp, tp + fn),
                  "positive_f1": pos_f1, "binary_macro_f1": (pos_f1 + neg_f1) / 2,
                  **{f"accuracy_{case}": case_accuracy[:, index]
                     for index, case in enumerate(CASE_NAMES)}}
        samples[method] = values
        intervals[method] = {metric: np.quantile(value, [.025, .975]).tolist()
                             for metric, value in values.items()}
    differences = {}
    for first_name, second_name in (("vit5", "transmil"), ("mean", "vit5"), ("mean", "transmil")):
        comparison = f"{second_name}_minus_{first_name}"
        differences[comparison] = {}
        for mode in ("selected", "final"):
            first, second = f"{first_name}/{mode}", f"{second_name}/{mode}"
            if first in samples and second in samples:
                differences[comparison][mode] = {
                    metric: np.quantile(samples[second][metric] - samples[first][metric], [.025, .975]).tolist()
                    for metric in samples[first]}
    return {"groups": len(members), "repeats": repeats, "intervals": intervals,
            "paired_model_difference_intervals": differences,
            "interpretation": "95% percentile bootstrap of whole opposite-label nuisance pairs, conditional on fixed checkpoints; not training-seed uncertainty"}


def make_checkpoint_model(model_name, protocol):
    return build_model(model_name, protocol.get("coordinate_scale") or 32, protocol["width"],
                       transmil_landmarks=protocol["transmil_landmarks"])


def collect(run_dirs, *, repeats=2000):
    from scripts.probe_mondriaan_attention_batch import preflight

    roots = {directory.parent.resolve() for directory in run_dirs}
    if len(roots) != 1:
        raise ValueError("Independent seed directories must belong to one frozen batch")
    batch_root = next(iter(roots))
    batch, _ = preflight(batch_root)
    completion = json.loads((batch_root / "evaluation_complete.json").read_text())
    if (completion.get("batch_id") != batch["batch_id"]
            or completion.get("models_evaluated") != len(batch["seeds"]) * len(MODELS)):
        raise ValueError("Primary evaluation did not complete the frozen nine-fit batch")
    protocols = [json.loads((directory / "protocol.json").read_text()) for directory in run_dirs]
    protocol = compatible_protocols(protocols)
    if protocol["batch_id"] != batch["batch_id"]:
        raise ValueError("Seed protocols do not belong to the completed frozen batch")
    orbit_audit = json.loads((batch_root / "appearance_orbit_audit.json").read_text())
    if orbit_audit.get("uses_final_test_images") is not False:
        raise ValueError("Appearance-orbit audit must concern pools rather than final test images")
    rows, diagnostics, arrays = [], {}, {}
    for root, seed_protocol in zip(run_dirs, protocols, strict=True):
        results = json.loads((root / "results.json").read_text())
        expected = {(seed, model) for seed in seed_protocol["seeds"] for model in MODELS}
        if (len(results) != len(expected)
                or {(record["seed"], record["model"]) for record in results} != expected):
            raise ValueError("Experiment has incomplete or duplicated model/seed results")
        if any(record.get("batch_id") != batch["batch_id"] for record in results):
            raise ValueError("Primary results belong to a different frozen batch")
        torch.set_num_threads(seed_protocol["threads"])
        for seed in seed_protocol["seeds"]:
            args = argparse.Namespace(**seed_protocol)
            pools, training, validation, create_test = make_local_splits(args, seed)
            orbit_checks = appearance_orbit_checks(pools)
            if orbit_checks != orbit_audit["seeds"][str(seed)]:
                raise ValueError("Appearance-orbit audit changed on reconstructed pools")
            if (any(orbit_checks["overlaps"].values())
                    or set(orbit_checks["unique_geometry_orbits"].values()) != {256}):
                raise ValueError("Frozen appearance pools share colour-renaming geometry")
            raw_test = create_test()
            manifest = json.loads((root / f"test_manifest_seed{seed}.json").read_text())
            if manifest.get("batch_id") != batch["batch_id"]:
                raise ValueError("Primary test manifest belongs to a different frozen batch")
            manifest_digest = verify_manifest(raw_test, manifest)
            checks = {split: dataset_checks(data) for split, data in
                      (("train", training), ("validation", validation), ("test", raw_test))}
            checks["overlap"] = audit_splits({"train": training, "validation": validation,
                                              "test": raw_test}, require_disjoint_appearances=True)
            data = tensor_data(raw_test)
            data["intended_cases"] = CASE_NAMES
            exact_predictions = aligned_patch_equality_oracle(data["x"])
            oracle_metrics = classification_metrics(raw_test["y"], exact_predictions, data["cases"],
                                                    intended_cases=CASE_NAMES)
            if oracle_metrics["accuracy"] != 1.0:
                raise ValueError("Full aligned patch inputs do not reproduce the exact matching labels")
            predictions = {}
            arrays[f"labels_seed{seed}"] = raw_test["y"]
            arrays[f"cases_seed{seed}"] = np.asarray(data["cases"])
            arrays[f"nuisance_pair_seed{seed}"] = raw_test["pair_ids"]
            arrays[f"predictions_full_patch_oracle_seed{seed}"] = exact_predictions
            for name in MODELS:
                directory = root / f"{name}_seed{seed}"
                record = next(row for row in results if row["model"] == name and row["seed"] == seed)
                training_report = json.loads((directory / "training.json").read_text())
                selected = verify_selection(training_report, seed_protocol)
                if selected["step"] != record["selected_step"]:
                    raise ValueError("Results changed the training-selected checkpoint")
                with np.load(directory / "test_predictions.npz") as saved_predictions:
                    for key, expected_array in (("targets", raw_test["y"]), ("cases", data["cases"]),
                                                ("pair_ids", raw_test["pair_ids"])):
                        if not np.array_equal(saved_predictions[key], np.asarray(expected_array)):
                            raise ValueError(f"Stored test prediction inputs changed: {key}")
                    frozen_predictions = {"selected": saved_predictions["predictions"].copy(),
                                          "final": saved_predictions["final_predictions"].copy()}
                for mode, filename, expected_step, metrics_key, val in (
                    ("selected", "best.pt", selected["step"], "test", selected["validation"]),
                    ("final", "final.pt", seed_protocol["steps"], "test_final_update",
                     training_report["history"][-1]["validation"]),
                ):
                    state = torch.load(directory / filename, weights_only=True)
                    verify_checkpoint(state, seed_protocol, seed, name, expected_step=expected_step)
                    model = make_checkpoint_model(name, seed_protocol)
                    model.load_state_dict(state["model_state"])
                    parameters = sum(parameter.numel() for parameter in model.parameters())
                    if parameters != record["parameters"] or parameters != state["settings"]["parameter_count"]:
                        raise ValueError("Model parameter count changed during reconstruction")
                    started = time.perf_counter()
                    metrics, predicted = evaluate(model, data, seed_protocol["batch_size"], return_predictions=True)
                    reevaluation_seconds = time.perf_counter() - started
                    verify_metrics(metrics, record[metrics_key])
                    if not np.array_equal(predicted, frozen_predictions[mode]):
                        raise ValueError("Re-evaluated checkpoint predictions changed")
                    rows.append({"method": name, "kind": "learned", "seed": seed, "checkpoint": mode,
                                 "step": state["step"], "parameters": parameters,
                                 "training_seconds": record["seconds"], "tokens": record["tokens"],
                                 "recorded_selected_inference_seconds": record["test_inference_seconds"],
                                 "reevaluation_inference_seconds": reevaluation_seconds,
                                 "process_peak_rss_bytes": record["process_peak_rss_bytes"],
                                 "memory_scope": record["memory_scope"], "test": metrics, "validation": val,
                                 "train_final": training_report["train_final"],
                                 "validation_final": training_report["history"][-1]["validation"],
                                 "checkpoint_settings": state["settings"], "test_manifest_digest": manifest_digest})
                    predictions[f"{name}/{mode}"] = predicted
                    arrays[f"predictions_{name}_{mode}_seed{seed}"] = predicted
            statistics = {"rgb_diversity": rgb_diversity,
                          "normalized_rgb_diversity": lambda data: rgb_diversity(data, normalize_palette=True),
                          "hamming_diversity": hamming_diversity}
            for name, statistic in statistics.items():
                rule = fit_interval(statistic(training), training["y"])
                predicted = predict(rule, statistic(raw_test))
                metrics = classification_metrics(raw_test["y"], predicted, data["cases"], intended_cases=CASE_NAMES)
                rows.append({"method": name, "kind": "complete_object_scalar", "seed": seed,
                             "checkpoint": "training_interval", "step": None, "parameters": 0,
                             "training_seconds": None, "tokens": None,
                             "recorded_selected_inference_seconds": None,
                             "reevaluation_inference_seconds": None, "process_peak_rss_bytes": None,
                             "memory_scope": "not measured", "training_rule": rule, "test": metrics,
                             "test_manifest_digest": manifest_digest})
                predictions[name] = predicted
                arrays[f"predictions_{name}_seed{seed}"] = predicted
            diagnostics[str(seed)] = {
                "source_run_directory": str(root), "source_checks": checks,
                "appearance_geometry_orbits": orbit_checks,
                "full_patch_equality_oracle": {"metrics": oracle_metrics,
                                               "input": "full lossless patch x only; -1 background sentinel; exact four-patch duplicate multiplicities",
                                               "uses_coordinates_or_true_ids": False,
                                               "scope": "nonlearned information control on aligned one-object-per-patch scenes"},
                "test_manifest_digest": manifest_digest, "all_test_pixels_match_frozen_manifest": True,
                "checkpoint_metrics_reproduced": True,
                "uncertainty": paired_nuisance_bootstrap(raw_test["y"], data["cases"], predictions,
                                                        raw_test["pair_ids"], seed=seed, repeats=repeats),
            }
    return protocol, rows, diagnostics, arrays


def flattened_rows(rows):
    output = []
    for record in rows:
        row = {key: record[key] for key in (
            "method", "kind", "seed", "checkpoint", "step", "parameters", "training_seconds", "tokens",
            "recorded_selected_inference_seconds", "reevaluation_inference_seconds", "process_peak_rss_bytes",
            "memory_scope", "test_manifest_digest")}
        row.update({f"test_{key}": value for key, value in flatten_metrics(record["test"]).items()})
        row.update({f"validation_{metric}": record.get("validation", {}).get(metric)
                    for metric in ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "binary_macro_f1")})
        for split in ("train_final", "validation_final"):
            row.update({f"{split}_{metric}": record.get(split, {}).get(metric)
                        for metric in ("accuracy", "worst_case_accuracy", "macro_case_accuracy", "binary_macro_f1")})
        output.append(row)
    return output


def write_figure(path, rows, diagnostics, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    for column, mode in enumerate(("selected", "final")):
        for row_index, metric in enumerate(("accuracy", "worst_case_accuracy")):
            ax = axes[row_index, column]
            for method, shift, marker, color in (("mean", -.16, "^", "#6d7a51"),
                                                 ("vit5", 0, "o", "#3563a5"),
                                                 ("transmil", .16, "s", "#c2662a")):
                records = [next(record for record in rows if record["method"] == method
                                and record["checkpoint"] == mode and record["seed"] == seed) for seed in seeds]
                x = np.arange(len(seeds)) + shift
                bounds = np.array([diagnostics[str(seed)]["uncertainty"]["intervals"][f"{method}/{mode}"][metric]
                                   for seed in seeds]) * 100
                y = np.array([record["test"][metric] for record in records]) * 100
                ax.vlines(x, bounds[:, 0], bounds[:, 1], color=color)
                ax.scatter(np.repeat(x, 2), bounds.ravel(), marker="_", color=color)
                ax.scatter(x, y, marker=marker, color=color, label=method, zorder=3)
            ax.set_xticks(range(len(seeds)), list(map(str, seeds)))
            ax.set_ylim(0, 102)
            ax.set_xlabel("Experiment seed")
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"{mode.capitalize()} checkpoint · {metric.replace('_', ' ')}")
            ax.grid(axis="y", alpha=.2)
            ax.legend()
    fig.suptitle("Richer appearances: mean pooling, ViT and TransMIL at 4,000 updates\n"
                 "Intervals resample test nuisance pairs at fixed checkpoints; they do not measure seed uncertainty")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_report(path, protocol, rows, diagnostics, prefix_name):
    def span(records, metric):
        values = [100 * record["test"][metric] for record in records]
        return f"{min(values):.1f}–{max(values):.1f}%"
    selected_rows = [row for row in rows if row["checkpoint"] == "selected"]
    performance = []
    for name in MODELS:
        records = [row for row in selected_rows if row["method"] == name]
        performance.append(f"{name} reaches {span(records, 'accuracy')} overall and "
                           f"{span(records, 'worst_case_accuracy')} on its weakest repetition case")
    transmil_final = [row for row in rows if row["method"] == "transmil" and row["checkpoint"] == "final"]
    def fitted_span(split):
        scores = [100 * row[split]["accuracy"] for row in transmil_final]
        return f"{min(scores):.1f}–{max(scores):.1f}%"
    lines = ["# Richer-appearance mean pooling, ViT and TransMIL comparison", "", "October 11, 2026.", "",
             "At validation-selected checkpoints, " + "; ".join(performance) + ".", "",
             f"At the final 4,000-update checkpoint, TransMIL fits the training images at {fitted_span('train_final')}, "
             f"but reaches {fitted_span('validation_final')} on validation and {span(transmil_final, 'accuracy')} "
             "on test. Its strong training fit accompanies limited transfer to disjoint appearance pools. "
             "The complete-object Hamming diagnostic reaches " + span(
                 [row for row in rows if row["method"] == "hamming_diversity"], "accuracy") + ", "
             "and the full-patch exact-equality oracle reproduces every label. The input retains the "
             "needed information; these learned-model scores do not establish an attention requirement.", "",
             "This is a bounded comparison of three fitted models at the same number of updates. "
             "Validation selects checkpoints; no model sees the final test during fitting. "
             "The table reports observed minima–maxima across three experiment seeds, not confidence intervals.", "",
             "| Checkpoint | Method | Parameters | Overall | Worst case | Macro-case | Positive F1 | Binary macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("selected", "final", "training_interval"):
        for name in (MODELS if mode != "training_interval" else SCALARS):
            records = [record for record in rows if record["method"] == name and record["checkpoint"] == mode]
            lines.append(f"| {mode} | {name} | {records[0]['parameters']:,} | " + " | ".join(
                span(records, metric) for metric in ("accuracy", "worst_case_accuracy", "macro_case_accuracy",
                                                    "positive_f1", "binary_macro_f1")) + " |")
    lines.extend(["", "![Checkpoint comparison](" + prefix_name + "_scores.png)", "",
                  "| Validation-selected model | One pair | All distinct | Two pairs | Triple | All matching |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name in MODELS:
        records = [record for record in selected_rows if record["method"] == name]
        values = []
        for case in ("one_pair", "all_distinct", "two_pairs", "three_matching", "four_matching"):
            scores = [100 * record["test"]["by_case"][case]["accuracy"] for record in records]
            values.append(f"{min(scores):.1f}–{max(scores):.1f}%")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend(["", "| Model / seed | Final train accuracy | Final train worst case | Final validation accuracy | Final validation worst case |",
                  "|---|---:|---:|---:|---:|"])
    for row in rows:
        if row["checkpoint"] == "final":
            scores = [row[split][metric] for split, metric in (
                ("train_final", "accuracy"), ("train_final", "worst_case_accuracy"),
                ("validation_final", "accuracy"), ("validation_final", "worst_case_accuracy"))]
            lines.append(f"| {row['method']} / {row['seed']} | " + " | ".join(
                f"{100*score:.1f}%" for score in scores) + " |")
    lines.extend(["", "Training and validation in this table use the same fixed 4,000-update checkpoint, "
                  "independently of the validation-selected update. High training fit with a validation "
                  "gap indicates limited transfer under the current disjoint-appearance split; low "
                  "training scores indicate difficulty fitting the training task within this budget. "
                  "A gap is compatible with overfitting but does not prove a dictionary-memorization "
                  "strategy, and poor fit does not establish a representational impossibility."])
    lines.extend(["", "| Seed | Checkpoint | Comparison | Overall difference | Paired 95% test-sample interval | Worst-case difference | Paired interval |",
                  "|---:|---|---|---:|---:|---:|---:|"])
    for seed in protocol["seeds"]:
        for mode in ("selected", "final"):
            records = {row["method"]: row for row in rows if row["seed"] == seed and row["checkpoint"] == mode}
            for first, second in (("vit5", "transmil"), ("mean", "vit5"), ("mean", "transmil")):
                cells = []
                for metric in ("accuracy", "worst_case_accuracy"):
                    difference = 100 * (records[second]["test"][metric] - records[first]["test"][metric])
                    interval = diagnostics[str(seed)]["uncertainty"]["paired_model_difference_intervals"]
                    low, high = interval[f"{second}_minus_{first}"][mode][metric]
                    cells.extend([f"{difference:+.1f} pp", f"[{100*low:+.1f}, {100*high:+.1f}] pp"])
                lines.append(f"| {seed} | {mode} | {second} − {first} | " + " | ".join(cells) + " |")
    lines.extend(["", "These intervals resample the 512 complete positive/negative nuisance pairs together, "
                  "with identical bootstrap draws for all models and scalar controls. They condition on "
                  "the fitted checkpoints and estimate finite-test sampling variation. Three training "
                  "seeds support descriptive ranges and paired differences; they do not establish a "
                  "precise population-level confidence interval over training runs.", "",
                  f"The frozen primary task uses {protocol['train_size']:,} training, {protocol['validation_size']:,} "
                  f"validation and {protocol['test_size']:,} test images per seed ({', '.join(map(str, protocol['seeds']))}). "
                  "The separate appearance dictionaries contain 256 identities each. Every identity is a "
                  "freely arranged 4×4 pattern with four pixels of each of four colours. Four objects "
                  "occupy distinct aligned patches in a 32×32 scene. Models receive all 64 lossless "
                  "48-value patches including background. All three models see this identical input. "
                  "Shared random palettes, exact foreground masks "
                  "and object counts are paired across labels. Test denominators are 512 one-pair positives "
                  "and 128 examples of each negative family.", "",
                  f"All three models use {protocol['steps']:,} CPU float32 AdamW updates, batch {protocol['batch_size']}, "
                  f"learning rate {protocol['lr']:g}, weight decay 0.0001 and gradient clipping at 1. "
                  f"Validation is measured at initialization and every {protocol['eval_every']} updates. "
                  "The earliest checkpoint maximizing worst-case accuracy, then binary macro-F1, then "
                  "overall accuracy is selected. Final-update scores are a separate, fixed policy. "
                  "These are equal update budgets and datasets; model capacity, cost per update, "
                  "inductive bias and architecture-specific hyperparameter sensitivity can differ.", "",
                  "A batch preflight first verifies all nine completed fits, artifact checksums and frozen "
                  "source versions before test reconstruction. The report then rebuilds all source splits "
                  "and checks their labels, paired nuisances, "
                  "pixel hashes and appearance overlap. Every final-test image matches its saved manifest. "
                  "Both checkpoint policies are reconstructed using frozen model settings and their metrics "
                  "must reproduce recorded results. Validation histories are audited against the stated "
                  "checkpoint policy. No post-test checkpoint selection or model fitting occurs here.", "",
                  "The independent [appearance-orbit audit](../runs/mondriaan_oct11_attention/appearance_orbit_audit.json) "
                  "normalizes colour names by first occurrence, identifying geometry equivalent under "
                  "shared colour renaming. All nine frozen pools have 256 distinct geometry orbits, "
                  "with zero train/validation/test orbit overlap for every seed; this report independently "
                  "reproduces that pool audit. Identity holdout alone could in principle share such "
                  "geometry, but these particular pools do not. The pool audit generates no final-test images.", "",
                  "A nonlearned exact-equality oracle independently reproduces every primary-test label "
                  "(100% overall and in every repetition case for all three seeds) from full lossless "
                  "patch tokens alone. It discards only the -1 background sentinel, requires exactly "
                  "four foreground 48-value tokens, and returns positive precisely for byte-exact "
                  "duplicate multiplicities [1, 1, 2]. It uses no coordinates, object identities, "
                  "palettes or segmentation metadata. This verifies that the aligned input retains "
                  "the label information. It is an algorithmic control, not a fitted architecture "
                  "comparison; it relies on the current one-complete-object-per-patch alignment and "
                  "does not solve off-grid reconstruction or establish an attention requirement.", "",
                  "TransMIL supplies a spatial correlated-MIL baseline. The "
                  "[original implementation](https://github.com/szc19990412/TransMIL/blob/main/models/TransMIL.py) "
                  "projects input features, applies two Nyström-attention residual layers with a class token, "
                  "and places depthwise 7×7, 5×5 and 3×3 positional convolutions between them. It reshapes "
                  "the patch sequence to a square grid. In this complete 8×8 raster, that grid corresponds "
                  "to true patch adjacency. TransMIL does not provide an explicit scene-level hierarchy; this experiment "
                  "therefore tests a stronger spatial model, while local-to-canvas hierarchical modeling "
                  "remains a separate comparison. The "
                  "[TransMIL paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html) "
                  "motivates correlated multiple-instance learning, rather than a necessity claim for "
                  "global attention on this synthetic puzzle.", "",
                  "The [local design audit](mondriaan_transmil_design.md) documents a physical-grid "
                  "wrapper that sorts tokens and coordinates into raster order, reconstructs the true "
                  "rectangular grid and rejects incomplete or inconsistent coordinates. It uses raw "
                  "48-value patches instead of pretrained WSI features. "
                  f"The adaptation has 512 hidden units, 8 heads and {protocol['transmil_landmarks']} "
                  "Nyström landmarks, replacing the original 256-landmark setting. At 65 attention "
                  "tokens, 32 landmarks internally pad to 96 sequence tokens; 256 landmarks pad to 256 "
                  "and substantially raise runtime. This preserves the spatial physical grid but changes "
                  "the attention approximation, so results refer to this explicitly adapted TransMIL. "
                  "Neither method receives identities or supplied object grouping.", "",
                  "The scalar interval controls are fitted on the same primary training data and evaluated "
                  "on the exact same final-test images. They measure mean pairwise RGB, palette-normalized "
                  "RGB or pixel-Hamming distance using supplied complete-object grouping. They are "
                  "diagnostics, not learned models with equal input access. Their training objective "
                  "maximizes overall accuracy of a contiguous score interval, while checkpoint selection "
                  "for learned models prioritizes worst-case accuracy. Their scores quantify statistical "
                  "shortcuts that can contribute to apparent exact-matching performance. This report "
                  "does not remove those shortcuts or isolate reconstruction, palette robustness, or "
                  "relational reasoning as the source of any model advantage.", "",
                  "| Model / seed | Selected update | Training minutes | Original selected-test seconds | Whole-process RSS GiB |",
                  "|---|---:|---:|---:|---:|"])
    for row in selected_rows:
        lines.append(f"| {row['method']} / {row['seed']} | {row['step']:,} | "
                     f"{row['training_seconds']/60:.2f} | {row['recorded_selected_inference_seconds']:.3f} | "
                     f"{row['process_peak_rss_bytes']/1024**3:.3f} |")
    lines.extend(["", "Training time includes validation and artifact work inside the runner. Latency is "
                  "the CPU wall time for the complete 1,024-image test at batch 32; it is not isolated "
                  "kernel timing. RSS is the original evaluation-process high-water mark, including datasets and "
                  "earlier evaluated models and seeds, not model-specific peak memory. Re-evaluation "
                  "latencies are also retained in CSV/JSON and may differ with process load. "
                  "All reported times come from the measured frozen batch, with no extrapolation from "
                  "the design feasibility microbenchmark. Historical "
                  "2,000-update pooling runs are not ranked as an equal-budget architecture comparison.", "",
                  f"Artifacts: [{prefix_name}_summary.csv]({prefix_name}_summary.csv), "
                  f"[audited JSON]({prefix_name}.json), [predictions]({prefix_name}_predictions.npz). "
                  "Frozen protocols and checkpoints remain in `runs/mondriaan_oct11_attention/seed*`."])
    path.write_text("\n".join(lines) + "\n")


def write_outputs(prefix, protocol, rows, diagnostics, arrays):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table = flattened_rows(rows)
    with prefix.with_name(prefix.name + "_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    def serializable(value):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value
    prefix.with_suffix(".json").write_text(json.dumps(serializable({
        "protocol": protocol, "results": rows, "audits": diagnostics,
        "seed_uncertainty": "three training seeds: observed descriptive ranges, no seed-population confidence interval",
    }), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(prefix.with_name(prefix.name + "_predictions.npz"), **arrays)
    write_figure(prefix.with_name(prefix.name + "_scores.png"), rows, diagnostics, protocol["seeds"])
    write_report(prefix.with_suffix(".md"), protocol, rows, diagnostics, prefix.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=[
        Path(f"runs/mondriaan_oct11_attention/seed{seed}") for seed in (42, 7, 2026)])
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/mondriaan_attention_comparison"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    write_outputs(args.output_prefix, *collect(args.run_dirs, repeats=args.bootstrap_repeats))


if __name__ == "__main__":
    main()
