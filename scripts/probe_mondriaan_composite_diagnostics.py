"""Frozen-checkpoint composite scaling and learned local-model -> OR diagnostics.

First run --stage train-local, then --stage evaluate after the composite runner
has finished. Local fitting reconstructs the exact constituent development banks
from the composite protocol. Test reconstruction and all scale/count diagnostics
happen only during evaluate; none changes a checkpoint or selects a new one.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.probe_mondriaan_benchmark import (
    build_model, evaluate, grouped_test_uncertainty, make_local_splits, tensor_data, train,
)
from scripts.mondriaan_metrics import CASE_NAMES, classification_metrics
from sparse_wsi_vit.datasets.mondriaan_dataset.composite import (
    aggregate_local_labels, build_balanced_composite, composite_subgroup_metrics,
    lattice_origins, opposite_label_global_count_fixture, patch_tokens,
    scene_pool_from_data, verify_composite, verify_split_pools,
)


def _rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def _hashes(images):
    return [hashlib.sha256(image.tobytes()).hexdigest() for image in images]


def _protocol(benchmark_dir):
    protocol = json.loads((benchmark_dir / "protocol.json").read_text())
    if (protocol["task"] != "composite" or protocol["rule"] != "or"
            or protocol["scene_count"] != 4 or protocol["canvas_size"] != 80
            or protocol["scene_layout_size"] != 80 or protocol["coordinate_scale"] != 80):
        raise ValueError("This frozen diagnostic expects the declared S80/M4/OR/scale80 protocol")
    return protocol, argparse.Namespace(**protocol)


def train_local(benchmark_dir, output_dir, *, steps=2000):
    protocol, args = _protocol(benchmark_dir)
    local_protocol = {"parent_protocol": protocol, "model": "mean", "steps": steps,
        "eval_every": 400, "coordinate_scale": 32, "learning_rate": protocol["lr"],
        "test_policy": "No test scenes generated in train-local stage",
        "development_policy": "Exact constituent training/validation scenes reconstructed from parent seeds",
        "selection": "Local validation worst repetition-case, binary macro-F1, then accuracy",
        "diagnostic_scope": "2000-update local model, not a budget-matched comparison to 400-update full canvas"}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "local_protocol.json"
    if destination.exists():
        raise ValueError("Local protocol already exists; choose a new output directory")
    destination.write_text(json.dumps(local_protocol, indent=2) + "\n")
    torch.set_num_threads(args.threads)
    for seed in args.seeds:
        _, local_training, local_validation, _ = make_local_splits(args, seed)
        training, validation = tensor_data(local_training), tensor_data(local_validation)
        training["intended_cases"] = validation["intended_cases"] = CASE_NAMES
        torch.manual_seed(seed)
        model = build_model("mean", 32)
        directory = output_dir / f"local_mean_seed{seed}"
        report = train(model, training, validation, seed=seed, steps=steps,
            batch_size=args.batch_size, lr=args.lr, eval_every=400, output_dir=directory,
            settings={"model": "local_mean", "seed": seed, **local_protocol})
        report["constituent_training_image_sha256"] = _hashes(local_training["images"])
        report["constituent_validation_image_sha256"] = _hashes(local_validation["images"])
        (directory / "training.json").write_text(json.dumps(report, indent=2) + "\n")


def pad_canvas(data, size):
    """Preserve every pixel, identity, origin and nuisance pair; append black space."""
    old_size = data["canvas_size"]
    if size < old_size or size % 4:
        raise ValueError("Padding requires a larger patch-aligned size")
    result = dict(data)
    result["images"] = np.pad(data["images"], ((0, 0), (0, size - old_size),
                                              (0, size - old_size), (0, 0)))
    tokens = [patch_tokens(image) for image in result["images"]]
    result["x"] = np.stack([item[0] for item in tokens])
    result["coords"] = np.stack([item[1] for item in tokens])
    result["canvas_size"] = size
    result["recommended_rope_high"] = size
    return result


def _normalize_cases(data):
    result = dict(data)
    result["case_mixes"] = result["cases"]
    result["cases"] = [f"positive_scene_count_{count}" for count in result["positive_scene_counts"]]
    result = tensor_data(result)
    result["intended_cases"] = tuple(f"positive_scene_count_{n}" for n in range(result["scene_count"] + 1))
    return result


def _evaluate_canvas(model, raw_data, *, seed, method, selected_step, sweep):
    data = _normalize_cases(raw_data)
    start = time.perf_counter()
    metrics, predictions = evaluate(model, data, batch_size=32, return_predictions=True)
    seconds = time.perf_counter() - start
    subgroup = composite_subgroup_metrics(predictions, raw_data)
    return {"seed": seed, "method": method, "sweep": sweep, "selected_step": selected_step,
        "canvas_size": raw_data["canvas_size"], "scene_count": raw_data["scene_count"],
        "coordinate_scale": 80, "samples": len(raw_data["y"]), "tokens": data["x"].shape[1],
        "foreground_fraction": 64 * raw_data["scene_count"] / raw_data["canvas_size"] ** 2,
        "test": metrics, "canvas_subgroups": subgroup, "inference_seconds": seconds,
        "images_per_second": len(raw_data["y"]) / seconds,
        "process_cumulative_peak_rss_bytes": _rss(),
        "memory_scope": "Whole-process high-water RSS including data, models and earlier evaluations; not isolated model peak",
        "test_sample_uncertainty": grouped_test_uncertainty(data, predictions, seed)}


@torch.inference_mode()
def _local_diagnostic(model, raw_data, *, seed, selected_step):
    shape = raw_data["local_x"].shape
    local = {"x": torch.from_numpy(raw_data["local_x"].reshape(-1, 64, 48)),
             "coords": torch.from_numpy(raw_data["local_coords"].reshape(-1, 64, 2)),
             "y": torch.from_numpy(raw_data["local_labels"].ravel()),
             "cases": np.asarray(raw_data["local_cases"]).ravel().tolist(),
             "intended_cases": CASE_NAMES}
    start = time.perf_counter()
    local_metrics, local_predictions = evaluate(model, local, batch_size=32, return_predictions=True)
    predictions = aggregate_local_labels(local_predictions.reshape(shape[:2]), rule="or")
    seconds = time.perf_counter() - start
    subgroup = composite_subgroup_metrics(predictions, raw_data)
    normalized = _normalize_cases(raw_data)
    canvas_metrics = classification_metrics(raw_data["y"], predictions, normalized["cases"],
                                            intended_cases=normalized["intended_cases"])
    wrong_local = (local_predictions.reshape(shape[:2]) != raw_data["local_labels"])
    count_errors = Counter(wrong_local.sum(axis=1).tolist())
    return {"seed": seed, "method": "local_mean_then_or", "sweep": "hierarchical_diagnostic",
            "selected_step": selected_step, "canvas_size": raw_data["canvas_size"],
            "scene_count": raw_data["scene_count"], "coordinate_scale": 32,
            "samples": len(raw_data["y"]), "tokens": 64 * raw_data["scene_count"],
            "foreground_fraction": 64 * raw_data["scene_count"] / raw_data["canvas_size"] ** 2,
            "test": canvas_metrics, "local_test": local_metrics, "canvas_subgroups": subgroup,
            "local_confusion_matrix": local_metrics["confusion_matrix"],
            "canvas_confusion_matrix": canvas_metrics["confusion_matrix"],
            "wrong_local_predictions_per_canvas": dict(sorted(count_errors.items())),
            "canvas_errors_despite_all_local_correct": int(np.sum((predictions != raw_data["y"])
                                                                   & ~wrong_local.any(axis=1))),
            "inference_seconds": seconds, "images_per_second": len(raw_data["y"]) / seconds,
            "process_cumulative_peak_rss_bytes": _rss(),
            "memory_scope": "Whole-process high-water RSS; hierarchical token count excludes blank inter-region canvas",
            "test_sample_uncertainty": grouped_test_uncertainty(normalized, predictions, seed)}


def evaluate_frozen(benchmark_dir, output_dir):
    protocol, args = _protocol(benchmark_dir)
    if not (benchmark_dir / "results.json").exists():
        raise ValueError("Full-canvas training/test evaluation must finish before frozen diagnostics")
    torch.set_num_threads(args.threads)
    results, checks, fixture_results = [], {}, []
    for seed in args.seeds:
        _, local_training, local_validation, create_local_test = make_local_splits(args, seed)
        train_bank = scene_pool_from_data(local_training)
        validation_bank = scene_pool_from_data(local_validation)
        local_test = create_local_test()
        test_bank = scene_pool_from_data(local_test)
        checks[str(seed)] = {"split_checks": verify_split_pools([train_bank, validation_bank, test_bank]),
                            "scales": {}}
        origins = lattice_origins(80, 4)
        base = build_balanced_composite(test_bank, n_samples=args.test_size,
            canvas_size=80, scene_count=4, rule="or", seed=seed + 9000,
            palette_mode="shared_per_canvas", fixed_origins=origins)
        manifest = json.loads((benchmark_dir / f"test_manifest_seed{seed}.json").read_text())
        if _hashes(base["images"]) != manifest["image_sha256"]:
            raise ValueError("Reconstructed base canvases disagree with the frozen test manifest")
        for split, data in [("training", local_training), ("validation", local_validation)]:
            report = json.loads((output_dir / f"local_mean_seed{seed}" / "training.json").read_text())
            if _hashes(data["images"]) != report[f"constituent_{split}_image_sha256"]:
                raise ValueError("Learned local diagnostic used different constituent development scenes")
        views = [("padding", pad_canvas(base, size)) for size in [80, 128, 256]]
        slots128 = lattice_origins(128, 9)
        expanded_origins = np.array(list(origins) + [p for p in slots128 if tuple(p) not in {tuple(o) for o in origins}])
        for m in [1, 4, 9]:
            view = build_balanced_composite(test_bank, n_samples=args.test_size,
                canvas_size=128, scene_count=m, rule="or", seed=seed + 9000,
                palette_mode="shared_per_canvas", fixed_origins=expanded_origins[:m])
            if m == 4 and not np.array_equal(view["images"], views[1][1]["images"]):
                raise ValueError("M4 count sweep differs from the same M4 padded canvases")
            views.append(("scene_count", view))
        for sweep, view in views:
            checks[str(seed)]["scales"][f"{sweep}_S{view['canvas_size']}_M{view['scene_count']}"] = verify_composite(view, test_bank)
        for method in args.models:
            checkpoint = torch.load(benchmark_dir / f"{method}_seed{seed}" / "best.pt", weights_only=True)
            model = build_model(method, 80, args.width)
            model.load_state_dict(checkpoint["model_state"])
            for sweep, view in views:
                result = _evaluate_canvas(model, view, seed=seed, method=method,
                    selected_step=checkpoint["step"], sweep=sweep)
                result["checkpoint_sha256"] = hashlib.sha256((benchmark_dir / f"{method}_seed{seed}" / "best.pt").read_bytes()).hexdigest()
                if sweep == "padding" and view["canvas_size"] == 80:
                    original = json.loads((benchmark_dir / f"{method}_seed{seed}" / "test.json").read_text())
                    if result["test"]["accuracy"] != original["test"]["accuracy"]:
                        raise ValueError("Frozen baseline evaluation differs from original selected checkpoint")
                results.append(result)
            fixture = opposite_label_global_count_fixture()
            model.eval()
            with torch.inference_mode():
                logits = model(torch.from_numpy(fixture["x"]), torch.from_numpy(fixture["coords"]))["logits"]
            fixture_results.append({"seed": seed, "model": method, "truth": [0, 1],
                "prediction": logits.argmax(dim=1).tolist(), "positive_probability": logits.softmax(dim=1)[:, 1].tolist(),
                "maximum_logit_difference": float((logits[0] - logits[1]).abs().max())})
        local_checkpoint = torch.load(output_dir / f"local_mean_seed{seed}" / "best.pt", weights_only=True)
        local_model = build_model("mean", 32, args.width)
        local_model.load_state_dict(local_checkpoint["model_state"])
        results.append(_local_diagnostic(local_model, base, seed=seed, selected_step=local_checkpoint["step"]))
        print(f"Completed frozen composite diagnostics seed{seed}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "scaling_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output_dir / "fixture_predictions.json").write_text(json.dumps(fixture_results, indent=2) + "\n")
    rows = []
    for result in results:
        row = {key: value for key, value in result.items() if not isinstance(value, (dict, list))}
        row.update({name: result["test"].get(name) for name in ["accuracy", "worst_case_accuracy", "macro_case_accuracy",
                                                              "positive_f1", "binary_macro_f1"]})
        row["worst_case_mix_accuracy"] = result["canvas_subgroups"]["worst_subgroup_accuracy"]
        if "local_test" in result:
            row["local_accuracy"] = result["local_test"]["accuracy"]
            row["local_worst_case_accuracy"] = result["local_test"]["worst_case_accuracy"]
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train-local", "evaluate"), required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-steps", type=int, default=2000)
    args = parser.parse_args()
    if args.stage == "train-local":
        train_local(args.benchmark_dir, args.output_dir, steps=args.local_steps)
    else:
        evaluate_frozen(args.benchmark_dir, args.output_dir)


if __name__ == "__main__":
    main()
