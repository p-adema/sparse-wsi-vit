"""Freeze, fit, and globally gated evaluation of the approved October 11 batch.

  .venv/bin/python scripts/probe_mondriaan_attention_batch.py --stage freeze
  .venv/bin/python scripts/probe_mondriaan_attention_batch.py --stage fit --seed 42
  .venv/bin/python scripts/probe_mondriaan_attention_batch.py --stage evaluate

Fit seeds may run in separate processes. Evaluation checks every seed/model
before preparing any data or constructing any final-test images.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts import probe_mondriaan_benchmark as benchmark  # noqa: E402


SETTINGS = {
    "task": "local", "variant": "pixels16", "appearance_holdout": False,
    "appearance_pool_size": 256, "appearance_seed_offset": 103000,
    "min_distance": 0, "palette": "shared", "placement": "aligned",
    "representation": "scene", "object_counts": [4],
    "models": ["vit5", "transmil", "mean"], "seeds": [42, 7, 2026],
    "train_size": 4096, "validation_size": 512, "test_size": 1024,
    "test_seed_offset": 366000, "steps": 4000, "restart_at": 2000,
    "resume_prefix": None, "eval_every": 400, "batch_size": 32, "lr": 2e-4,
    "width": None, "transmil_landmarks": 32, "canvas_size": 96,
    "scene_count": 4, "scene_layout_size": None, "coordinate_scale": None,
    "rule": "or", "exact_k": 1, "threads": 1, "device": "cpu",
    "precision": "float32",
    "test_policy": "No primary or challenge test images until all nine fits and validation selections complete",
    "selection": "validation worst-case, binary macro-F1, accuracy",
    "parameter_policy": "mean width nearest ViT capacity; TransMIL original 512 hidden with explicit 32 landmarks",
    "local_rule": "exactly one identity occurs twice; every other identity occurs once",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    paths = [Path(__file__), REPO / "scripts/probe_mondriaan_benchmark.py",
             REPO / "scripts/mondriaan_metrics.py",
             REPO / "src/sparse_wsi_vit/datasets/mondriaan_dataset/controlled.py"]
    paths.extend(REPO / f"src/sparse_wsi_vit/models/{name}.py" for name in
                 ("mondriaan_pooling", "mondriaan_transmil", "transmil", "vit5_dense"))
    paths.extend(sorted((REPO / "src/sparse_wsi_vit/models/vit_5").glob("*.py")))
    return {str(path.relative_to(REPO)): file_hash(path) for path in paths}


def library_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "numpy", "nystrom-attention", "einops", "timm"):
        result[package] = importlib.metadata.version(package)
    return result


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def seed_root(root, seed):
    return Path(root) / f"seed{seed}"


def individual_protocol(batch, seed):
    protocol = copy.deepcopy(batch)
    protocol.update(seeds=[seed], output_dir=str(seed_root(batch["output_dir"], seed)))
    return protocol


def freeze(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Batch root is not empty; refusing to overwrite a frozen protocol")
    torch.set_num_threads(SETTINGS["threads"])
    counts = {}
    for name in SETTINGS["models"]:
        model = benchmark.build_model(name, 32, SETTINGS["width"], SETTINGS["transmil_landmarks"])
        counts[name] = sum(parameter.numel() for parameter in model.parameters())
    batch = {**copy.deepcopy(SETTINGS), "output_dir": str(root), "source_sha256": source_hashes(),
             "library_versions": library_versions(), "parameter_counts": counts,
             "batch_schema": 1}
    batch["batch_id"] = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "batch_protocol.json", batch, exclusive=True)
    for seed in batch["seeds"]:
        directory = seed_root(root, seed)
        directory.mkdir()
        write_json(directory / "protocol.json", individual_protocol(batch, seed), exclusive=True)
    return batch


def load_batch(root):
    root = Path(root).resolve()
    batch = json.loads((root / "batch_protocol.json").read_text())
    for key, expected in SETTINGS.items():
        if batch.get(key) != expected:
            raise ValueError(f"Frozen approved setting changed: {key}")
    if batch["output_dir"] != str(root):
        raise ValueError("Frozen batch root differs from requested root")
    original_id = batch["batch_id"]
    unsigned = {key: value for key, value in batch.items() if key != "batch_id"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != original_id:
        raise ValueError("Frozen batch protocol hash mismatch")
    if source_hashes() != batch["source_sha256"]:
        raise ValueError("Experiment/model source changed since protocol freeze")
    if library_versions() != batch["library_versions"]:
        raise ValueError("Python/library environment changed since protocol freeze")
    for seed in batch["seeds"]:
        saved = json.loads((seed_root(root, seed) / "protocol.json").read_text())
        if saved != individual_protocol(batch, seed):
            raise ValueError(f"Individual protocol differs from frozen batch: seed{seed}")
    return batch


def model_settings(protocol, name, seed):
    return {**protocol, "seed": seed, "model": name,
            "parameter_count": protocol["parameter_counts"][name], "coordinate_scale": 32}


def fit(root, seed):
    batch = load_batch(root)
    if seed not in batch["seeds"]:
        raise ValueError("Fit seed is not part of the frozen batch")
    directory = seed_root(batch["output_dir"], seed)
    if (directory / "fit_complete.json").exists():
        return validate_seed(batch, seed)
    if any((directory / f"{name}_seed{seed}").exists() for name in batch["models"]):
        raise ValueError("Partial fit artifacts exist; refusing to silently overwrite training")
    protocol = individual_protocol(batch, seed)
    args = argparse.Namespace(**protocol)
    args.output_dir = directory
    torch.set_num_threads(args.threads)
    _, training, validation, _create_test = benchmark.prepare_task(args, seed)
    for name in args.models:
        torch.manual_seed(seed)
        model = benchmark.build_model(name, 32, args.width, args.transmil_landmarks)
        settings = model_settings(protocol, name, seed)
        benchmark.train(model, training, validation, seed=seed, steps=args.steps,
                        batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                        output_dir=directory / f"{name}_seed{seed}", settings=settings)
    artifacts = {}
    for name in args.models:
        artifacts[name] = {filename: file_hash(directory / f"{name}_seed{seed}" / filename)
                           for filename in ("best.pt", "final.pt", "training.json")}
    write_json(directory / "fit_complete.json", {"batch_id": batch["batch_id"], "seed": seed,
               "steps": args.steps, "models": args.models, "artifact_sha256": artifacts}, exclusive=True)
    return validate_seed(batch, seed)


def required_fit_files(batch, seed):
    directory = seed_root(batch["output_dir"], seed)
    return [directory / "fit_complete.json"] + [
        directory / f"{name}_seed{seed}" / filename for name in batch["models"]
        for filename in ("best.pt", "final.pt", "training.json")]


def validate_seed(batch, seed):
    directory = seed_root(batch["output_dir"], seed)
    marker = json.loads((directory / "fit_complete.json").read_text())
    expected = {"batch_id": batch["batch_id"], "seed": seed,
                "steps": batch["steps"], "models": batch["models"]}
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid fit completion marker: seed{seed}")
    protocol = individual_protocol(batch, seed)
    runs = []
    for name in batch["models"]:
        run_dir = directory / f"{name}_seed{seed}"
        for filename in ("best.pt", "final.pt", "training.json"):
            if file_hash(run_dir / filename) != marker["artifact_sha256"][name][filename]:
                raise ValueError(f"Changed fitted artifact: seed{seed}/{name}/{filename}")
        report = json.loads((run_dir / "training.json").read_text())
        settings = model_settings(protocol, name, seed)
        if report["settings"] != settings or report["parameters"] != settings["parameter_count"]:
            raise ValueError(f"Training report conflicts with frozen protocol: seed{seed}/{name}")
        expected_steps = [0, *range(batch["eval_every"], batch["steps"] + 1, batch["eval_every"])]
        if [entry["step"] for entry in report["history"]] != expected_steps:
            raise ValueError(f"Incomplete validation history: seed{seed}/{name}")
        ranks = [benchmark.checkpoint_rank(entry["validation"]) for entry in report["history"]]
        selected_index = ranks.index(max(ranks))
        if (report["selected_step"] != expected_steps[selected_index]
                or tuple(report["selected_rank"]) != ranks[selected_index]):
            raise ValueError(f"Invalid validation selection: seed{seed}/{name}")
        for filename, expected_step in (("final.pt", batch["steps"]),
                                        ("best.pt", report["selected_step"])):
            state = torch.load(run_dir / filename, map_location="cpu", weights_only=True)
            required = ("model_state", "optimizer_state", "sampler_state", "torch_rng_state")
            if any(key not in state for key in required) or state["step"] != expected_step:
                raise ValueError(f"Incomplete training checkpoint: seed{seed}/{name}/{filename}")
            if state["settings"] != settings:
                raise ValueError(f"Checkpoint conflicts with frozen protocol: seed{seed}/{name}")
            if filename == "final.pt":
                optimizer_states = state["optimizer_state"].get("state", {})
                if not optimizer_states or any(int(value["step"]) != batch["steps"]
                                               for value in optimizer_states.values()):
                    raise ValueError(f"Optimizer did not complete all updates: seed{seed}/{name}")
        runs.append((seed, name, run_dir, report))
    return runs


def preflight(root):
    """Gate for both primary and separately implemented challenge evaluation."""
    batch = load_batch(root)
    missing = [str(path) for seed in batch["seeds"] for path in required_fit_files(batch, seed)
               if not path.is_file()]
    if missing:
        raise ValueError("All nine fits must finish before any final test is generated; missing "
                         + ", ".join(missing))
    runs = [run for seed in batch["seeds"] for run in validate_seed(batch, seed)]
    return batch, runs


def evaluate_batch(root):
    batch, runs = preflight(root)  # Must precede even prepare_task, including all test closures.
    if (Path(root) / "evaluation_complete.json").exists():
        raise ValueError("Final evaluation already exists; refusing silent replacement")
    torch.set_num_threads(batch["threads"])
    all_results = []
    from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import audit_splits, dataset_checks
    for seed in batch["seeds"]:
        directory = seed_root(batch["output_dir"], seed)
        protocol = individual_protocol(batch, seed)
        args = argparse.Namespace(**protocol)
        args.output_dir = directory
        _, training, validation, create_test = benchmark.prepare_task(args, seed)
        raw_test = create_test()
        checks = {"test": dataset_checks(raw_test), "overlap": audit_splits(
            {"train": training, "validation": validation, "test": raw_test},
            require_disjoint_appearances=True)}
        write_json(directory / f"data_checks_seed{seed}.json", checks)
        write_json(directory / f"test_manifest_seed{seed}.json", {
            "split": "test", "samples": len(raw_test["y"]), "batch_id": batch["batch_id"],
            "image_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in raw_test["images"]]})
        test = benchmark.tensor_data(raw_test)
        test["intended_cases"] = benchmark.planned_cases(args)
        results = []
        for run_seed, name, run_dir, report in runs:
            if run_seed != seed:
                continue
            model = benchmark.build_model(name, 32, args.width, args.transmil_landmarks)
            state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state"])
            started = time.perf_counter()
            metrics, predictions = benchmark.evaluate(model, test, args.batch_size, return_predictions=True)
            inference_seconds = time.perf_counter() - started
            state = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state"])
            final_metrics, final_predictions = benchmark.evaluate(
                model, test, args.batch_size, return_predictions=True)
            result = {"model": name, "seed": seed, "batch_id": batch["batch_id"],
                      "selected_step": report["selected_step"], "parameters": report["parameters"],
                      "test": metrics, "test_final_update": final_metrics,
                      "seconds": report["seconds"], "tokens": int(test["x"].shape[1]),
                      "test_inference_seconds": inference_seconds,
                      "test_images_per_second": len(test["y"]) / inference_seconds,
                      "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                      * (1 if sys.platform == "darwin" else 1024),
                      "memory_scope": "whole evaluation-process high-water RSS; includes data and earlier models, not isolated model peak",
                      "test_sample_uncertainty": benchmark.grouped_test_uncertainty(test, predictions, seed)}
            write_json(run_dir / "test.json", result)
            np.savez_compressed(run_dir / "test_predictions.npz", targets=test["y"].numpy(),
                                predictions=predictions, final_predictions=final_predictions,
                                cases=np.asarray(test["cases"]), pair_ids=np.asarray(test["pair_ids"]))
            results.append(result)
            print(f"FINAL {name}/seed{seed} test={metrics['accuracy']:.3f} "
                  f"worst={metrics['worst_case_accuracy']:.3f}", flush=True)
        benchmark.write_results(directory, results)
        all_results.extend(results)
    benchmark.write_results(Path(batch["output_dir"]), all_results)
    write_json(Path(batch["output_dir"]) / "evaluation_complete.json", {
        "batch_id": batch["batch_id"], "models_evaluated": len(all_results),
        "policy": batch["test_policy"]}, exclusive=True)
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("freeze", "fit", "evaluate"))
    parser.add_argument("--root", type=Path, default=REPO / "runs/mondriaan_oct11_attention")
    parser.add_argument("--seed", type=int, choices=SETTINGS["seeds"])
    args = parser.parse_args()
    if (args.stage == "fit") != (args.seed is not None):
        parser.error("--seed is required for fit and accepted only for fit")
    if args.stage == "freeze":
        result = freeze(args.root)
        print(f"FROZEN {result['batch_id']} at {args.root}", flush=True)
    elif args.stage == "fit":
        fit(args.root, args.seed)
    else:
        evaluate_batch(args.root)


if __name__ == "__main__":
    main()
