"""Finish interrupted independent restart jobs from their best serialized branch state.

The best checkpoint also contains the optimizer and exact stochastic streams.
Earlier updates are replayed when the best checkpoint precedes the interruption;
the selected checkpoint rank is retained. This changes no frozen protocol.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_benchmark import (
    build_model,
    checkpoint_rank,
    evaluate,
    grouped_test_uncertainty,
    planned_cases,
    prepare_task,
    tensor_data,
    train,
    write_results,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import audit_splits, dataset_checks


def finish(root):
    if (root / "results.json").exists():
        raise ValueError("Job already finished; preserve its results")
    args = argparse.Namespace(**json.loads((root / "protocol.json").read_text()))
    args.output_dir = root
    if args.task != "restart" or args.models != ["vit5"] or len(args.seeds) != 1:
        raise ValueError("Recovery requires an independent local ViT restart job")
    torch.set_num_threads(args.threads)
    seed = args.seeds[0]
    _, training, validation, create_test = prepare_task(args, seed)
    base = root / f"vit5_seed{seed}"
    for arm in ("uninterrupted", "restart"):
        directory = base / arm
        if (directory / "final.pt").exists():
            continue
        source = directory / "best.pt"
        if not source.exists():
            raise ValueError("No branch checkpoint available")
        state = torch.load(source, weights_only=True)
        expected = state["step"] - (args.restart_at if arm == "restart" else 0)
        counters = {int(item["step"].item()) for item in state["optimizer_state"]["state"].values()}
        if counters != {expected} or not args.restart_at <= state["step"] < args.steps:
            raise ValueError("Saved branch state does not match the frozen update budget")
        archive = directory / "before_recovery"
        archive.mkdir(exist_ok=False)
        shutil.copyfile(source, archive / "best.pt")
        model = build_model("vit5", args.coordinate_scale or 32, args.width)
        model.load_state_dict(state["model_state"])
        selected = evaluate(model, validation, args.batch_size)
        report = train(model, training, validation, seed=seed, steps=args.steps,
                       batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
                       output_dir=directory, settings=state["settings"], initial_state=state,
                       inherited_selection={"rank": checkpoint_rank(selected), "step": state["step"],
                                            "checkpoint": archive / "best.pt"})
        report.update(recovery={"source_checkpoint_step": state["step"],
                                "optimizer_and_rng_preserved": True,
                                "prior_selected_checkpoint_retained": True},
                      seconds_scope="recovered segment only; earlier branch runtime unavailable")
        (directory / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    # Final test is first constructed after both frozen arms have completed.
    raw_test = create_test()
    checks = {"test": dataset_checks(raw_test),
              "overlap": audit_splits({"train": training, "validation": validation, "test": raw_test},
                                       require_disjoint_appearances=args.appearance_holdout)}
    (root / f"data_checks_seed{seed}.json").write_text(json.dumps(checks, indent=2) + "\n")
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in raw_test["images"]]
    (root / f"test_manifest_seed{seed}.json").write_text(json.dumps(
        {"split": "test", "samples": len(hashes), "image_sha256": hashes}, indent=2) + "\n")
    test = tensor_data(raw_test)
    test["intended_cases"] = planned_cases(args)
    results = []
    for arm in ("uninterrupted", "restart"):
        directory = base / arm
        report = json.loads((directory / "training.json").read_text())
        model = build_model("vit5", args.coordinate_scale or 32, args.width)
        model.load_state_dict(torch.load(directory / "best.pt", weights_only=True)["model_state"])
        metrics, predicted = evaluate(model, test, args.batch_size, return_predictions=True)
        model.load_state_dict(torch.load(directory / "final.pt", weights_only=True)["model_state"])
        result = {"model": "vit5", "seed": seed, "arm": arm,
                  "selected_step": report["selected_step"], "parameters": report["parameters"],
                  "test": metrics, "test_final_update": evaluate(model, test, args.batch_size),
                  "seconds": report["seconds"], "tokens": test["x"].shape[1],
                  "seconds_scope": report.get("seconds_scope", "branch suffix only"),
                  "test_sample_uncertainty": grouped_test_uncertainty(test, predicted, seed)}
        (directory / "test.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(f"FINAL seed{seed}/{arm}: test={metrics['accuracy']:.3f} "
              f"worst={metrics['worst_case_accuracy']:.3f}", flush=True)
    write_results(root, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    finish(parser.parse_args().run_dir)
