"""Continue pair-only and full-task checkpoints on the full strict-two task.

Both arms use the same second-stage images, minibatches, seeds, architecture,
and restarted AdamW optimizer. They differ in the first 2,000 training updates:
balanced one-pair/all-distinct versus the original five-case full task.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_controls import train_one
from scripts.probe_mondriaan_matching_controls import prepare_data


PARENTS = {
    "pair_to_full": Path("runs/mondriaan_matching_controls/pair_only"),
    "full_to_full": Path("runs/mondriaan_patch_training/more_data/all24_aligned"),
}


def validate_parent(path, model, seed, train_size, eval_size, arm, expected_steps=2000):
    """Reject a mismatched first-stage experiment before loading its weights."""
    report = json.loads(path.with_suffix(".json").read_text())
    settings = report["settings"]
    expected = {"model": model, "seed": seed, "train_size": train_size,
                "eval_size": eval_size // 4 if arm == "pair_to_full" else eval_size,
                "random_colors": False, "vocabulary_size": 24,
                "object_placement": "aligned", "steps": expected_steps}
    expected.update({"control": "pair_only"} if arm == "pair_to_full" else
                    {"condition": "all24_aligned"})
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Parent {path}: {key}={settings.get(key)!r}, expected {value!r}")
    if report["tokens_per_sample"] != 64 or report["history"][-1]["step"] != expected_steps:
        raise ValueError("Parent must be a completed 64-token first-stage experiment")
    if not path.is_file():
        raise FileNotFoundError(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=("vit5", "abmil"),
                        default=["vit5", "abmil"])
    parser.add_argument("--arms", nargs="+", choices=tuple(PARENTS), default=list(PARENTS))
    parser.add_argument("--parent-root", type=Path, default=Path("."))
    parser.add_argument("--parent-steps", type=int, default=2000,
                        help="Expected first-stage updates; override for smoke checks only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("train_size", "eval_size", "batch_size"):
        if getattr(args, key) < 8 or getattr(args, key) % 8:
            parser.error(f"{key} must be a positive multiple of eight")
    if min(args.steps, args.eval_every, args.threads, args.parent_steps) < 1:
        parser.error("steps, eval-every, threads, and parent-steps must be positive")
    parents = {}
    for arm in args.arms:
        for model in args.models:
            path = args.parent_root / PARENTS[arm] / f"{model}_grid_seed{args.seed}.pt"
            parent = validate_parent(path, model, args.seed, args.train_size,
                                     args.eval_size, arm, args.parent_steps)
            parents[arm, model] = (path, parent)
    torch.set_num_threads(args.threads)
    training, full_eval, pair_eval, checks = prepare_data(args.train_size, args.eval_size,
                                                        args.seed)
    train = training["upweighted"]  # The exact original full-task pool, without upweighting.
    root = args.output_dir
    for arm in args.arms:
        args.output_dir = root / arm
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check_path = args.output_dir / f"data_checks_seed{args.seed}.json"
        check_path.write_text(json.dumps(checks, indent=2) + "\n")
        for model in args.models:
            path, parent = parents[arm, model]
            known = set(train["type_multisets"])
            views = {name: {**data, "unseen_combinations": torch.tensor([
                types not in known for types in data["type_multisets"]])}
                for name, data in {"standard": full_eval, "pair_balanced": pair_eval}.items()}
            report = train_one(
                model, "grid", args.seed, train, views, args,
                experiment_settings={
                    "curriculum_arm": arm, "stage": 2,
                    "parent_steps": parent["settings"]["steps"],
                    "total_steps": parent["settings"]["steps"] + args.steps,
                    "vocabulary_size": 24, "object_placement": "aligned",
                    "rope_high": 32, "init_scale": 1e-4,
                    "data_checks_file": str(check_path),
                    "minibatch_case_counts": {"one_pair": args.batch_size // 2,
                                              "each_negative": args.batch_size // 8},
                    "first_stage_all_distinct_label": False,
                    "first_stage_one_pair_label": True,
                }, save_checkpoint=True, initial_checkpoint=path,
                sampler_seed=args.seed + 4000,
            )
            expected_initial = (parent["history"][-1]["robustness"]["full_task"]
                                if arm == "pair_to_full" else
                                parent["history"][-1]["evaluation"])
            actual = report["history"][0]["evaluation"]
            if (actual["accuracy"] != expected_initial["accuracy"] or
                    actual["by_case"] != expected_initial["by_case"]):
                raise RuntimeError("Initial full-task scores do not reproduce the parent checkpoint")


if __name__ == "__main__":
    main()
