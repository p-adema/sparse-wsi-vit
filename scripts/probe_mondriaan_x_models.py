"""Run ViT-5, ABMIL, MeanMLP, or TransMIL on Mondriaan-X full-grid tokens."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan_x import make_data_x, check_data_x
from scripts.probe_mondriaan_controls import train_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        choices=("vit5", "abmil", "meanmlp", "transmil"),
                        default=["abmil", "meanmlp"])
    parser.add_argument("--num-tiles", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mondriaan_x"))
    args = parser.parse_args()

    for key in ("train_size", "eval_size", "batch_size", "eval_batch_size"):
        n = getattr(args, key)
        if n < 1 or (key != "eval_batch_size" and (n < 8 or n % 8)):
            parser.error(f"{key} must be positive" +
                         (" and divisible by 8" if key != "eval_batch_size" else ""))
    if args.num_tiles < 1 or int(args.num_tiles ** 0.5) ** 2 != args.num_tiles:
        parser.error("num-tiles must be a positive perfect square")
    if min(args.steps, args.eval_every, args.threads) < 1:
        parser.error("steps, eval-every, and threads must be positive")
    if args.device == "cuda" and not args.smoke_test and not torch.cuda.is_available():
        parser.error("CUDA requested but no GPU is available")

    torch.set_num_threads(args.threads)
    train = make_data_x(args.train_size, args.seed, args.num_tiles)
    test = make_data_x(args.eval_size, args.seed + 1, args.num_tiles)
    checks = {"train": check_data_x(train, args.num_tiles),
              "test": check_data_x(test, args.num_tiles)}
    print(json.dumps(checks, indent=2), flush=True)
    if args.smoke_test:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = train["x"].shape[1]
    for name in args.models:
        print(f"Running {name} on {tokens} tokens ({args.device})", flush=True)
        train_one(name, f"x_tokens{tokens}", args.seed, train,
                  {"standard": test}, args)


if __name__ == "__main__":
    main()
