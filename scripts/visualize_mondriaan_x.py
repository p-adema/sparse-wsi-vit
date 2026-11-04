"""Visualize Mondriaan-X cases across token scales.

Outputs:
  mondriaan_x_examples.png: positive AABC sample across four scales
  mondriaan_x_cases.png: all five cases across four scales

Run from the repository root:
    python scripts/visualize_mondriaan_x.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan_x import make_data_x

SCALES = (1, 4, 16, 64)
CASES = (
    ("one_pair", "AABC · positive"),
    ("all_distinct", "ABCD · negative"),
    ("two_pairs", "AABB · negative"),
    ("three_matching", "AAAB · negative"),
    ("four_matching", "AAAA · negative"),
)


def reconstruct(tokens):
    """Convert ordered 4×4 RGB patch tokens back into an RGB image."""
    tokens = tokens.detach().cpu().numpy()
    n = len(tokens)
    side = int(np.sqrt(n))
    if side * side != n or tokens.shape[1] != 48:
        raise ValueError(f"Expected a square grid of 48-D tokens; got {tokens.shape}")
    patches = tokens.reshape(side, side, 4, 4, 3)
    image = patches.transpose(0, 2, 1, 3, 4).reshape(side * 4, side * 4, 3)
    return np.clip(image, 0, 1)


def case_index(data, case):
    """Use dataset-provided case membership, never a hard-coded sample index."""
    if "indices" in data and case in data["indices"]:
        indices = data["indices"][case]
        if len(indices):
            return int(indices[0])
    # Fallback for datasets exposing per-sample case labels instead.
    for key in ("cases", "case_names", "case_types"):
        if key in data:
            for i, label in enumerate(data[key]):
                if label == case:
                    return i
    raise KeyError(
        f"Cannot locate case {case!r}. Available dataset keys: {list(data)}. "
        "Inspect make_data_x() to find its case-index metadata."
    )


def plot_grid(datasets, cases, output, dpi):
    fig, axes = plt.subplots(len(cases), len(SCALES),
                             figsize=(14, 3.35 * len(cases)),
                             squeeze=False)
    for row, (case, label) in enumerate(cases):
        for col, tiles in enumerate(SCALES):
            data = datasets[tiles]
            idx = case_index(data, case)
            ax = axes[row, col]
            ax.imshow(reconstruct(data["x"][idx]), interpolation="nearest")
            if row == 0:
                ax.set_title(f"{tiles} tiles | {data['x'].shape[1]:,} tokens", fontsize=12)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Mondriaan-X: four foreground objects at increasing scales",
                 fontsize=16, y=0.995)
    fig.text(0.5, 0.008,
             "A, B, C, D denote distinct 4×4 color patterns. Only AABC is positive. "
             "Each image has exactly four foreground tokens.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.975), h_pad=1.1, w_pad=0.8)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=32,
                        help="Samples per scale; enough to include every case")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {tiles: make_data_x(args.samples, seed=args.seed, num_tiles=tiles)
                for tiles in SCALES}
    plot_grid(datasets, CASES[:1], args.output_dir / "mondriaan_x_examples.png", args.dpi)
    plot_grid(datasets, CASES, args.output_dir / "mondriaan_x_cases.png", args.dpi)


if __name__ == "__main__":
    main()
