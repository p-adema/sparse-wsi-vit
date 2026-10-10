"""Draw actual CPU-probe samples, token selection, and paired robustness views."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mondriaan-mpl-cache"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_mondriaan_controls import foreground_view, make_variant, novel_vocabulary
from scripts.probe_mondriaan_vit5 import make_data
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import PATTERNED_VOCAB


CASE_NAMES = {
    "all_distinct": "4 distinct",
    "one_pair": "1 pair + 2 distinct",
    "two_pairs": "2 pairs",
    "three_matching": "3 matching + 1 distinct",
    "four_matching": "4 matching",
}
GOLD = "#f6b632"
INK = "#172b40"


def scene(ax, image, *, removed=False, window=None):
    shown = image.copy()
    if removed:
        shown[(shown == 0).all(axis=-1)] = [0.88, 0.89, 0.91]
    ax.imshow(shown, interpolation="nearest", extent=(0, 32, 32, 0))
    for line in range(0, 33, 4):
        ax.axvline(line, color="#8d939a", lw=0.45, alpha=0.65)
        ax.axhline(line, color="#8d939a", lw=0.45, alpha=0.65)
    if removed:
        for y in range(0, 32, 4):
            for x in range(0, 32, 4):
                if np.any(image[y:y + 4, x:x + 4]):
                    ax.add_patch(Rectangle((x, y), 4, 4, fill=False, ec=GOLD, lw=1.4))
    if window:
        x, y = window
        ax.add_patch(Rectangle((x, y), 12, 12, fill=False, ec=GOLD, lw=1.5))
    ax.set_xlim(0, 32)
    ax.set_ylim(32, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#81909c")


def zoom(ax, image, window):
    x, y = window
    ax.imshow(image[y:y + 12, x:x + 12], interpolation="nearest", extent=(0, 12, 12, 0))
    for line in range(0, 13, 4):
        ax.axvline(line, color="#b7bdc4", lw=1)
        ax.axhline(line, color="#b7bdc4", lw=1)
    ax.set_xlim(0, 12)
    ax.set_ylim(12, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GOLD)
        spine.set_linewidth(1.3)


def object_tokens(data, index):
    tokens = foreground_view(data)["x"][index].numpy().reshape(4, 4, 4, 3)
    return tokens


def save(fig, output_dir, name):
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=190, facecolor="white")
    plt.close(fig)


def tokenization_figure(data, index, output_dir):
    fig = plt.figure(figsize=(13, 5.4))
    grid = fig.add_gridspec(1, 3, left=0.045, right=0.965, bottom=0.22, top=0.79,
                           wspace=0.22, width_ratios=[1, 1, 1.12])
    full = fig.add_subplot(grid[0])
    selected = fig.add_subplot(grid[1])
    image = data["images"][index]
    scene(full, image)
    scene(selected, image, removed=True)
    full.set_title("All 64 patches retained", pad=14, color=INK, weight="bold")
    selected.set_title("Only 4 foreground patches retained", pad=14, color=INK, weight="bold")
    full.set_xlabel("4 object patches + 60 background patches", labelpad=12)
    selected.set_xlabel("Gray patches omitted; gold patches kept", labelpad=12)
    tokens = object_tokens(data, index)
    subgrid = grid[2].subgridspec(2, 2, wspace=0.18, hspace=0.4)
    for i, token in enumerate(tokens):
        ax = fig.add_subplot(subgrid[i // 2, i % 2])
        ax.imshow(token, interpolation="nearest")
        ax.set_title(f"Token {i + 1}", fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.81, 0.84, "The four object tokens, enlarged", ha="center", color=INK,
             weight="bold", fontsize=12)
    fig.suptitle("Same Mondriaan sample; two input selections", fontsize=19, color=INK, y=0.97)
    fig.text(0.5, 0.89, "Truth = True  ·  One matching pair and two distinct objects",
             ha="center", fontsize=12, color=INK)
    fig.text(0.5, 0.09,
             "Each 4×4 patch becomes 48 ordered RGB values; its pixels are flattened, preserving the pattern.\n"
             "Both selections were tested with ViT5 Dense, ABMIL, and an MLP → mean → MLP baseline.",
             ha="center", va="center", fontsize=11, linespacing=1.7, color=INK)
    save(fig, output_dir, "sample_tokenization")


def vocabulary_panel(ax):
    # One row of training patterns, a blank separator, three rows of new patterns.
    canvas = np.ones((23, 29, 3), dtype=np.float32)
    patterns = list(np.array(PATTERNED_VOCAB)) + novel_vocabulary()
    for i, pattern in enumerate(patterns):
        row = 0 if i < 6 else 1 + (i - 6) // 6
        column = i if i < 6 else (i - 6) % 6
        y = 0 if row == 0 else 8 + (row - 1) * 5
        patch = pattern.reshape(2, 2, 3).repeat(2, axis=0).repeat(2, axis=1)
        canvas[y:y + 4, column * 5:column * 5 + 4] = patch
    ax.imshow(canvas, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 5.6, "6 training patterns above", color=INK, fontsize=9, va="center")
    ax.text(0, 22.2, "18 unseen patterns above", color=INK, fontsize=9, va="center")
    ax.set_title("Same four colors; different arrangements", fontsize=11, pad=14,
                 weight="bold", color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def accuracy_range(output_dir, key):
    values = []
    for seed in (42, 7):
        path = output_dir / f"vit5_grid_seed{seed}.json"
        report = json.loads(path.read_text())
        final = report["history"][-1]
        metrics = final["evaluation"] if key == "standard" else final["robustness"][key]
        values.append(100 * metrics["accuracy"])
    return f"{min(values):.1f}%" if min(values) == max(values) else f"{min(values):.1f}–{max(values):.1f}%"


def robustness_figure(standard, novel, interior, shifted1, shifted2, output_dir):
    i = standard["indices"]["one_pair"][0].item()
    j = interior["indices"]["one_pair"][0].item()
    fig = plt.figure(figsize=(13, 10.8))
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.95, bottom=0.17, top=0.83,
                           wspace=0.22, hspace=0.47, height_ratios=[1, 1.4])
    for column, (data, title, key) in enumerate([
        (standard, "Familiar patterns", "standard"),
        (novel, "Unseen permutations", "novel_permutations"),
    ]):
        ax = fig.add_subplot(grid[0, column])
        scene(ax, data["images"][i])
        ax.set_title(title, fontsize=12, pad=12, weight="bold", color=INK)
        ax.set_xlabel("Set accuracy: " + accuracy_range(output_dir, key), labelpad=10,
                      fontsize=11, color=INK)
    vocabulary_panel(fig.add_subplot(grid[0, 2]))
    # Prefer a window containing just one object in all three views, so neighboring
    # objects entering the crop do not obscure the fragmentation being illustrated.
    windows = [(max(0, min(x - 4, 20)), max(0, min(y - 4, 20)))
               for y, x in interior["positions"][j]]

    def extra_pixels(window):
        x, y = window
        return sum(np.any(data["images"][j][y:y + 12, x:x + 12], axis=-1).sum() - 16
                   for data in [interior, shifted1, shifted2])

    window = min(windows, key=extra_pixels)
    for column, (data, title, key) in enumerate([
        (interior, "Aligned shift control", "shift_control"),
        (shifted1, "Objects moved +1 px in x and y", "shift_1px"),
        (shifted2, "Objects moved +2 px in x and y", "shift_2px"),
    ]):
        subgrid = grid[1, column].subgridspec(2, 1, height_ratios=[1, 0.56], hspace=0.17)
        ax = fig.add_subplot(subgrid[0])
        scene(ax, data["images"][j], window=window)
        ax.set_title(title, fontsize=11, pad=12, weight="bold", color=INK)
        inset = fig.add_subplot(subgrid[1])
        zoom(inset, data["images"][j], window)
        inset.set_xlabel("Set accuracy: " + accuracy_range(output_dir, key), labelpad=10,
                         fontsize=11, color=INK)
    fig.suptitle("What the robustness tests changed", fontsize=20, color=INK, y=0.975)
    fig.text(0.5, 0.93, "Every sample shown has Truth = True: one pair + two distinct objects.",
             ha="center", fontsize=12, color=INK)
    fig.text(0.055, 0.88, "A. Appearance test: same positions and repetition counts",
             fontsize=12, color=INK, weight="bold")
    fig.text(0.055, 0.52, "B. Alignment test: a separate paired sample; no clipping or lost pixels",
             fontsize=12, color=INK, weight="bold")
    fig.text(0.5, 0.051,
             "Thin lines mark fixed 4×4 patch boundaries. Gold windows are enlarged below.\n"
             "All-grid inputs stay at 64 tokens. Foreground inputs change from 4 whole objects to 16 fragments.\n"
             "Accuracy captions summarize full-grid ViT5 evaluation across two seeds, on all samples in each set.",
             ha="center", fontsize=10.5, color=INK, linespacing=1.7)
    save(fig, output_dir, "sample_robustness")
    return i, j


def truth_table_figure(data, output_dir):
    fig = plt.figure(figsize=(10.5, 14.4))
    grid = fig.add_gridspec(5, 2, left=0.075, right=0.955, bottom=0.08, top=0.872,
                           width_ratios=[1, 1.8], wspace=0.22, hspace=0.4)
    indices = {}
    for row, (case, name) in enumerate(CASE_NAMES.items()):
        index = data["indices"][case][0].item()
        indices[case] = index
        ax = fig.add_subplot(grid[row, 0])
        scene(ax, data["images"][index])
        truth = bool(data["y"][index].item())
        ax.set_title(f"{name}\nTruth = {truth}", fontsize=12, color=INK, pad=8, weight="bold")
        strip = grid[row, 1].subgridspec(1, 4, wspace=0.18)
        for column, token in enumerate(object_tokens(data, index)):
            token_ax = fig.add_subplot(strip[0, column])
            token_ax.imshow(token, interpolation="nearest")
            token_ax.set_title(f"Token {column + 1}", fontsize=10, pad=9)
            token_ax.set_xticks([])
            token_ax.set_yticks([])
        # Recover the truth from the actual crops, independently of the stored type IDs.
        counts = Counter(token.tobytes() for token in object_tokens(data, index))
        assert truth == (max(counts.values()) == 2 and list(counts.values()).count(2) == 1)
    fig.suptitle("All five tested configurations", fontsize=20, color=INK, y=0.974)
    fig.text(0.5, 0.941, "Actual familiar-pattern evaluation images and their four object tokens",
             ha="center", fontsize=12, color=INK)
    fig.text(0.5, 0.034,
             "Only one pair + two distinct objects is positive. Tokens are displayed in model input order.\n"
             "Grid lines and captions are visual annotations; the models receive ordered RGB values.",
             ha="center", fontsize=11, color=INK, linespacing=1.7)
    save(fig, output_dir, "sample_truth_table")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mondriaan_object_controls"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    # Exactly the seed-42 evaluation datasets and transformations used by the CPU runner.
    standard = make_data(1024, 43, 1, False, tokenization="object", retain_images=True)
    novel = make_variant(standard, novel=True, seed=142)
    interior = make_data(1024, 44, 1, False, tokenization="object", retain_images=True,
                         interior_only=True)
    shifted1 = make_variant(interior, shift=1)
    shifted2 = make_variant(interior, shift=2)
    index = standard["indices"]["one_pair"][0].item()
    tokenization_figure(standard, index, args.output_dir)
    standard_index, shift_index = robustness_figure(
        standard, novel, interior, shifted1, shifted2, args.output_dir)
    indices = truth_table_figure(standard, args.output_dir)
    metadata = {
        "run_seed": 42, "standard_generation_seed": 43, "interior_generation_seed": 44,
        "novel_mapping_seed": 142, "evaluation_size": 1024,
        "standard_positive_index": standard_index, "shift_positive_index": shift_index,
        "truth_table_indices": indices,
        "sample_sha256": {
            name: hashlib.sha256(data["images"][i].tobytes()).hexdigest()
            for name, data, i in [("standard", standard, standard_index),
                                  ("novel", novel, standard_index),
                                  ("shift_control", interior, shift_index),
                                  ("shift_1px", shifted1, shift_index),
                                  ("shift_2px", shifted2, shift_index)]
        },
    }
    (args.output_dir / "sample_visualization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved sample_tokenization, sample_robustness, and sample_truth_table as PNG and SVG.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
