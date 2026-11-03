"""Visualise the effect of spatial pooling on Mondriaan images.

Produces three PDFs — one per pool level. Each shows the full image with
a zoom inset (connected by white lines) of one key object.

    mondriaan_ps1.pdf  — pixel resolution
    mondriaan_ps2.pdf  — 2×2 average pool
    mondriaan_ps4.pdf  — 4×4 average pool

Usage:
    uv run visualize_pooling.py [--seed N] [--label {0,1}] [--image_size N] [--random_colors]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent / "src"))
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator

FNAMES = {1: "mondriaan_ps1.pdf", 2: "mondriaan_ps2.pdf", 4: "mondriaan_ps4.pdf"}


def _titles(image_size: int) -> dict:
    return {
        ps: f"{'Pixel resolution' if ps == 1 else f'{ps}×{ps} average pool'}\n"
            f"({(image_size // ps) ** 2:,} tokens)"
        for ps in (1, 2, 4)
    }


def _pool(img: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size == 1:
        return img
    H, W, C = img.shape
    H2, W2 = H // pool_size, W // pool_size
    return img[:H2 * pool_size, :W2 * pool_size].reshape(
        H2, pool_size, W2, pool_size, C
    ).mean(axis=(1, 3))


def _zoom(img: np.ndarray, target_px: int) -> np.ndarray:
    H, W, _ = img.shape
    factor = target_px // H
    return np.kron(img, np.ones((factor, factor, 1), dtype=np.float32))


def _add_boxes(ax, positions, pool_size, display_px, image_size=64):
    cell = display_px / (image_size // pool_size)
    obj_tokens = 4 // pool_size
    for (y, x) in positions:
        rect = mpatches.Rectangle(
            ((x // pool_size) * cell - 0.5, (y // pool_size) * cell - 0.5),
            obj_tokens * cell, obj_tokens * cell,
            linewidth=1.5, edgecolor="yellow", facecolor="none",
        )
        ax.add_patch(rect)


def _inset_pos(y, x, image_size=64):
    """Pick the inset corner farthest from the object."""
    if y < image_size / 2 and x < image_size / 2:
        return [0.57, 0.0, 0.41, 0.41]   # object top-left  → inset bottom-right
    if y < image_size / 2 and x >= image_size / 2:
        return [0.0, 0.0, 0.41, 0.41]    # object top-right → inset bottom-left
    if y >= image_size / 2 and x < image_size / 2:
        return [0.57, 0.57, 0.41, 0.41]  # object bot-left  → inset top-right
    return [0.0, 0.57, 0.41, 0.41]       # object bot-right → inset top-left


def save_panel(img, positions, pool_size, display_px=512, image_size=32):
    pooled = _pool(img, pool_size)
    full_zoomed = _zoom(pooled, display_px)

    cell = display_px / (image_size // pool_size)
    obj_tokens = 4 // pool_size

    # zoom in on the first object
    y, x = positions[0]
    py = (y // pool_size) * cell
    px_ = (x // pool_size) * cell
    obj_px = obj_tokens * cell

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(full_zoomed, interpolation="nearest")
    ax.axis("off")
    _add_boxes(ax, positions, pool_size, display_px, image_size)

    # inset axes in the far corner
    axins = ax.inset_axes(_inset_pos(y, x, image_size))
    axins.imshow(full_zoomed, interpolation="nearest")
    axins.axis("off")

    # set zoom window with fixed padding (16 display px regardless of pool size)
    pad = 16
    x0, x1 = px_ - pad, px_ + obj_px + pad
    y0, y1 = py - pad, py + obj_px + pad  # image coords (y down)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y1, y0)  # y-axis inverted for imshow

    # white zoom rectangle on main image
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="white", facecolor="none", zorder=5,
    ))

    # connecting lines and inset border drawn in data coordinates.
    # the inset at axes-fraction [ix, iy, iw, ih] maps to data coords as:
    #   x_data = x_frac * display_px
    #   y_data = (1 - y_frac) * display_px   (imshow flips y)
    ipos = _inset_pos(y, x)
    ix, iy, iw, ih = ipos

    # white inset border
    inset_x = ix * display_px
    inset_y = (1.0 - (iy + ih)) * display_px   # top edge (y is flipped)
    ax.add_patch(mpatches.Rectangle(
        (inset_x, inset_y), iw * display_px, ih * display_px,
        linewidth=1.5, edgecolor="white", facecolor="none", zorder=7, clip_on=False,
    ))

    def frac_to_data(xf, yf):
        return xf * display_px, (1.0 - yf) * display_px

    if ix > 0.5:  # inset on the right → connect from right edge of zoom box
        za, zb = (x1, y0), (x1, y1)
        ia = frac_to_data(ix, iy + ih)  # inset top-left  (high y_frac → low y_data)
        ib = frac_to_data(ix, iy)       # inset bottom-left
    else:         # inset on the left → connect from left edge
        za, zb = (x0, y0), (x0, y1)
        ia = frac_to_data(ix + iw, iy + ih)
        ib = frac_to_data(ix + iw, iy)

    for (start, end) in [(za, ia), (zb, ib)]:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color="white", linewidth=1.0, zorder=6, clip_on=False)

    fig.tight_layout(pad=0.2)

    fname = FNAMES[pool_size]
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int, default=7)
    parser.add_argument("--label",        type=int, choices=[0, 1], default=1)
    parser.add_argument("--image_size",   type=int, default=32)
    parser.add_argument("--random_colors", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img, _, _, positions = MondriaanGenerator.generate_single(
        image_size=args.image_size,
        object_type="patterned",
        target_label=args.label,
        min_sep=16,
        random_colors=args.random_colors,
    )

    for ps in [1, 2, 4]:
        save_panel(img, positions, ps, image_size=args.image_size)


if __name__ == "__main__":
    main()
