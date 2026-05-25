"""Mondriaan: a synthetic benchmark for testing pairwise global-dependencies
*and* pixel-level pattern reasoning.

Four 4×4-pixel objects placed at random positions in a 32×32 black image.
Label=1 when exactly one matching pair exists among the four objects;
label=0 when all four are distinct.

With random_colors=True (used in experiments): each object independently
samples 4 random RGB colors from Uniform([0,1]³). The matching pair in
label=1 is pixel-identical. Solid-color control (object_type="solid"):
each object is one solid random color; mean color IS discriminative.

Usage (visualization)--See extract_mondriaan.py for (large-scale) data extraction.
-----
    uv run src/sparse_wsi_vit/datasets/mondriaan_dataset/mondriaan.py \\
        --visualize --n_samples 4
"""

import random
from collections import Counter

import numpy as np

# ── fixed base colors (used when random_colors=False) ───────────────────────
# Mean of {RED, CYAN, GREEN, MAGENTA} = (0.5, 0.5, 0.5) under any permutation.
RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
CYAN = np.array([0.0, 1.0, 1.0], dtype=np.float32)
GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)
MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)

_R, _C, _G, _M = RED, CYAN, GREEN, MAGENTA

# ── patterned vocabulary (fixed-color mode) ──────────────────────────────────
# 6 permutations of {R,C,G,M}; pixel order is [TL, TR, BL, BR].
# All pairwise Hamming distances are 3 or 4 (verified exhaustively).
PATTERNED_VOCAB = [
    [_R, _C, _G, _M],  # 0
    [_C, _R, _M, _G],  # 1
    [_G, _M, _R, _C],  # 2
    [_M, _G, _C, _R],  # 3
    [_R, _G, _M, _C],  # 4
    [_C, _M, _G, _R],  # 5
]
PATTERNED_NAMES = [
    "RC/GM",
    "CR/MG",
    "GM/RC",
    "MG/CR",
    "RG/MC",
    "CM/GR",
]

# ── permutation indices shared between fixed and random-color modes ──────────
# These define the 6 arrangement types as index permutations of [c0,c1,c2,c3].
# Hamming distance ≥ 3 between any two rows (property of the permutations,
# independent of the specific colors chosen).
_PATTERNED_PERMS = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
    [0, 2, 3, 1],
    [1, 3, 2, 0],
]

# ── solid-colour control vocabulary (fixed-colour mode) ───────────────────────
# Mean color IS discriminative (label=1 biases the mean; label=0 is grey).
SOLID_VOCAB = [
    [_R, _R, _R, _R],  # 0 – solid red
    [_C, _C, _C, _C],  # 1 – solid cyan
    [_G, _G, _G, _G],  # 2 – solid green
    [_M, _M, _M, _M],  # 3 – solid magenta
]
SOLID_NAMES = ["R", "C", "G", "M"]


# ── generator ─────────────────────────────────────────────────────────────────


class MondriaanGenerator:
    """Synthetic pixel-level pattern-matching benchmark generator."""

    @staticmethod
    def _random_positions(image_size: int, n: int, min_sep: int = 16):
        """Sample n non-overlapping 4×4-aligned top-left corners.

        Positions are on a 4-pixel grid (0, 4, 8, …, image_size−4).
        Euclidean distance between any two top-left corners ≥ min_sep.
        """
        grid = list(range(0, image_size - 3, 4))
        positions = []
        max_attempts = n * 20_000
        attempts = 0
        while len(positions) < n:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not place {n} objects in {image_size}×{image_size} "
                    f"with min_sep={min_sep} after {max_attempts} attempts."
                )
            attempts += 1
            y = random.choice(grid)
            x = random.choice(grid)
            if any((y - py) ** 2 + (x - px) ** 2 < min_sep**2 for py, px in positions):
                continue
            positions.append((y, x))
        return positions

    @staticmethod
    def generate_single(
        image_size: int = 64,
        object_type: str = "patterned",
        target_label: int | None = None,
        min_sep: int = 16,
        random_colors: bool = False,
    ):
        """Generate one Mondriaan sample.

        Parameters
        ----------
        image_size:
            Height = width of the square image (default 64).
        object_type:
            ``"patterned"`` (mean-preserving, pooling fails) or
            ``"solid"`` (control, pooling succeeds).
        target_label:
            Force label to 0 or 1. None → chosen uniformly at random.
        min_sep:
            Minimum Euclidean distance between object top-left corners (px).
        random_colors:
            If True, sample 4 colors uniformly from [0,1]^3 per image.

        Returns
        -------
        image : np.ndarray, shape (H, W, 3), float32
        label : int, 0 or 1
        obj_types : list[int], indices into the vocabulary (length 4)
        positions : list[tuple[int, int]], top-left (y, x) corners (length 4)
        """
        image = np.zeros((image_size, image_size, 3), dtype=np.float32)
        positions = MondriaanGenerator._random_positions(image_size, 4, min_sep)

        if target_label is None:
            target_label = random.randint(0, 1)

        if random_colors and object_type == "patterned":

            def _rand_pat():
                return np.random.uniform(0, 1, size=(4, 3)).astype(np.float32)

            if target_label == 1:
                pair_pat = _rand_pat()
                entries = [
                    (pair_pat, 0),
                    (pair_pat, 0),
                    (_rand_pat(), 1),
                    (_rand_pat(), 2),
                ]
            else:
                entries = [(_rand_pat(), i) for i in range(4)]
            random.shuffle(entries)
            pats, obj_types = zip(*entries)
            obj_types = list(obj_types)
            for (y, x), (tl, tr, bl, br) in zip(positions, pats):
                image[y : y + 2, x : x + 2] = tl
                image[y : y + 2, x + 2 : x + 4] = tr
                image[y + 2 : y + 4, x : x + 2] = bl
                image[y + 2 : y + 4, x + 2 : x + 4] = br
        else:
            if random_colors:
                base = np.random.uniform(0, 1, size=(4, 3)).astype(np.float32)
                vocab = [[base[i]] * 4 for i in range(4)]
            else:
                vocab = PATTERNED_VOCAB if object_type == "patterned" else SOLID_VOCAB
            n_types = len(vocab)

            if target_label == 1:
                pair = random.randrange(n_types)
                others = random.sample([t for t in range(n_types) if t != pair], 2)
                obj_types = [pair, pair] + others
                random.shuffle(obj_types)
            else:
                obj_types = random.sample(range(n_types), 4)

            for (y, x), t in zip(positions, obj_types):
                tl, tr, bl, br = vocab[t]
                image[y : y + 2, x : x + 2] = tl
                image[y : y + 2, x + 2 : x + 4] = tr
                image[y + 2 : y + 4, x : x + 2] = bl
                image[y + 2 : y + 4, x + 2 : x + 4] = br

        # strict label check — same rule as HalliGalli
        counts = Counter(obj_types)
        vals = list(counts.values())
        label = 1 if (max(vals) == 2 and sum(1 for v in vals if v >= 2) == 1) else 0

        return image, label, obj_types, positions


# ── visualisation ─────────────────────────────────────────────────────────────


def _show_samples(
    n: int = 4,
    object_type: str = "patterned",
    image_only: bool = False,
    highlight: bool = True,
    zoom: int = 16,
    **kwargs,
):
    """Visualise n Mondriaan samples.

    Each image is shown at *zoom*× magnification so individual pixels are
    visible. Matching pairs are boxed in green; singletons in yellow.

    When *image_only* is True, each sample is saved as a separate PDF
    (``mondriaan_{j}_label{lbl}.pdf``) with no axes or text panel — suitable
    for direct inclusion in a paper.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available")
        return

    names = PATTERNED_NAMES if object_type == "patterned" else SOLID_NAMES

    def _add_boxes(ax, obj_types, positions, lbl):
        if not highlight:
            return
        counts = Counter(obj_types)
        repeated = {t for t, k in counts.items() if k >= 2}
        pair_idx = (
            {i for i, t in enumerate(obj_types) if t in repeated} if lbl == 1 else set()
        )
        for i, (y, x) in enumerate(positions):
            color = "lime" if i in pair_idx else "yellow"
            rect = mpatches.Rectangle(
                (x * zoom - 0.5, y * zoom - 0.5),
                4 * zoom,
                4 * zoom,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
        return pair_idx

    if image_only:
        for j in range(n):
            img, lbl, obj_types, positions = MondriaanGenerator.generate_single(
                object_type=object_type, **kwargs
            )
            zoomed = np.kron(img, np.ones((zoom, zoom, 1), dtype=np.float32))
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(zoomed, interpolation="nearest")
            ax.axis("off")
            _add_boxes(ax, obj_types, positions, lbl)
            fname = f"mondriaan_{j}_label{lbl}.pdf"
            fig.tight_layout(pad=0)
            plt.savefig(fname, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    if n == 1:
        axes = axes[:, None]

    for j in range(n):
        img, lbl, obj_types, positions = MondriaanGenerator.generate_single(
            object_type=object_type, **kwargs
        )
        zoomed = np.kron(img, np.ones((zoom, zoom, 1), dtype=np.float32))

        ax = axes[0, j]
        ax.imshow(zoomed, interpolation="nearest")
        ax.set_title(f"label={lbl}", fontsize=9)
        ax.axis("off")
        pair_idx = _add_boxes(ax, obj_types, positions, lbl) or set()

        txt = "\n".join(
            f"{i}: {names[t]}{'  *' if i in pair_idx else ''}"
            for i, t in enumerate(obj_types)
        )
        axes[1, j].text(
            0.05,
            0.5,
            txt,
            fontsize=8,
            family="monospace",
            verticalalignment="center",
            transform=axes[1, j].transAxes,
        )
        axes[1, j].axis("off")

    fig.tight_layout()
    fname = f"mondriaan_samples_{object_type}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {fname}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Mondriaan benchmark generator")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument(
        "--object_type",
        choices=["patterned", "solid"],
        default="patterned",
    )
    parser.add_argument("--min_sep", type=int, default=8)
    parser.add_argument(
        "--random_colors",
        action="store_true",
        help="Sample 4 random colours per image instead of fixed {R,C,G,M}.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument(
        "--image_only",
        action="store_true",
        help="Save one PDF per sample, no axes or text panel.",
    )
    parser.add_argument(
        "--no_highlight",
        action="store_true",
        help="Disable highlight boxes on key objects.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=16,
        help="Pixel zoom factor for visualisation (default 16).",
    )
    args = parser.parse_args()

    if args.visualize:
        _show_samples(
            n=args.n_samples,
            object_type=args.object_type,
            image_only=args.image_only,
            highlight=not args.no_highlight,
            zoom=args.zoom,
            image_size=args.image_size,
            min_sep=args.min_sep,
            random_colors=args.random_colors,
        )
    else:
        img, lbl, types, pos = MondriaanGenerator.generate_single(
            image_size=args.image_size,
            object_type=args.object_type,
            min_sep=args.min_sep,
            random_colors=args.random_colors,
        )
        names = PATTERNED_NAMES if args.object_type == "patterned" else SOLID_NAMES
        print(f"label={lbl}  types={[names[t] for t in types]}  positions={pos}")
