"""HalliGalli: a synthetic benchmark for long-range spatial reasoning.

Four "key" shapes are placed at symmetric positions in the image
(corners by default).  The **global binary label** is 1 when exactly
one pair of matching shape types exists among the four, and 0 otherwise.
No single patch carries enough information to solve the task — the model
must compare distant patches against each other.

Designed to stress-test hierarchical encoders (HiPT, ABMIL, TransMIL):
shapes are deliberately tiny relative to image/region size, randomly
rotated and deformed, and embedded in dense visual clutter, so that
compressing a region into a fixed-dim vector is lossy enough to destroy
the shape-identity signal.

Key parameters
--------------
image_size        Height = Width.  Scale to 1024–4096+ for WSI regimes.
separation        0→1, how far apart the four key shapes are.
clutter_density   Average number of clutter elements per 256×256 region.
num_distractors   Full distractor shapes (same size as key shapes).
shape_radius      Radius of key shapes.  Default auto-scales to
                  image_size * 0.008 (tiny).
noise_sigma       Per-pixel Gaussian noise.

Shape difficulty
----------------
- Random rotation per shape (0–360°).
- ±30% scale jitter per shape.
- Smooth elastic deformation of the coordinate grid.
- Random color (uninformative).
- Partial-shape confounders near key positions.
- Dense background clutter (blobs, lines, rectangles).

Reference
---------
Moens, Beets-Tan & Pooch, "SONIC: Spectral Oriented Neural Invariant
Convolutions", ICLR 2026, Section 4 (page 6) and Table 1.
"""

import random
from collections import Counter

import numpy as np


# ── constants ─────────────────────────────────────────────────────────

ALL_SHAPES = ["circle", "triangle", "square", "cross", "star"]

BASE_COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.5, 0.5, 0.5],
    [0.8, 0.2, 0.4],
    [0.2, 0.6, 0.8],
    [0.9, 0.9, 0.9],
    [0.3, 0.3, 0.0],
]


def _random_color():
    base = np.array(random.choice(BASE_COLORS))
    return np.clip(
        base * np.random.uniform(0.5, 1.3) + np.random.uniform(-0.15, 0.15, 3),
        0,
        1,
    )


# ── shape rasteriser with rotation + deformation ──────────────────────


def _shape_mask(shape, r, angle_rad, deform_strength=0.0):
    """Return a boolean mask for *shape* in a local (2r+1)² patch.

    The shape is centred in the patch, rotated by *angle_rad*, and
    optionally deformed with a smooth displacement field.
    """
    size = 2 * r + 1
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)

    # inverse-rotate coordinates
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    yr = c * yy + s * xx
    xr = -s * yy + c * xx

    # smooth elastic deformation
    if deform_strength > 0 and r >= 4:
        coarse = max(3, r // 4)
        for arr in (yr, xr):
            noise = np.random.randn(coarse, coarse).astype(np.float64)
            # upsample with bilinear via numpy (no scipy needed), 2-pass
            rows = np.linspace(0, coarse - 1, size)
            cols = np.linspace(0, coarse - 1, size)
            # row-wise interp then col-wise
            tmp = np.zeros((coarse, size))
            for i in range(coarse):
                tmp[i] = np.interp(cols, np.arange(coarse), noise[i])
            upsampled = np.zeros((size, size))
            for j in range(size):
                upsampled[:, j] = np.interp(rows, np.arange(coarse), tmp[:, j])
            arr += upsampled * deform_strength * r

    # test membership in the un-rotated canonical shape
    if shape == "circle":
        mask = (yr**2 + xr**2) <= r**2

    elif shape == "square":
        mask = (np.abs(yr) <= r * 0.85) & (np.abs(xr) <= r * 0.85)

    elif shape == "triangle":
        h = r * 1.5
        # apex at top (-h/2), base at bottom (+h/2)
        y_norm = (yr + h / 2) / h  # 0 at apex, 1 at base
        half_w = np.maximum(r * y_norm, 0)
        mask = (y_norm >= 0) & (y_norm <= 1) & (np.abs(xr) <= half_w)

    elif shape == "cross":
        w = max(1, r * 0.3)
        mask = ((np.abs(yr) <= w) & (np.abs(xr) <= r * 0.9)) | (
            (np.abs(xr) <= w) & (np.abs(yr) <= r * 0.9)
        )

    elif shape == "star":
        dist = np.sqrt(yr**2 + xr**2)
        angle = np.arctan2(yr, xr)
        inner = 0.4 * r
        outer = inner + (r - inner) * (np.cos(5 * angle) + 1) / 2
        mask = dist <= outer

    else:
        raise ValueError(shape)

    return mask


def _stamp_shape(
    canvas, shape, cy, cx, r, angle_rad=0.0, deform_strength=0.0, color=None
):
    """Draw a rotated+deformed shape onto *canvas* (H, W, 3)."""
    H, W = canvas.shape[:2]
    mask = _shape_mask(shape, r, angle_rad, deform_strength)
    size = mask.shape[0]
    half = size // 2

    # compute bounding box, clipping to canvas
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    # clip
    my0 = max(0, -y0)
    mx0 = max(0, -x0)
    my1 = size - max(0, y1 - H)
    mx1 = size - max(0, x1 - W)
    cy0 = max(0, y0)
    cx0 = max(0, x0)
    cy1 = min(H, y1)
    cx1 = min(W, x1)

    if cy1 <= cy0 or cx1 <= cx0:
        return

    region = mask[my0:my1, mx0:mx1]
    if color is None:
        color = _random_color()
    canvas[cy0:cy1, cx0:cx1][region] = color


# ── clutter generators ────────────────────────────────────────────────


def _draw_clutter(
    canvas, density, min_r=2, max_r_frac=0.015, exclude_positions=None, exclude_radius=0
):
    """Scatter dense visual clutter across the canvas.

    *density* is the average number of elements per 256×256 region.
    Elements: small blobs, short lines, tiny rectangles.
    Clutter whose centre falls within *exclude_radius* of any position
    in *exclude_positions* is rejected and resampled.
    """
    H, W = canvas.shape[:2]
    area = H * W
    region_area = 256 * 256
    n_elements = max(1, int(density * area / region_area))
    max_r = max(min_r + 1, int(max(H, W) * max_r_frac))
    exclude_positions = exclude_positions or []

    placed = 0
    attempts = 0
    max_attempts = n_elements * 20  # safety cap
    while placed < n_elements and attempts < max_attempts:
        attempts += 1
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)

        if exclude_positions and any(
            abs(cy - ky) < exclude_radius and abs(cx - kx) < exclude_radius
            for ky, kx in exclude_positions
        ):
            continue

        color = _random_color()
        kind = np.random.randint(0, 3)

        if kind == 0:
            # small filled blob
            r = np.random.randint(min_r, max_r + 1)
            y0, y1 = max(0, cy - r), min(H, cy + r)
            x0, x1 = max(0, cx - r), min(W, cx + r)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
            canvas[y0:y1, x0:x1][blob] = color

        elif kind == 1:
            # short line (horizontal or vertical)
            length = np.random.randint(min_r, max_r * 3)
            thick = max(1, np.random.randint(1, min_r + 1))
            if np.random.rand() < 0.5:
                y0, y1 = max(0, cy), min(H, cy + thick)
                x0, x1 = max(0, cx), min(W, cx + length)
            else:
                y0, y1 = max(0, cy), min(H, cy + length)
                x0, x1 = max(0, cx), min(W, cx + thick)
            canvas[y0:y1, x0:x1] = color

        else:
            # tiny rectangle
            rh = np.random.randint(min_r, max_r + 1)
            rw = np.random.randint(min_r, max_r + 1)
            y0, y1 = max(0, cy - rh), min(H, cy + rh)
            x0, x1 = max(0, cx - rw), min(W, cx + rw)
            canvas[y0:y1, x0:x1] = color

        placed += 1


def _draw_confounders(canvas, positions, r, n_per_key=2):
    """Place partial/cropped shape fragments near each key position.

    These have the same visual vocabulary as key shapes but are
    truncated, so they can't be reliably identified — they compete
    for capacity in the region embedding.
    """
    H, W = canvas.shape[:2]
    for ky, kx in positions:
        for _ in range(n_per_key):
            shape = random.choice(ALL_SHAPES)
            angle = np.random.uniform(0, 2 * np.pi)
            # offset so the fragment overlaps the edge of the region
            offset_y = np.random.randint(-3 * r, 3 * r + 1)
            offset_x = np.random.randint(-3 * r, 3 * r + 1)
            frag_y = ky + offset_y
            frag_x = kx + offset_x
            # use a smaller or similar radius
            frag_r = max(2, int(r * np.random.uniform(0.5, 1.2)))
            _stamp_shape(
                canvas,
                shape,
                frag_y,
                frag_x,
                frag_r,
                angle_rad=angle,
                deform_strength=0.3,
            )


# ── core generator ────────────────────────────────────────────────────


class HalliGalliGenerator:
    """Synthetic long-range global classification benchmark.

    Shapes are tiny, rotated, deformed, and embedded in dense clutter
    so that hierarchical pooling methods (HiPT, ABMIL) cannot reliably
    compress shape identity into a region embedding.
    """

    @staticmethod
    def _key_positions(H, W, separation):
        cy, cx = H / 2, W / 2
        max_dy = H / 2 - H * 0.08
        max_dx = W / 2 - W * 0.08
        dy = separation * max_dy
        dx = separation * max_dx
        return [
            (int(cy - dy), int(cx - dx)),
            (int(cy - dy), int(cx + dx)),
            (int(cy + dy), int(cx - dx)),
            (int(cy + dy), int(cx + dx)),
        ]

    @staticmethod
    def generate_single(
        image_size=64,
        noise_sigma=0.0,
        shape_radius=None,
        num_distractors=0,
        separation=1.0,
        clutter_density=15,
        confounders_per_key=2,
        scale_jitter=0.3,
        deform_strength=0.25,
        key_deform_strength=0.0,
        target_label=None,
    ):
        """Create one HalliGalli sample.

        Returns (image, label, corner_shapes, positions) where image is
        (H, W, 3) float32, label is 0/1, and positions are the (y, x)
        centres of the four key shapes.
        shape_radius defaults to max(3, image_size * 0.008).
        key_deform_strength defaults to 0 so key shapes stay identifiable;
        difficulty comes from deformed distractors/confounders in the same region.
        """
        H = W = image_size
        image = np.zeros((H, W, 3), dtype=np.float32)

        # base radius: tiny relative to image
        r_base = (
            shape_radius
            if shape_radius is not None
            else max(3, int(image_size * 0.008))
        )

        # compute key positions up-front so clutter can avoid them
        positions = HalliGalliGenerator._key_positions(H, W, separation)

        # ── background clutter (drawn first, behind everything) ───
        if clutter_density > 0:
            _draw_clutter(
                image,
                clutter_density,
                exclude_positions=positions,
                exclude_radius=3 * r_base,
            )

        # ── four key shapes ───────────────────────────────────────
        if target_label == 1:
            # exactly one pair + two distinct singletons
            pair = random.choice(ALL_SHAPES)
            singletons = random.sample([s for s in ALL_SHAPES if s != pair], 2)
            corner_shapes = [pair, pair] + singletons
            random.shuffle(corner_shapes)
        elif target_label == 0:
            # equal mix of all-different (ABCD) and two-pairs (AABB)
            if random.random() < 0.5:
                corner_shapes = random.sample(ALL_SHAPES, 4)
            else:
                two = random.sample(ALL_SHAPES, 2)
                corner_shapes = two * 2
                random.shuffle(corner_shapes)
        else:
            corner_shapes = random.choices(ALL_SHAPES, k=4)

        for (cy, cx), shape_name in zip(positions, corner_shapes):
            # per-shape variation
            r = max(
                2, int(r_base * np.random.uniform(1 - scale_jitter, 1 + scale_jitter))
            )
            angle = np.random.uniform(0, 2 * np.pi)
            _stamp_shape(
                image,
                shape_name,
                cy,
                cx,
                r,
                angle_rad=angle,
                deform_strength=key_deform_strength,
            )

        # ── partial-shape confounders near key regions ────────────
        if confounders_per_key > 0:
            _draw_confounders(image, positions, r_base, n_per_key=confounders_per_key)

        # ── full distractor shapes (away from keys) ──────────────
        if num_distractors > 0:
            for _ in range(num_distractors):
                for _try in range(50):
                    dy = np.random.randint(r_base + 1, H - r_base - 1)
                    dx = np.random.randint(r_base + 1, W - r_base - 1)
                    too_close = any(
                        abs(dy - ky) < 4 * r_base and abs(dx - kx) < 4 * r_base
                        for ky, kx in positions
                    )
                    if not too_close:
                        shape = random.choice(ALL_SHAPES)
                        r = max(
                            2,
                            int(
                                r_base
                                * np.random.uniform(1 - scale_jitter, 1 + scale_jitter)
                            ),
                        )
                        angle = np.random.uniform(0, 2 * np.pi)
                        _stamp_shape(
                            image,
                            shape,
                            dy,
                            dx,
                            r,
                            angle_rad=angle,
                            deform_strength=deform_strength,
                        )
                        break

        # ── label ─────────────────────────────────────────────────
        counts = Counter(corner_shapes)
        matched = [s for s, n in counts.items() if n == 2]
        label = 1 if len(matched) == 1 else 0

        # ── per-pixel noise ───────────────────────────────────────
        if noise_sigma > 0:
            image += noise_sigma * np.random.randn(H, W, 3).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return image, label, corner_shapes, positions


# ── visualisation ─────────────────────────────────────────────────────


def _show_samples(n=8, highlight=True, **kwargs):
    """Render a row of samples with optional highlight circles on key shapes.

    When *highlight* is True, each of the four key shapes is circled.
    If the sample is positive (label=1), the two shapes forming the
    matching pair are circled in green; the two non-matching shapes
    in yellow.  For negative samples, all four are circled in yellow.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("matplotlib not available")
        return

    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6.0))
    if n == 1:
        axes = axes[:, None]

    for j in range(n):
        img, lbl, shapes, positions = HalliGalliGenerator.generate_single(**kwargs)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"label={lbl}", fontsize=9)
        axes[0, j].axis("off")

        if highlight:
            # determine which corners form the matching pair
            counts = Counter(shapes)
            pair_shape = next((s for s, k in counts.items() if k == 2), None)
            pair_idx = (
                {i for i, s in enumerate(shapes) if s == pair_shape}
                if pair_shape and lbl == 1
                else set()
            )

            # radius of the highlight circle, scaled to image size
            H = img.shape[0]
            r = kwargs.get("shape_radius") or max(3, int(H * 0.008))
            hi_r = max(r * 2.5, 8)

            for i, (cy, cx) in enumerate(positions):
                color = "lime" if i in pair_idx else "yellow"
                axes[0, j].add_patch(
                    Circle(
                        (cx, cy),
                        hi_r,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.8,
                    )
                )

        txt_lines = [
            f"{i}: {s}{' *' if lbl == 1 and s == pair_shape else ''}"
            for i, s in enumerate(shapes)
        ]
        axes[1, j].text(
            0.05,
            0.5,
            "\n".join(txt_lines),
            fontsize=8,
            family="monospace",
            verticalalignment="center",
            transform=axes[1, j].transAxes,
        )
        axes[1, j].axis("off")

    sep = kwargs.get("separation", 1.0)
    nd = kwargs.get("num_distractors", 0)
    cd = kwargs.get("clutter_density", 15)
    r = kwargs.get("shape_radius", None)
    sz = kwargs.get("image_size", 64)
    r_eff = r if r else max(3, int(sz * 0.008))
    fig.suptitle(
        f"HalliGalli  (size={sz}, r={r_eff}, sep={sep}, "
        f"distractors={nd}, clutter={cd})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    plt.savefig("halligalli_samples.pdf", dpi=600, bbox_inches="tight")
    plt.show()
    print("Saved halligalli_samples.pdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("HalliGalli benchmark generator")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--num_distractors", type=int, default=20)
    parser.add_argument("--separation", type=float, default=1.0)
    parser.add_argument("--clutter_density", type=float, default=15)
    parser.add_argument("--confounders_per_key", type=int, default=2)
    parser.add_argument("--shape_radius", type=int, default=None)
    parser.add_argument("--scale_jitter", type=float, default=0.3)
    parser.add_argument("--deform_strength", type=float, default=0.25)
    parser.add_argument("--key_deform_strength", type=float, default=0.0)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Plot and save sample images to halligalli_samples.png",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of samples to visualize (--visualize only)",
    )
    parser.add_argument(
        "--no_highlight",
        action="store_true",
        help="Disable highlight circles on key shapes (--visualize only)",
    )
    args = parser.parse_args()

    kw = {
        k: v
        for k, v in vars(args).items()
        if k not in ("visualize", "n_samples", "no_highlight")
    }

    if args.visualize:
        _show_samples(n=args.n_samples, highlight=not args.no_highlight, **kw)
