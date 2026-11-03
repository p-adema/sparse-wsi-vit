"""Visualise fixed-colour vs random-colour Mondriaan samples side by side."""
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATTERNED_PERMS = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
    [0, 2, 3, 1],
    [1, 3, 2, 0],
]

FIXED_COLORS = np.array([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 1.0],  # Cyan
    [0.0, 1.0, 0.0],  # Green
    [1.0, 0.0, 1.0],  # Magenta
], dtype=np.float32)


def random_positions(image_size, n, min_sep=8, rng=None):
    if rng is None:
        rng = random
    grid = list(range(0, image_size - 3, 4))
    positions = []
    for _ in range(n * 10_000):
        if len(positions) == n:
            break
        y, x = rng.choice(grid), rng.choice(grid)
        if all((y - py)**2 + (x - px)**2 >= min_sep**2 for py, px in positions):
            positions.append((y, x))
    return positions


def generate(image_size=32, random_colors=False, target_label=None, seed=None):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if random_colors:
        colors = np_rng.uniform(0, 1, size=(4, 3)).astype(np.float32)
    else:
        colors = FIXED_COLORS

    vocab = [[colors[i] for i in perm] for perm in PATTERNED_PERMS]
    n_types = len(vocab)

    if target_label is None:
        target_label = rng.randint(0, 1)

    if target_label == 1:
        pair = rng.randrange(n_types)
        others = rng.sample([t for t in range(n_types) if t != pair], 2)
        obj_types = [pair, pair] + others
        rng.shuffle(obj_types)
    else:
        obj_types = rng.sample(range(n_types), 4)

    positions = random_positions(image_size, 4, min_sep=8, rng=rng)
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)

    for (y, x), t in zip(positions, obj_types):
        tl, tr, bl, br = vocab[t]
        image[y:y+2,   x:x+2]   = tl
        image[y:y+2,   x+2:x+4] = tr
        image[y+2:y+4, x:x+2]   = bl
        image[y+2:y+4, x+2:x+4] = br

    counts = Counter(obj_types)
    vals = list(counts.values())
    label = 1 if (max(vals) == 2 and sum(1 for v in vals if v >= 2) == 1) else 0

    return image, label, obj_types, positions, colors


def add_boxes(ax, obj_types, positions, label, zoom):
    counts = Counter(obj_types)
    repeated = {t for t, k in counts.items() if k >= 2}
    pair_idx = {i for i, t in enumerate(obj_types) if t in repeated} if label == 1 else set()
    for i, (y, x) in enumerate(positions):
        color = "lime" if i in pair_idx else "yellow"
        rect = mpatches.Rectangle(
            (x * zoom - 0.5, y * zoom - 0.5), 4 * zoom, 4 * zoom,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)


ZOOM = 12
N_EACH = 4  # positive + negative for each design

fig, axes = plt.subplots(
    2, N_EACH,
    figsize=(N_EACH * 2.2, 5.5),
    gridspec_kw={"hspace": 0.05, "wspace": 0.08},
)

seeds = [1, 7, 42, 99]

for col, (label_target, seed) in enumerate(zip([1, 0, 1, 0], seeds)):
    for row, random_colors in enumerate([False, True]):
        img, lbl, obj_types, positions, colors = generate(
            random_colors=random_colors,
            target_label=label_target,
            seed=seed,
        )
        zoomed = np.kron(img, np.ones((ZOOM, ZOOM, 1)))
        ax = axes[row, col]
        ax.imshow(zoomed, interpolation="nearest")
        add_boxes(ax, obj_types, positions, lbl, ZOOM)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"label={lbl}", fontsize=9, pad=3)

# Column headers
for col, txt in enumerate(["label=1", "label=0", "label=1", "label=0"]):
    pass  # already set in title

# Row labels
for row, txt in enumerate(["Fixed\ncolors", "Random\ncolors"]):
    axes[row, 0].set_ylabel(txt, fontsize=10, labelpad=6, rotation=0,
                             ha="right", va="center")

fig.suptitle(
    "Mondriaan patterned  |  lime = matching pair  |  yellow = singleton",
    fontsize=10, y=1.01,
)

plt.savefig("mondriaan_color_comparison.png", dpi=150, bbox_inches="tight")
print("Saved mondriaan_color_comparison.png")
