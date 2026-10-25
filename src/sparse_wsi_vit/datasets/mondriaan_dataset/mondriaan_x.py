"""Mondriaan-X: four objects, six fixed patterns, variable background context.

Place in src/sparse_wsi_vit/datasets/mondriaan_dataset/mondriaan_x.py.
The 1-tile case delegates to the original probe's make_data for exact parity.
"""
from collections import Counter
import random

import numpy as np
import torch

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator
from sparse_wsi_vit.datasets.mondriaan_dataset.extract_mondriaan import BACKGROUND_SENTINEL

CASES = {
    (1, 1, 1, 1): "all_distinct",
    (1, 1, 2): "one_pair",
    (2, 2): "two_pairs",
    (1, 3): "three_matching",
    (4,): "four_matching",
}


def _canvas_shape(num_tiles):
    side = int(num_tiles ** 0.5)
    if side * side != num_tiles or num_tiles < 1:
        raise ValueError("num_tiles must be a positive perfect square (1, 4, 16, 64, ...)")
    return side * 32, side * 32


def _sample_positions(side, count=4, min_sep=8):
    """Return pixel-aligned top-left coordinates with the original separation rule."""
    grid = list(range(0, side - 3, 4))
    positions = []
    for _ in range(count):
        for _attempt in range(100_000):
            y, x = random.choice(grid), random.choice(grid)
            if all((y - py) ** 2 + (x - px) ** 2 >= min_sep ** 2
                   for py, px in positions):
                positions.append((y, x))
                break
        else:
            raise RuntimeError("Could not place objects")
    return positions


def make_data_x(size, seed, num_tiles, *, retain_metadata=False):
    """Generate stratified Mondriaan-X data with exactly four foreground tokens.

    Returns the same x/y/coords/cases/indices interface as probe_mondriaan_vit5.
    x: (size, 64*num_tiles, 48), coords: (size, 64*num_tiles, 2).
    Coordinates use pixel units, as in the original probe.
    """
    if size < 8 or size % 8:
        raise ValueError("size must be a positive multiple of 8")
    height, width = _canvas_shape(num_tiles)

    # Exact original distribution and RNG sequence at 64 tokens.
    if num_tiles == 1:
        from scripts.probe_mondriaan_vit5 import make_data
        return make_data(size, seed, 1, False, tokenization="object",
                         retain_images=retain_metadata)

    random.seed(seed)
    np.random.seed(seed)
    nrows, ncols = height // 4, width // 4
    ys, xs = np.mgrid[:nrows, :ncols]
    coords = np.stack((xs.ravel() * 4, ys.ravel() * 4), axis=1).astype(np.float32)
    n_tokens = nrows * ncols

    # No giant HxWx3 canvases are created; tokenization is lossless.
    features = np.full((size, n_tokens, 48), BACKGROUND_SENTINEL, dtype=np.float32)
    labels = np.empty(size, dtype=np.int64)
    cases = []
    type_multisets = []
    positions_meta = []
    object_types_meta = []
    quotas = {case: size // (2 if case == "one_pair" else 8) for case in CASES.values()}
    counts = Counter()
    attempts = 0
    filled = 0

    while filled < size:
        image, label, types, source_positions = MondriaanGenerator.generate_single(
            image_size=32, min_sep=8, object_type="patterned",
            target_label=attempts % 2, random_colors=False,
        )
        attempts += 1
        case = CASES[tuple(sorted(Counter(types).values()))]
        if counts[case] >= quotas[case]:
            continue
        new_positions = _sample_positions(height, min_sep=8)
        for (sy, sx), (dy, dx) in zip(source_positions, new_positions):
            token = image[sy:sy + 4, sx:sx + 4].reshape(48)
            features[filled, (dy // 4) * ncols + dx // 4] = token
        labels[filled] = label
        cases.append(case)
        type_multisets.append(tuple(sorted(types)))
        if retain_metadata:
            positions_meta.append(new_positions)
            object_types_meta.append(types)
        counts[case] += 1
        filled += 1

    data = {
        "x": torch.from_numpy(features),
        "y": torch.from_numpy(labels),
        "coords": torch.from_numpy(np.broadcast_to(coords, (size, n_tokens, 2)).copy()),
        "cases": cases,
        "type_multisets": type_multisets,
        "indices": {case: torch.tensor([i for i, c in enumerate(cases) if c == case])
                    for case in quotas},
    }
    if retain_metadata:
        data.update(positions=positions_meta, object_types=object_types_meta)
    return data


def check_data_x(data, num_tiles):
    """Fail loudly on broken labels, foreground counts, or shapes."""
    x, y, coords = data["x"], data["y"], data["coords"]
    assert x.shape == (len(y), 64 * num_tiles, 48), x.shape
    assert coords.shape == (len(y), 64 * num_tiles, 2), coords.shape
    foreground = ~(x == BACKGROUND_SENTINEL).all(dim=-1)
    assert torch.all(foreground.sum(dim=1) == 4), "Expected exactly four objects"
    assert int((y == 1).sum()) == len(y) // 2
    for i, types in enumerate(data["type_multisets"]):
        expected = int(tuple(sorted(Counter(types).values())) == (1, 1, 2))
        assert y[i].item() == expected
    return {"samples": len(y), "tokens": x.shape[1],
            "foreground_tokens_per_sample": 4,
            "case_counts": dict(Counter(data["cases"]))}
