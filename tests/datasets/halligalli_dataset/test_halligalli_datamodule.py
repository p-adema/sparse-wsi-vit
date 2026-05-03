"""Tests for HalliGalliH5DataModule collate logic."""

import torch
import pytest

from sparse_wsi_vit.experiments.datamodules.halligalli_h5_datamodule import (
    _mil_collate_fn,
)


def _make_bag(n_patches: int, feature_dim: int = 8, label: int = 0) -> dict:
    """Create a fake MIL bag with a regular grid of coords."""
    side = int(n_patches ** 0.5)
    assert side * side == n_patches, "n_patches must be a perfect square"
    coords = torch.tensor(
        [[j * 256, i * 256] for i in range(side) for j in range(side)],
        dtype=torch.float32,
    )
    return {
        "input": torch.randn(n_patches, feature_dim),
        "coords": coords,
        "label": torch.tensor(label),
        "slide_name": "test",
    }


def test_collate_adds_batch_dim():
    bag = _make_bag(16)
    out = _mil_collate_fn([bag])
    assert out["input"].shape == (1, 16, 8)
    assert out["label"].shape == (1,)
    assert out["coords"].shape == (1, 16, 2)


def test_collate_rejects_multi_bag():
    bag = _make_bag(16)
    with pytest.raises(AssertionError):
        _mil_collate_fn([bag, bag])


def test_corners_only_returns_four_patches():
    bag = _make_bag(64)  # 8×8 grid
    out = _mil_collate_fn([bag], corners_only=True)
    assert out["input"].shape == (1, 4, 8)
    assert out["coords"].shape == (1, 4, 2)


def test_corners_only_selects_extreme_coords():
    bag = _make_bag(64)  # 8×8 grid, coords from (0,0) to (1792,1792)
    out = _mil_collate_fn([bag], corners_only=True)
    coords = out["coords"][0]  # (4, 2)

    xs = coords[:, 0].tolist()
    ys = coords[:, 1].tolist()
    assert set(xs) == {0.0, 1792.0}, f"Unexpected x values: {xs}"
    assert set(ys) == {0.0, 1792.0}, f"Unexpected y values: {ys}"


def test_corners_only_features_match_coords():
    """Each returned feature vector must correspond to a corner coord in the original bag."""
    bag = _make_bag(64)
    out = _mil_collate_fn([bag], corners_only=True)

    orig_coords = bag["coords"]
    orig_feats = bag["input"]
    sel_coords = out["coords"][0]
    sel_feats = out["input"][0]

    for i in range(4):
        coord = sel_coords[i]
        feat = sel_feats[i]
        # find the matching row in the original bag
        matches = (orig_coords == coord).all(dim=1)
        assert matches.any(), f"Corner coord {coord} not found in original coords"
        orig_feat = orig_feats[matches][0]
        assert torch.allclose(feat, orig_feat), "Feature mismatch for corner patch"


def test_corners_only_false_returns_all_patches():
    bag = _make_bag(64)
    out = _mil_collate_fn([bag], corners_only=False)
    assert out["input"].shape == (1, 64, 8)
