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


def test_collate_preserves_all_patches():
    bag = _make_bag(64)
    out = _mil_collate_fn([bag])
    assert out["input"].shape == (1, 64, 8)
    assert out["coords"].shape == (1, 64, 2)


def test_collate_passthrough_fields():
    """Non-tensor fields in the bag dict are preserved unchanged."""
    bag = _make_bag(16)
    out = _mil_collate_fn([bag])
    assert out["slide_name"] == "test"
