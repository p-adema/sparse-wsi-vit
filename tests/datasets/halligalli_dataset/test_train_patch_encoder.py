"""Tests for synthetic patch generation and ShapePatchDataset."""

import numpy as np
import pytest
import torch

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import ALL_SHAPES
from sparse_wsi_vit.datasets.halligalli_dataset.train_patch_encoder import (
    BACKGROUND_CLASS,
    NUM_CLASSES,
    ShapePatchDataset,
    _gen_background_patch,
    _gen_shape_patch,
)


# ── _gen_shape_patch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_gen_shape_patch_all_shapes(shape):
    patch = _gen_shape_patch(shape, patch_size=64, shape_radius=20, clutter_density=0)
    assert patch.shape == (64, 64, 3)


def test_gen_shape_patch_pixel_range():
    patch = _gen_shape_patch("circle", 64, 20, clutter_density=4)
    assert patch.min() >= 0.0
    assert patch.max() <= 1.0


def test_gen_shape_patch_dtype():
    patch = _gen_shape_patch("square", 64, 20, clutter_density=0)
    assert patch.dtype == np.float32


def test_gen_shape_patch_not_all_black():
    """A patch with a shape should have at least one non-zero pixel."""
    patch = _gen_shape_patch("circle", 64, 20, clutter_density=0)
    assert patch.max() > 0.0


# ── _gen_background_patch ─────────────────────────────────────────────────────


def test_gen_background_patch_shape():
    patch = _gen_background_patch(patch_size=64, clutter_density=4)
    assert patch.shape == (64, 64, 3)


def test_gen_background_patch_pixel_range():
    patch = _gen_background_patch(64, clutter_density=4)
    assert patch.min() >= 0.0
    assert patch.max() <= 1.0


def test_gen_background_patch_dtype():
    patch = _gen_background_patch(64, clutter_density=0)
    assert patch.dtype == np.float32


def test_gen_background_patch_zero_clutter_is_black():
    """No clutter and no shape → pure black canvas."""
    patch = _gen_background_patch(64, clutter_density=0)
    assert (patch == 0.0).all()


# ── constants ─────────────────────────────────────────────────────────────────


def test_num_classes_equals_shapes_plus_background():
    assert NUM_CLASSES == len(ALL_SHAPES) + 1


def test_background_class_index():
    assert BACKGROUND_CLASS == len(ALL_SHAPES)


# ── ShapePatchDataset ─────────────────────────────────────────────────────────


def test_dataset_length():
    ds = ShapePatchDataset(n_per_class=10)
    assert len(ds) == NUM_CLASSES * 10


def test_dataset_item_tensor_shape():
    ds = ShapePatchDataset(n_per_class=4, augment=False)
    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)


def test_dataset_item_dtype():
    ds = ShapePatchDataset(n_per_class=4, augment=False)
    img, _ = ds[0]
    assert img.dtype == torch.float32


def test_dataset_pixel_range():
    ds = ShapePatchDataset(n_per_class=4, augment=False)
    img, _ = ds[0]
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_dataset_label_type():
    ds = ShapePatchDataset(n_per_class=4, augment=False)
    _, label = ds[0]
    assert isinstance(label, int)


def test_dataset_label_range():
    ds = ShapePatchDataset(n_per_class=4, augment=False)
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label < NUM_CLASSES


def test_dataset_class_labels_sequential():
    """Items are grouped by class: first n_per_class items have label 0, etc."""
    n = 5
    ds = ShapePatchDataset(n_per_class=n, augment=False)
    for cls_idx in range(NUM_CLASSES):
        for sample_idx in range(n):
            _, label = ds[cls_idx * n + sample_idx]
            assert label == cls_idx, (
                f"Expected label {cls_idx} at index {cls_idx * n + sample_idx}, got {label}"
            )


def test_dataset_pregenerate_reproducible():
    """Same seed → identical validation patches."""
    ds1 = ShapePatchDataset(n_per_class=5, augment=False, pregenerate=True, seed=42)
    ds2 = ShapePatchDataset(n_per_class=5, augment=False, pregenerate=True, seed=42)
    img1, _ = ds1[0]
    img2, _ = ds2[0]
    assert torch.allclose(img1, img2)


def test_dataset_pregenerate_different_seeds():
    """Different seeds → different patches (with overwhelming probability)."""
    ds1 = ShapePatchDataset(n_per_class=5, augment=False, pregenerate=True, seed=0)
    ds2 = ShapePatchDataset(n_per_class=5, augment=False, pregenerate=True, seed=99)
    img1, _ = ds1[0]
    img2, _ = ds2[0]
    assert not torch.allclose(img1, img2)


def test_dataset_pregenerate_cache_size():
    n = 7
    ds = ShapePatchDataset(n_per_class=n, pregenerate=True, seed=0)
    assert len(ds._cache) == NUM_CLASSES * n


def test_dataset_augment_false_no_randomness():
    """With augment=False and pregenerate=True, two reads of the same index agree."""
    ds = ShapePatchDataset(n_per_class=5, augment=False, pregenerate=True, seed=0)
    img_a, _ = ds[0]
    img_b, _ = ds[0]
    assert torch.allclose(img_a, img_b)
