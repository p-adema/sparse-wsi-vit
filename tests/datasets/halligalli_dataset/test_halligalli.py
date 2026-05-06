"""Tests for HalliGalliGenerator."""

import numpy as np
from collections import Counter

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import (
    ALL_SHAPES,
    HalliGalliGenerator,
)


def test_generate_single_output_types():
    """generate_single returns (ndarray, int, list[str], list[tuple])."""
    img, label, shapes, positions = HalliGalliGenerator.generate_single(image_size=64)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert isinstance(label, int)
    assert isinstance(shapes, list) and len(shapes) == 4
    assert isinstance(positions, list) and len(positions) == 4


def test_generate_single_image_shape():
    """Image has shape (H, W, 3) matching image_size."""
    for size in (32, 64, 128):
        img, *_ = HalliGalliGenerator.generate_single(image_size=size)
        assert img.shape == (size, size, 3)


def test_generate_single_pixel_range():
    """All pixel values are clipped to [0, 1]."""
    img, *_ = HalliGalliGenerator.generate_single(image_size=64, noise_sigma=0.1)
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_generate_single_shapes_from_vocabulary():
    """All four corner shapes are drawn from ALL_SHAPES."""
    _, _, shapes, _ = HalliGalliGenerator.generate_single(image_size=64)
    assert all(s in ALL_SHAPES for s in shapes)


def test_generate_single_label_any_pair():
    """Label is 1 iff any shape type appears more than once."""
    for _ in range(50):
        _, label, shapes, _ = HalliGalliGenerator.generate_single(image_size=32)
        counts = Counter(shapes)
        expected = 1 if any(n >= 2 for n in counts.values()) else 0
        assert label == expected


def test_target_label_0_all_distinct():
    """target_label=0 always yields four distinct corner shapes."""
    for _ in range(30):
        _, label, shapes, _ = HalliGalliGenerator.generate_single(
            image_size=32, target_label=0
        )
        assert label == 0
        assert len(set(shapes)) == 4, f"Expected 4 distinct shapes, got {shapes}"


def test_target_label_1_has_pair():
    """target_label=1 always yields at least one repeated shape."""
    for _ in range(30):
        _, label, shapes, _ = HalliGalliGenerator.generate_single(
            image_size=32, target_label=1
        )
        assert label == 1
        counts = Counter(shapes)
        assert any(n >= 2 for n in counts.values()), f"No pair found in {shapes}"


def test_shape_radius_parameter():
    """Custom shape_radius is accepted without error."""
    img, _, _, _ = HalliGalliGenerator.generate_single(
        image_size=64, shape_radius=20, target_label=1
    )
    assert img.shape == (64, 64, 3)
