"""Tests for HalliGalliGenerator."""

import numpy as np
import pytest
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


def test_generate_single_label_exactly_one_pair():
    """Label is 1 iff exactly one shape type appears exactly twice."""
    for _ in range(50):
        _, label, shapes, _ = HalliGalliGenerator.generate_single(image_size=32)
        counts = Counter(shapes)
        pairs = [s for s, n in counts.items() if n == 2]
        expected = 1 if len(pairs) == 1 else 0
        assert label == expected


def test_generate_single_separation_affects_positions():
    """Higher separation pushes key positions further from centre."""
    def spread(sep):
        positions = []
        for _ in range(10):
            _, _, _, pos = HalliGalliGenerator.generate_single(
                image_size=128, separation=sep
            )
            cy = [p[0] for p in pos]
            positions.append(max(cy) - min(cy))
        return np.mean(positions)

    assert spread(0.9) > spread(0.3)


@pytest.mark.parametrize("target_label", [0, 1])
def test_target_label_is_respected(target_label):
    """generate_single with target_label always returns the requested label."""
    for _ in range(20):
        _, label, _, _ = HalliGalliGenerator.generate_single(
            image_size=32, target_label=target_label
        )
        assert label == target_label


def test_target_label_alternation_is_balanced():
    """Alternating target_label=idx%2 yields exactly 50/50 label split."""
    labels = [
        HalliGalliGenerator.generate_single(image_size=32, target_label=i % 2)[1]
        for i in range(100)
    ]
    assert labels.count(0) == 50
    assert labels.count(1) == 50
