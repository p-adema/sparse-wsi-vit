"""Tests for HalliGalliGenerator, HalliGalliDataset, and HalliGalliDataModule."""

import numpy as np
import pytest
import torch
from collections import Counter

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import (
    ALL_SHAPES,
    HalliGalliGenerator,
)
from sparse_wsi_vit.experiments.datamodules.halligalli_datamodule import (
    HalliGalliDataModule,
    HalliGalliDataset,
)


# ---------------------------------------------------------------------------
# HalliGalliGenerator.generate_single
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HalliGalliDataset
# ---------------------------------------------------------------------------

def test_dataset_len():
    """__len__ returns the length passed at construction."""
    ds = HalliGalliDataset(length=17, image_size=32, patch_size=16)
    assert len(ds) == 17


def test_dataset_item_keys():
    """Each item has 'input' and 'label' keys."""
    ds = HalliGalliDataset(length=1, image_size=32, patch_size=16)
    sample = ds[0]
    assert "input" in sample
    assert "label" in sample


def test_dataset_bag_shape():
    """Bag shape is (N, D) where N=(image_size//patch_size)^2, D=3*patch_size^2."""
    image_size, patch_size = 64, 16
    ds = HalliGalliDataset(length=1, image_size=image_size, patch_size=patch_size)
    bag = ds[0]["input"]
    expected_N = (image_size // patch_size) ** 2
    expected_D = 3 * patch_size ** 2
    assert bag.shape == (expected_N, expected_D)


def test_dataset_label_dtype_and_range():
    """Label is an int64 tensor with value 0 or 1."""
    ds = HalliGalliDataset(length=20, image_size=32, patch_size=16)
    for i in range(20):
        label = ds[i]["label"]
        assert label.dtype == torch.int64
        assert label.item() in (0, 1)


def test_dataset_invalid_patch_size_raises():
    """HalliGalliDataModule raises when patch_size does not divide image_size."""
    with pytest.raises(ValueError):
        HalliGalliDataModule(image_size=64, patch_size=12)


# ---------------------------------------------------------------------------
# HalliGalliDataModule
# ---------------------------------------------------------------------------

def test_datamodule_channels():
    """input_channels and output_channels are derived from patch_size."""
    dm = HalliGalliDataModule(image_size=64, patch_size=16)
    assert dm.input_channels == 3 * 16 ** 2   # 768
    assert dm.output_channels == 2


def test_datamodule_setup_creates_datasets():
    """setup() populates train, val, and test datasets with correct lengths."""
    dm = HalliGalliDataModule(
        train_size=10, val_size=4, test_size=4,
        image_size=32, patch_size=16,
    )
    dm.setup()
    assert len(dm.train_dataset) == 10
    assert len(dm.val_dataset) == 4
    assert len(dm.test_dataset) == 4


def test_datamodule_dataloader_batch_shape():
    """DataLoader yields batches of shape (B, N, D) with correct label shape."""
    dm = HalliGalliDataModule(
        train_size=8, val_size=4, test_size=4,
        batch_size=4, num_workers=0,
        image_size=32, patch_size=16,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    B, N, D = batch["input"].shape
    assert B == 4
    assert N == (32 // 16) ** 2
    assert D == 3 * 16 ** 2
    assert batch["label"].shape == (4,)


def test_datamodule_abmil_forward():
    """Bags from the DataModule pass through ABMIL without error."""
    from sparse_wsi_vit.models.abmil import ABMIL

    dm = HalliGalliDataModule(
        train_size=4, val_size=2, test_size=2,
        batch_size=4, num_workers=0,
        image_size=32, patch_size=16,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    model = ABMIL(in_features=dm.input_channels, hidden_dim=32, out_features=2)
    model.eval()
    with torch.no_grad():
        out = model(batch["input"], return_attention=True)

    assert out["logits"].shape == (4, 2)
    assert out["attention"].shape == (4, 1, (32 // 16) ** 2)
