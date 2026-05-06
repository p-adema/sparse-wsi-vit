"""Tests for ShapePatchCNN and CNNPatchExtractor."""

import numpy as np
import pytest
import torch
from PIL import Image

from sparse_wsi_vit.datasets.halligalli_dataset.patch_cnn import ShapePatchCNN


# ── helpers ───────────────────────────────────────────────────────────────────


def _save_checkpoint(tmp_path, embed_dim=64, num_classes=6):
    """Save a tiny ShapePatchCNN checkpoint for loader tests."""
    model = ShapePatchCNN(embed_dim=embed_dim, num_classes=num_classes)
    ckpt_path = tmp_path / "patch_cnn.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "val_acc": 0.99,
            "epoch": 1,
            "patch_size": 64,
            "shape_radius": 20,
        },
        ckpt_path,
    )
    return ckpt_path


# ── ShapePatchCNN ─────────────────────────────────────────────────────────────


def test_cnn_default_dims():
    model = ShapePatchCNN()
    assert model.embed_dim == 256
    assert model.num_classes == 6


def test_cnn_custom_dims():
    model = ShapePatchCNN(embed_dim=128, num_classes=5)
    assert model.embed_dim == 128
    assert model.num_classes == 5


def test_cnn_forward_shape():
    model = ShapePatchCNN()
    out = model(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 6)


def test_cnn_forward_features_shape():
    model = ShapePatchCNN()
    feats = model.forward_features(torch.randn(4, 3, 64, 64))
    assert feats.shape == (4, 256)


def test_cnn_forward_features_custom_embed_dim():
    model = ShapePatchCNN(embed_dim=128)
    feats = model.forward_features(torch.randn(2, 3, 64, 64))
    assert feats.shape == (2, 128)


def test_cnn_single_sample():
    model = ShapePatchCNN()
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape == (1, 6)
    assert model.forward_features(x).shape == (1, 256)


def test_cnn_output_dtype():
    model = ShapePatchCNN()
    x = torch.randn(2, 3, 64, 64)
    assert model(x).dtype == torch.float32
    assert model.forward_features(x).dtype == torch.float32


def test_cnn_params_trainable():
    model = ShapePatchCNN()
    assert all(p.requires_grad for p in model.parameters())


def test_cnn_head_and_features_differ():
    """forward() and forward_features() must not return the same tensor."""
    model = ShapePatchCNN(embed_dim=6, num_classes=6)
    x = torch.randn(1, 3, 64, 64)
    assert not torch.allclose(model(x), model.forward_features(x))


# ── CNNPatchExtractor ─────────────────────────────────────────────────────────


def test_extractor_loads(tmp_path):
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    ckpt = _save_checkpoint(tmp_path, embed_dim=64)
    ext = CNNPatchExtractor(ckpt, device=torch.device("cpu"))
    assert ext.embed_dim == 64


def test_extractor_forward_shape(tmp_path):
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    ckpt = _save_checkpoint(tmp_path, embed_dim=64)
    ext = CNNPatchExtractor(ckpt, device=torch.device("cpu"))
    out = ext(torch.rand(4, 3, 64, 64))
    assert out.shape == (4, 64)


def test_extractor_params_frozen(tmp_path):
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    ckpt = _save_checkpoint(tmp_path)
    ext = CNNPatchExtractor(ckpt, device=torch.device("cpu"))
    assert not any(p.requires_grad for p in ext.parameters())


def test_extractor_model_in_eval_mode(tmp_path):
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    ckpt = _save_checkpoint(tmp_path)
    ext = CNNPatchExtractor(ckpt, device=torch.device("cpu"))
    assert not ext.model.training


def test_extractor_transform_output_shape():
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    patch = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    t = CNNPatchExtractor.transform(patch)
    assert t.shape == (3, 64, 64)


def test_extractor_transform_pixel_range():
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    patch = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    t = CNNPatchExtractor.transform(patch)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_extractor_transform_dtype():
    from sparse_wsi_vit.datasets.halligalli_dataset.extract_halligalli import (
        CNNPatchExtractor,
    )
    patch = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    assert CNNPatchExtractor.transform(patch).dtype == torch.float32
