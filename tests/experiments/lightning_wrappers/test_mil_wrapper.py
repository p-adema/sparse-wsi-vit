"""Tests for MILWrapper initialization and forward logic."""

import torch
import pytest

from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.models.abmil import ABMIL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_wrapper() -> MILWrapper:
    net = ABMIL(in_features=64, hidden_dim=32, out_features=1, num_branches=1)
    cfg = ExperimentConfig()
    return MILWrapper(network=net, cfg=cfg, use_bce_loss=True)


def _make_multiclass_wrapper(num_classes: int = 3) -> MILWrapper:
    net = ABMIL(in_features=64, hidden_dim=32, out_features=num_classes, num_branches=1)
    cfg = ExperimentConfig()
    return MILWrapper(network=net, cfg=cfg, use_bce_loss=False)


def _make_batch(n_patches: int = 20, feature_dim: int = 64, label: int = 1) -> dict:
    return {
        "input": torch.randn(1, n_patches, feature_dim),
        "label": torch.tensor([label], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_binary_wrapper_init():
    """Binary wrapper uses BCEWithLogitsLoss and the binary accuracy metric."""
    wrapper = _make_binary_wrapper()
    assert wrapper.multiclass is False
    assert isinstance(wrapper.loss_metric, torch.nn.BCEWithLogitsLoss)


def test_multiclass_wrapper_init():
    """Multiclass wrapper uses CrossEntropyLoss when use_bce_loss=False."""
    wrapper = _make_multiclass_wrapper()
    assert wrapper.multiclass is True
    assert isinstance(wrapper.loss_metric, torch.nn.CrossEntropyLoss)


def test_multiclass_bce_wrapper_init():
    """Multiclass wrapper uses BCEWithLogitsLoss when use_bce_loss=True."""
    net = ABMIL(in_features=64, hidden_dim=32, out_features=3, num_branches=1)
    wrapper = MILWrapper(network=net, cfg=ExperimentConfig(), use_bce_loss=True)
    assert wrapper.multiclass is True
    assert isinstance(wrapper.loss_metric, torch.nn.BCEWithLogitsLoss)


# ---------------------------------------------------------------------------
# _step (shared forward + loss, no Lightning logging)
# ---------------------------------------------------------------------------

def test_binary_step_returns_scalar_loss():
    """Binary _step should return a scalar loss and integer predictions."""
    wrapper = _make_binary_wrapper()
    batch = _make_batch(label=1)
    loss, preds, out = wrapper._step(batch, wrapper.train_acc)

    assert loss.ndim == 0, "Loss must be a scalar"
    assert preds.dtype in (torch.int32, torch.int64)
    assert "logits" in out


def test_multiclass_step_returns_scalar_loss():
    """Multiclass _step should return a scalar loss with argmax predictions."""
    wrapper = _make_multiclass_wrapper(num_classes=3)
    batch = _make_batch(label=2)
    loss, preds, out = wrapper._step(batch, wrapper.train_acc)

    assert loss.ndim == 0
    assert preds.shape == (1,)
    assert preds[0].item() in (0, 1, 2)


def test_step_accumulates_accuracy():
    """Calling _step twice should accumulate accuracy over both batches."""
    wrapper = _make_binary_wrapper()
    for _ in range(3):
        wrapper._step(_make_batch(label=0), wrapper.train_acc)

    acc = wrapper.train_acc.compute()
    assert 0.0 <= acc.item() <= 1.0
    wrapper.train_acc.reset()


# ---------------------------------------------------------------------------
# Integration: ABMIL + MILWrapper forward
# ---------------------------------------------------------------------------

def test_mil_wrapper_integration_binary():
    """End-to-end: ABMIL inside MILWrapper produces a finite loss."""
    wrapper = _make_binary_wrapper()
    batch = _make_batch(n_patches=50, feature_dim=64, label=0)
    loss, _, _ = wrapper._step(batch, wrapper.train_acc)

    assert torch.isfinite(loss), "Loss must be finite"


def test_mil_wrapper_integration_multiclass():
    """End-to-end: ABMIL inside MILWrapper produces a finite CE loss."""
    wrapper = _make_multiclass_wrapper(num_classes=4)
    batch = {
        "input": torch.randn(1, 30, 64),
        "label": torch.tensor([2], dtype=torch.long),
    }
    loss, _, _ = wrapper._step(batch, wrapper.train_acc)

    assert torch.isfinite(loss), "Loss must be finite"
