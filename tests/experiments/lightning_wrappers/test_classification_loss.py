"""Tests for the classification loss and wrapper functionality."""

import warnings

import torch

from sparse_wsi_vit.experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig


# Mock Network
class MockNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.out_proj = torch.nn.Linear(10, num_classes)

    def forward(self, input_and_condition):
        # Extract input from the dictionary passed by ClassificationWrapper
        x = input_and_condition["input"]
        return {"logits": self.out_proj(x)}


def test_bce_loss():
    """Test that BCE loss is correctly configured when use_bce_loss=True."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    # Test 1: Initialize with use_bce_loss=True
    wrapper_bce = ClassificationWrapper(net, cfg, use_bce_loss=True)
    assert isinstance(wrapper_bce.loss_metric, torch.nn.BCEWithLogitsLoss), \
        "Expected BCEWithLogitsLoss when use_bce_loss=True"

    # Test 2: Initialize with use_bce_loss=False (Default)
    wrapper_ce = ClassificationWrapper(net, cfg, use_bce_loss=False)
    assert isinstance(wrapper_ce.loss_metric, torch.nn.CrossEntropyLoss), \
        "Expected CrossEntropyLoss when use_bce_loss=False"


def test_binary_classification():
    """Test that binary classification uses BCEWithLogitsLoss."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=1)

    wrapper = ClassificationWrapper(net, cfg)
    assert isinstance(wrapper.loss_metric, torch.nn.BCEWithLogitsLoss), \
        "Expected BCEWithLogitsLoss for binary classification"
    assert wrapper.multiclass is False


def test_multiclass_classification():
    """Test that multiclass classification uses CrossEntropyLoss by default."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    wrapper = ClassificationWrapper(net, cfg)
    assert isinstance(wrapper.loss_metric, torch.nn.CrossEntropyLoss), \
        "Expected CrossEntropyLoss for multiclass classification"
    assert wrapper.multiclass is True


def test_prediction_methods():
    """Test the prediction methods for multiclass and binary classification."""
    # Multiclass
    logits_multi = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.1, 0.1]])
    preds_multi = ClassificationWrapper.multiclass_prediction(logits_multi)
    assert preds_multi.tolist() == [1, 0]

    # Binary
    logits_binary = torch.tensor([[0.5], [-0.5]])
    preds_binary = ClassificationWrapper.binary_prediction(logits_binary)
    assert preds_binary.tolist() == [1, 0]
