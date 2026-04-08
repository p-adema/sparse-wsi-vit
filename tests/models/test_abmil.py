import pytest
import torch
from sparse_wsi_vit.models.abmil import ABMIL


def test_abmil_initialization():
    """Test that the ABMIL model initializes correctly with default params."""
    model = ABMIL(in_features=1280, hidden_dim=256, out_features=1, num_branches=1)
    assert isinstance(model, ABMIL)
    assert hasattr(model, 'attention_v')
    assert hasattr(model, 'attention_u')


def test_abmil_forward_pass_batch():
    """Test ABMIL forward pass with batched input matching (B, N, D) shape."""
    model = ABMIL(in_features=1280, hidden_dim=128, out_features=2, num_branches=1)
    x = torch.randn(2, 50, 1280)  # Batch of 2, 50 patches, 1280 dim (CLS tokens)
    
    out = model(x)
    
    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (2, 2)


def test_abmil_forward_pass_unbatched():
    """Test ABMIL forward pass with unbatched input matching (N, D) shape."""
    model = ABMIL(in_features=1280, hidden_dim=64, out_features=1, num_branches=1)
    x = torch.randn(100, 1280)  # Single slide, 100 patches, 1280 dim
    
    out = model(x)
    
    assert isinstance(out, dict)
    assert "logits" in out
    # Since it adds a batch dimension dynamically, output should be (1, 1)
    assert out["logits"].shape == (1, 1)


def test_abmil_attention_weights():
    """Test ABMIL return_attention flag outputs correct bag weights."""
    model = ABMIL(in_features=1280, out_features=1, num_branches=2)

    x = torch.randn(4, 75, 1280)  # Batch of 4, 75 patches, 1280 dim
    out = model(x, return_attention=True)

    assert isinstance(out, dict)
    assert "logits" in out
    assert "attention" in out

    assert out["logits"].shape == (4, 1)

    # Check attention shape is (B, branches, N)
    attn = out["attention"]
    assert attn.shape == (4, 2, 75)

    # Check if attention weights sum to 1 over the N patches dimension
    attn_sum = attn.sum(dim=2)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum))


def test_abmil_invalid_input_dim():
    """Test that ABMIL raises ValueError for inputs with wrong number of dimensions."""
    model = ABMIL(in_features=1280)
    x_4d = torch.randn(2, 3, 50, 1280)
    with pytest.raises(ValueError):
        model(x_4d)


def test_abmil_invalid_feature_dim():
    """Test that ABMIL raises ValueError when the feature dimension does not match in_features."""
    model = ABMIL(in_features=1280)
    x_wrong_dim = torch.randn(2, 50, 512)
    with pytest.raises(ValueError):
        model(x_wrong_dim)


def test_abmil_attention_dropout_param():
    """Test that attention_dropout is forwarded to the classifier's Dropout layer."""
    model = ABMIL(in_features=64, hidden_dim=32, out_features=1, attention_dropout=0.5)
    dropout_layers = [m for m in model.classifier.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == 0.5
