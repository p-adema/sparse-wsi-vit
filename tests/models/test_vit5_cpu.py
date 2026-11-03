"""CPU execution must support both pixel and register rotary embeddings."""

import torch

from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded


def test_vit5_dense_cpu_forward_backward():
    model = VitDensePreEmbedded(
        in_features=3, out_features=2, downproj=64, depth=1, num_heads=4,
        rope_dynamic_high=32,
    ).cpu()
    pixels = torch.randn(2, 16, 3)
    coords = torch.rand(2, 16, 2) * 32
    logits = model(pixels, coords)["logits"]
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.down.weight.grad is not None
    assert torch.isfinite(model.down.weight.grad).all()
    assert model.vit5.blocks[0].attn.rope_reg is not None
