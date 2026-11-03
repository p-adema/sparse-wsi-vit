"""Pooling keeps arrangement information; coordinates and padding are explicit."""

import pytest
import torch
import copy

from sparse_wsi_vit.models.mondriaan_pooling import (
    LearnedPooling,
    comparable_pooling_width,
    pooling_parameter_count,
)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize("coordinates", [False, True])
def test_pooling_preserves_joint_permutation_and_padding(aggregation, coordinates):
    torch.manual_seed(42)
    model = LearnedPooling(width=16, features=8, aggregation=aggregation,
                           use_coordinates=coordinates, dropout=0).eval()
    x = torch.randn(2, 4, 48)
    coords = torch.randn(2, 4, 2)
    expected = model(x, coords)["logits"]
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(model(x[:, order], coords[:, order])["logits"], expected)
    padded_x = torch.cat([x, torch.randn(2, 3, 48)], dim=1)
    padded_coords = torch.cat([coords, torch.randn(2, 3, 2)], dim=1)
    torch.testing.assert_close(
        model(padded_x, padded_coords, torch.tensor([4, 4]))["logits"], expected)


def test_coordinate_free_pooling_ignores_coordinates_but_aware_pooling_uses_them():
    torch.manual_seed(7)
    x, coords = torch.randn(2, 4, 48), torch.zeros(2, 4, 2)
    changed = coords + 16
    free = LearnedPooling(width=16, features=8, dropout=0).eval()
    aware = LearnedPooling(width=16, features=8, use_coordinates=True, dropout=0).eval()
    torch.testing.assert_close(free(x, coords)["logits"], free(x, changed)["logits"])
    assert not torch.equal(aware(x, coords)["logits"], aware(x, changed)["logits"])


@pytest.mark.parametrize("coordinates", [False, True])
def test_background_reuse_equals_dense_encoding_in_values_and_gradients(coordinates):
    torch.manual_seed(2026)
    fast = LearnedPooling(width=16, features=8, use_coordinates=coordinates, dropout=0)
    dense = copy.deepcopy(fast)
    x = torch.randn(2, 8, 48)
    x[:, 3:] = -1
    coords = torch.arange(16).reshape(1, 8, 2).expand(2, -1, -1).float()
    inputs = torch.cat([x, coords / 32], dim=-1) if coordinates else x
    expected = dense.classifier(dense.encoder(inputs).mean(dim=1))
    actual = fast(x, coords)["logits"]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for fast_p, dense_p in zip(fast.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(fast_p.grad, dense_p.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("coordinates", [False, True])
def test_comparable_pooling_count_matches_allocation_and_vit_capacity(coordinates):
    width = comparable_pooling_width(use_coordinates=coordinates)
    model = LearnedPooling(width=width, use_coordinates=coordinates)
    count = sum(p.numel() for p in model.parameters())
    assert count == pooling_parameter_count(width, use_coordinates=coordinates)
    assert abs(count / 2_381_954 - 1) < 0.02
