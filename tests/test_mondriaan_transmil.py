"""TransMIL uses real patch adjacency and preserves joint permutation semantics."""

import pytest
import torch

from sparse_wsi_vit.models.mondriaan_transmil import MondriaanTransMIL
from sparse_wsi_vit.models.transmil import TransMIL


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def grid(height=8, width=8, batch=1):
    coords = torch.tensor([[4 * x, 4 * y] for y in range(height)
                           for x in range(width)], dtype=torch.float32)
    return coords[None].expand(batch, -1, -1).clone()


def test_exact_row_major_order_for_rectangular_per_sample_permutations():
    x = torch.arange(2 * 6 * 48).reshape(2, 6, 48).float()
    coords = grid(2, 3, 2)
    permutations = torch.tensor([[5, 2, 0, 4, 1, 3], [1, 0, 3, 2, 5, 4]])
    shuffled_x = x.gather(1, permutations[..., None].expand_as(x))
    shuffled_coords = coords.gather(1, permutations[..., None].expand_as(coords))
    ordered, height, width = MondriaanTransMIL(num_landmarks=16).order_grid(shuffled_x, shuffled_coords)
    assert (height, width) == (2, 3)
    torch.testing.assert_close(ordered, x, rtol=0, atol=0)


def test_joint_permutation_preserves_logits_and_token_gradients():
    torch.manual_seed(42)
    model = MondriaanTransMIL(num_landmarks=32).eval()
    x = torch.randn(2, 64, 48, requires_grad=True)
    coords = grid(batch=2)
    permutation = torch.randperm(64)
    expected = model(x, coords)["logits"]
    expected.sum().backward()
    shuffled = x.detach()[:, permutation].requires_grad_()
    actual = model(shuffled, coords[:, permutation])["logits"]
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(shuffled.grad, x.grad[:, permutation], rtol=0, atol=0)


def test_coordinate_reassignment_changes_predictions_for_same_token_bag():
    torch.manual_seed(7)
    model = MondriaanTransMIL(num_landmarks=32).eval()
    x, coords = torch.randn(1, 64, 48), grid()
    permutation = torch.randperm(64)
    first = model(x, coords)["logits"]
    reassigned = model(x, coords[:, permutation])["logits"]
    assert (first - reassigned).abs().max().item() > 1e-5


def test_stock_fidelity_on_an_already_complete_square_grid():
    torch.manual_seed(2026)
    adapted = MondriaanTransMIL(num_landmarks=256).eval()
    stock = TransMIL(in_features=48, out_features=2).eval()
    stock.load_state_dict(adapted.state_dict())
    x = torch.randn(1, 64, 48)
    with torch.no_grad():
        expected = stock(x)["logits"]
        actual = adapted(x, grid())["logits"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rectangular_ppeg_and_all_parameters_have_finite_gradients():
    torch.manual_seed(42)
    model = MondriaanTransMIL(num_landmarks=16)
    shapes = []
    hook = model.pos_layer.register_forward_pre_hook(
        lambda module, inputs: shapes.append(tuple(inputs[1:])))
    logits = model(torch.randn(2, 6, 48), grid(2, 3, 2))["logits"]
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    hook.remove()
    assert shapes == [(2, 3)]
    assert torch.isfinite(logits).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for convolution in (model.pos_layer.proj, model.pos_layer.proj1, model.pos_layer.proj2):
        assert convolution.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("failure", ["duplicate", "hole", "offgrid", "negative",
                                    "nan", "translated", "inconsistent_shape"])
def test_rejects_coordinates_without_a_complete_physical_grid(failure):
    coords = grid(2, 3, 2)
    if failure == "duplicate":
        coords[:, 1] = coords[:, 0]
    elif failure == "hole":
        coords[:, -1, 0] += 4
    elif failure == "offgrid":
        coords[:, 0, 0] += 1
    elif failure == "negative":
        coords[:, 0, 0] = -4
    elif failure == "nan":
        coords[:, 0, 0] = float("nan")
    elif failure == "translated":
        coords += 4
    else:
        coords[1] = grid(3, 2)[0]
    with pytest.raises(ValueError):
        MondriaanTransMIL(num_landmarks=16)(torch.randn(2, 6, 48), coords)


def test_rejects_sparse_complete_object_diagnostic_and_padded_lengths():
    model = MondriaanTransMIL(num_landmarks=16)
    objects = torch.randn(1, 4, 48)
    corners = torch.tensor([[[0, 0], [12, 0], [0, 12], [12, 12]]]).float()
    with pytest.raises(ValueError, match="complete rectangular grid"):
        model(objects, corners)
    with pytest.raises(ValueError, match="unpadded"):
        model(torch.randn(1, 64, 48), grid(), torch.tensor([63]))
    assert model(torch.randn(1, 64, 48), grid(), torch.tensor([64]))["logits"].shape == (1, 2)


@pytest.mark.parametrize("landmarks", [16, 32, 64, 256])
def test_landmark_setting_changes_approximation_not_capacity(landmarks):
    model = MondriaanTransMIL(num_landmarks=landmarks)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_172_434
    assert model.layer1.attn.num_landmarks == landmarks
    assert model.layer2.attn.num_landmarks == landmarks


@pytest.mark.parametrize("kwargs", [{"num_landmarks": 0}, {"num_landmarks": 1.5},
                                   {"patch_size": 8}, {"in_features": 64},
                                   {"out_features": 3}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MondriaanTransMIL(**kwargs)
