"""Resume, restart, and validation-selection controls have distinct semantics."""

import copy
import inspect

import pytest
import torch

from scripts import probe_mondriaan_benchmark as benchmark
from scripts.mondriaan_metrics import CASE_NAMES, classification_metrics
from sparse_wsi_vit.models.mondriaan_pooling import LearnedPooling


@pytest.fixture
def tiny_data():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(19)
    cases = ["one_pair"] * 4 + [case for case in CASE_NAMES if case != "one_pair"]
    return {
        "x": torch.randn(8, 4, 48, generator=generator),
        "coords": torch.tensor([[0, 0], [4, 0], [0, 4], [4, 4]]).float().expand(8, -1, -1),
        "y": torch.tensor([int(case == "one_pair") for case in cases]),
        "cases": cases,
        "intended_cases": list(CASE_NAMES),
    }


def tiny_model():
    return LearnedPooling(width=16, features=8, dropout=0.35)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_state_equal(left, right)
    else:
        assert actual == expected


def test_saved_optimizer_resume_matches_uninterrupted_sampling_and_dropout(
    tmp_path, monkeypatch, tiny_data,
):
    sampled, dropout_rng = [], []
    original_sampler = benchmark.balanced_batch

    def trace_batch(data, batch_size, generator):
        indices = original_sampler(data, batch_size, generator)
        sampled.append(indices.clone())
        dropout_rng.append(torch.get_rng_state().clone())
        return indices

    monkeypatch.setattr(benchmark, "balanced_batch", trace_batch)
    torch.manual_seed(101)
    uninterrupted = tiny_model()
    report = benchmark.train(
        uninterrupted, tiny_data, tiny_data, seed=42, steps=8, batch_size=8,
        eval_every=2, output_dir=tmp_path / "uninterrupted", save_midpoint=4,
    )
    full_indices, full_rng = sampled[:], dropout_rng[:]
    midpoint = torch.load(tmp_path / "uninterrupted/midpoint.pt", weights_only=True)
    expected = torch.load(tmp_path / "uninterrupted/final.pt", weights_only=True)
    assert midpoint["optimizer_state"]["state"]
    assert not torch.equal(full_rng[0], full_rng[1])  # Training dropout consumes randomness.

    sampled.clear()
    dropout_rng.clear()
    torch.manual_seed(987)  # Model construction must not disturb the restored trajectory.
    resumed = tiny_model()
    resumed_report = benchmark.train(
        resumed, tiny_data, tiny_data, seed=42, steps=8, batch_size=8,
        eval_every=2, output_dir=tmp_path / "resumed", initial_state=midpoint,
    )
    actual = torch.load(tmp_path / "resumed/final.pt", weights_only=True)
    assert_state_equal(sampled, full_indices[4:])
    assert_state_equal(dropout_rng, full_rng[4:])
    for key in ("model_state", "optimizer_state", "sampler_state", "torch_rng_state", "step"):
        assert_state_equal(actual[key], expected[key])
    assert [entry["validation"] for entry in resumed_report["history"]] == [
        entry["validation"] for entry in report["history"] if entry["step"] >= 4]


def test_restart_changes_only_optimizer_state_and_preserves_next_random_stream(tmp_path, tiny_data):
    torch.manual_seed(42)
    model = tiny_model()
    benchmark.train(model, tiny_data, tiny_data, seed=42, steps=4, batch_size=8,
                    eval_every=2, output_dir=tmp_path / "prefix")
    state = torch.load(tmp_path / "prefix/final.pt", weights_only=True)
    loaded = []
    for restart in (False, True):
        branch = tiny_model()
        optimizer = torch.optim.AdamW(branch.parameters(), lr=2e-4, weight_decay=1e-4)
        sampler = torch.Generator().manual_seed(999)
        step = benchmark.load_training_state(branch, optimizer, sampler, state,
                                             restart_optimizer=restart)
        snapshot = copy.deepcopy(benchmark.state_bundle(branch, optimizer, sampler, step, {}))
        indices = benchmark.balanced_batch(tiny_data, 8, sampler)
        branch.train()
        logits = benchmark.forward_batch(branch, tiny_data, indices)
        loaded.append((snapshot, indices, logits.detach(), torch.get_rng_state().clone()))
    resumed, restarted = loaded
    for key in ("model_state", "sampler_state", "torch_rng_state", "step", "settings"):
        assert_state_equal(resumed[0][key], restarted[0][key])
    assert_state_equal(resumed[0]["optimizer_state"]["param_groups"],
                       restarted[0]["optimizer_state"]["param_groups"])
    assert resumed[0]["optimizer_state"]["state"]
    assert restarted[0]["optimizer_state"]["state"] == {}
    for index in (1, 2, 3):
        assert_state_equal(resumed[index], restarted[index])


def test_checkpoint_selection_sees_validation_only_and_uses_planned_rank(
    tmp_path, monkeypatch, tiny_data,
):
    training = {**tiny_data, "split": "train"}
    validation = {**tiny_data, "split": "validation"}
    calls = []
    rankings = [(0.1, 0.9, 0.95), (0.4, 0.7, 0.8), (0.3, 0.99, 0.99)]

    def validation_scores(model, data, batch_size=32, intended_cases=None, **kwargs):
        calls.append(data["split"])
        if data is training:
            # train_final is recorded only after checkpoint selection finishes.
            return {"accuracy": 1.0}
        assert data is validation
        worst, macro_f1, accuracy = rankings[len(calls) - 1]
        return {"worst_case_accuracy": worst, "binary_macro_f1": macro_f1,
                "accuracy": accuracy, "macro_case_accuracy": 0.5, "positive_f1": 0.5,
                "selection_eligible": True}

    monkeypatch.setattr(benchmark, "evaluate", validation_scores)
    report = benchmark.train(tiny_model(), training, validation, seed=7, steps=2,
                             batch_size=8, eval_every=1, output_dir=tmp_path)
    assert report["selected_step"] == 1
    assert calls == ["validation", "validation", "validation", "train"]
    assert "test" not in inspect.signature(benchmark.train).parameters
    assert torch.load(tmp_path / "best.pt", weights_only=True)["step"] == 1
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    assert latest["step"] == 2
    for key in ("model_state", "optimizer_state", "sampler_state", "torch_rng_state", "step"):
        assert_state_equal(latest[key], final[key])


def test_missing_planned_validation_case_is_not_selection_eligible(tmp_path, tiny_data):
    keep = torch.tensor([case != "all_distinct" for case in tiny_data["cases"]])
    validation = {**tiny_data,
                  "x": tiny_data["x"][keep], "coords": tiny_data["coords"][keep],
                  "y": tiny_data["y"][keep],
                  "cases": [case for case in tiny_data["cases"] if case != "all_distinct"]}
    with pytest.raises(ValueError, match="missing an intended subgroup"):
        benchmark.train(tiny_model(), tiny_data, validation, seed=42, steps=1,
                        batch_size=8, eval_every=1, output_dir=tmp_path)
    assert not (tmp_path / "best.pt").exists()


def test_rank_rejects_partial_metrics_and_prioritizes_case_accuracy():
    partial = classification_metrics([0, 1], [0, 1], ["all_distinct", "one_pair"])
    assert not partial["selection_eligible"]
    with pytest.raises(ValueError, match="missing an intended subgroup"):
        benchmark.checkpoint_rank(partial)
    case_strong = {"worst_case_accuracy": 0.5, "binary_macro_f1": 0.6, "accuracy": 0.7}
    overall_strong = {"worst_case_accuracy": 0.4, "binary_macro_f1": 0.9, "accuracy": 0.9}
    assert benchmark.checkpoint_rank(case_strong) > benchmark.checkpoint_rank(overall_strong)


def test_branch_selection_keeps_better_common_prefix_checkpoint(tmp_path, monkeypatch, tiny_data):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(1042)
    state = copy.deepcopy(benchmark.state_bundle(model, optimizer, sampler, 2, {}))
    prefix = tmp_path / "prefix-best.pt"
    torch.save({**state, "step": 1}, prefix)

    def lower_validation_scores(*args, **kwargs):
        return {"worst_case_accuracy": 0.3, "binary_macro_f1": 0.6, "accuracy": 0.7,
                "macro_case_accuracy": 0.5, "positive_f1": 0.5, "selection_eligible": True}

    monkeypatch.setattr(benchmark, "evaluate", lower_validation_scores)
    branch = tmp_path / "branch"
    report = benchmark.train(
        model, tiny_data, tiny_data, seed=42, steps=4, batch_size=8, eval_every=1,
        output_dir=branch, initial_state=state,
        inherited_selection={"rank": (0.8, 0.9, 0.9), "step": 1, "checkpoint": prefix},
    )
    assert report["selected_step"] == 1
    assert tuple(report["selected_rank"]) == (0.8, 0.9, 0.9)
    assert torch.load(branch / "best.pt", weights_only=True)["step"] == 1
    assert (branch / "best.pt").read_bytes() == prefix.read_bytes()


def test_missing_canvas_subgroup_does_not_shrink_selection_objective(tmp_path, tiny_data):
    validation = {**tiny_data, "cases": ["positive_scene_count_0"] * 4
                  + ["positive_scene_count_1"] * 4,
                  "intended_cases": ["positive_scene_count_0", "positive_scene_count_1",
                                     "positive_scene_count_2"]}
    metrics = benchmark.evaluate(tiny_model(), validation, batch_size=8)
    assert metrics["missing_cases"] == ["positive_scene_count_2"]
    assert not metrics["selection_eligible"]
    with pytest.raises(ValueError, match="missing an intended subgroup"):
        benchmark.train(tiny_model(), tiny_data, validation, seed=42, steps=1,
                        batch_size=8, eval_every=1, output_dir=tmp_path)
