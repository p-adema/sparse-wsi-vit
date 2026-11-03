"""Meaningful controls for case balance and exact matching with all background."""

from collections import Counter

import numpy as np
import pytest
import torch

from scripts.probe_mondriaan_matching_controls import (
    control_indices,
    pair_indices,
    prepare_data,
)
from scripts.probe_mondriaan_patch_training import build_datasets


@pytest.fixture(scope="module")
def controls():
    return prepare_data(128, 128, 42)


def test_pair_control_changes_one_object_and_matches_backgrounds(controls):
    training, _, pair_eval, checks = controls
    pair = training["pair_only"]
    n = len(pair["y"]) // 2
    assert pair["x"].shape == (128, 64, 48)
    assert Counter(pair["cases"]) == {"one_pair": 64, "all_distinct": 64}
    assert checks["pair_training_positives_identical_to_original"]
    for i in range(n):
        assert pair["positions"][i] == pair["positions"][i + n]
        assert sum(a != b for a, b in zip(pair["object_types"][i],
                                        pair["object_types"][i + n], strict=True)) == 1
        assert len(set(pair["object_types"][i])) == 3
        assert len(set(pair["object_types"][i + n])) == 4
    for key in ("pair_training", "pair_evaluation"):
        assert checks[key]["background_layouts_exactly_balanced_across_labels"]
        assert checks[key]["pixel_reconstruction_exact"]
        assert checks[key]["pixel_label_oracle_accuracy"] == 1
    assert len(pair_eval["y"]) == 32
    assert checks["overlap"] == {"pair_only": 0, "upweighted": 0}


def test_upweighted_control_keeps_original_images_and_full_evaluation(controls):
    training, full_eval, _, _ = controls
    original_training, original_eval, _ = build_datasets(128, 128, 42)
    np.testing.assert_array_equal(training["upweighted"]["images"],
                                  original_training["all24_aligned"]["images"])
    np.testing.assert_array_equal(full_eval["images"], original_eval["all24_aligned"]["images"])


@pytest.mark.parametrize("pair_only", [False, True])
def test_sampling_retains_binary_balance_and_rotates_other_cases(controls, pair_only):
    training, _, _, _ = controls
    data = training["pair_only" if pair_only else "upweighted"]
    sampler = pair_indices if pair_only else control_indices
    generator = torch.Generator().manual_seed(42)
    totals = Counter()
    for _ in range(300):
        ids = sampler(data, 32, generator)
        assert len(ids) == 32
        assert data["y"][ids].sum().item() == 16
        cases = Counter(data["cases"][i] for i in ids.tolist())
        assert cases["one_pair"] == 16
        assert cases["all_distinct"] == (16 if pair_only else 8)
        if not pair_only:
            assert sorted(cases[c] for c in ("two_pairs", "three_matching", "four_matching")) == [2, 3, 3]
        totals.update(cases)
    if not pair_only:
        assert max(totals[c] for c in ("two_pairs", "three_matching", "four_matching")) - min(
            totals[c] for c in ("two_pairs", "three_matching", "four_matching")) < 60


def test_sampling_is_reproducible_and_independent_of_model_rng(controls):
    training, _, _, _ = controls
    data = training["upweighted"]
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    expected = control_indices(data, 32, first)
    torch.rand(1000)
    torch.testing.assert_close(control_indices(data, 32, second), expected)
