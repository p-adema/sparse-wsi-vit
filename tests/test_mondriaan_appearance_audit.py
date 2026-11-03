"""Scalar baselines fit training rules and preserve pair-task nuisance balance."""

import numpy as np
import pytest

from scripts.audit_mondriaan_appearances import (
    fit_interval,
    fit_threshold,
    distribution_matching_diagnostic,
    hamming_diversity,
    matched_pair_indices,
    predict,
    rgb_diversity,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    make_appearance_pools,
    make_controlled_dataset,
)


def test_interval_finds_middle_positives_and_handles_score_ties():
    scores = np.array([0, 1, 1, 2, 3, 4])
    labels = np.array([0, 1, 1, 1, 0, 0])
    rule = fit_interval(scores, labels)
    assert rule["training_accuracy"] == 1
    np.testing.assert_array_equal(predict(rule, np.array([0.1, 1.5, 3.5])), [0, 1, 0])
    assert fit_interval(scores, np.zeros(6))["training_accuracy"] == 1
    np.testing.assert_array_equal(predict(fit_interval(scores, np.zeros(6)), scores), np.zeros(6))


def test_threshold_supports_both_directions_and_constants():
    scores = np.arange(4)
    for labels in [np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), np.zeros(4), np.ones(4)]:
        rule = fit_threshold(scores, labels)
        assert rule["training_accuracy"] == 1
        np.testing.assert_array_equal(predict(rule, scores), labels)


def test_matched_pair_subset_has_exact_shared_layouts_and_palettes():
    data = make_controlled_dataset(make_appearance_pools()["train"], 32,
                                   palette_mode="shared")
    subset = matched_pair_indices(data)
    assert len(subset) == 8
    np.testing.assert_array_equal(data["y"][subset], [1, 0] * 4)
    for positive, negative in subset.reshape(-1, 2):
        assert data["positions"][positive] == data["positions"][negative]
        np.testing.assert_array_equal(data["palettes"][positive], data["palettes"][negative])
    scores = rgb_diversity(data)
    for index, case in enumerate(data["cases"]):
        if case == "all_matching":
            assert scores[index] == 0
        else:
            assert scores[index] > 0


def test_palette_invariant_diagnostic_and_zero_atom_are_preserved():
    data = make_controlled_dataset(make_appearance_pools()["train"], 64)
    recoloured = {**data, "palettes": data["palettes"] * 0.4 + 0.2}
    np.testing.assert_allclose(rgb_diversity(data, normalize_palette=True),
                               rgb_diversity(recoloured, normalize_palette=True), atol=1e-7)
    np.testing.assert_array_equal(hamming_diversity(data), hamming_diversity(recoloured))
    diagnostic = distribution_matching_diagnostic(data, hamming_diversity(data))
    assert diagnostic["zero_atom_separate"]
    assert diagnostic["positive_exact_zero_count"] == 0
    assert diagnostic["by_negative_case"]["all_matching"]["paired_bin_retention"] == 0
    with pytest.raises(ValueError, match="training data only"):
        distribution_matching_diagnostic({**data, "split": "test"}, hamming_diversity(data))
