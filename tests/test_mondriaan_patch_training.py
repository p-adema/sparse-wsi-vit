"""Proposed vocabulary and placement changes keep the exact 64-patch input."""

from collections import Counter

import numpy as np
import pytest

from scripts.probe_mondriaan_patch_training import build_datasets


@pytest.fixture(scope="module")
def datasets():
    return build_datasets(128, 128, 42)


def test_all24_training_changes_only_the_vocabulary(datasets):
    training, _, checks = datasets
    six, all24 = training["six_aligned"], training["all24_aligned"]
    assert checks["training"]["six_aligned"]["observed_vocabulary_ids"] == list(range(6))
    assert checks["training"]["all24_aligned"]["observed_vocabulary_ids"] == list(range(24))
    assert six["positions"] == all24["positions"]
    np.testing.assert_array_equal(six["y"], all24["y"])
    assert all24["x"].shape == (128, 64, 48)
    for old, new in zip(six["object_types"], all24["object_types"], strict=True):
        assert sorted(Counter(old).values()) == sorted(Counter(new).values())


def test_random_training_changes_only_positions_and_covers_every_phase(datasets):
    training, _, checks = datasets
    six, random = training["six_aligned"], training["six_random"]
    assert six["object_types"] == random["object_types"]
    np.testing.assert_array_equal(six["y"], random["y"])
    assert random["x"].shape == (128, 64, 48)
    assert set(checks["training"]["six_random"]["corner_phase_counts"]) == {
        f"{y},{x}" for y in range(4) for x in range(4)}
    assert set(checks["training"]["six_random"]["patch_fragments_per_object"]) == {1, 2, 4}
    # A common image translation would give all four objects the same phase.
    assert any(len({(y % 4, x % 4) for y, x in corners}) > 1
               for corners in random["positions"])


def test_labels_pixels_and_background_are_valid_in_all_conditions(datasets):
    _, _, checks = datasets
    for group in ("training", "evaluation"):
        for data in checks[group].values():
            assert data["shape"] == [128, 64, 48]
            assert data["pixel_reconstruction_exact"]
            assert data["pixel_label_oracle_accuracy"] == 1
            assert data["foreground_pixels"] == 64
            assert data["pixels_per_palette_color"] == 16
            assert data["background_layouts_exactly_balanced_across_labels"]
            assert data["case_counts"] == {"one_pair": 64, "all_distinct": 16,
                                          "two_pairs": 16, "three_matching": 16,
                                          "four_matching": 16}
    assert all(count == 0 for overlaps in checks["overlap"].values() for count in overlaps.values())


def test_increasing_training_size_keeps_evaluation_images_identical(datasets):
    _, original, _ = datasets
    _, enlarged, _ = build_datasets(256, 128, 42)
    for name, data in original.items():
        np.testing.assert_array_equal(data["images"], enlarged[name]["images"])
        np.testing.assert_array_equal(data["y"], enlarged[name]["y"])
        assert data["positions"] == enlarged[name]["positions"]
        assert data["object_types"] == enlarged[name]["object_types"]
