"""Quadrant features preserve the task; flattened patches also preserve off-grid pixels."""

from collections import Counter

import numpy as np
import pytest

from scripts.probe_mondriaan_controls import make_variant
from scripts.probe_mondriaan_generalization import data_checks, make_view, vocabulary_split
from scripts.probe_mondriaan_vit5 import make_data, tokenize_image


@pytest.fixture(scope="module")
def source():
    return make_data(40, 42, 2, False, retain_images=True, interior_only=True)


def reconstruct(tokens, coords, flattened):
    image = np.zeros((32, 32, 3), dtype=np.float32)
    for token, (x, y) in zip(tokens, coords, strict=True):
        if np.all(token == -1):
            continue
        image[y:y + 2, x:x + 2] = token.reshape(2, 2, 3) if flattened else token
    return image


def test_aligned_quadrants_are_lossless_and_pooling_bags_are_identical(source):
    bags = []
    for tokens, coords, original in zip(source["x"].numpy(), source["coords"].int().numpy(),
                                       source["images"], strict=True):
        assert tokens.shape == (256, 3)
        assert (~(tokens == -1).all(axis=-1)).sum() == 16
        np.testing.assert_array_equal(reconstruct(tokens, coords, False), original)
        bags.append(Counter(token.tobytes() for token in tokens))
    assert len(source["indices"]) == 5
    assert all(bag == bags[0] for bag in bags)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_flattened_quadrant_patches_preserve_all_pixels_after_shifts(source, shift):
    images = source["images"] if shift == 0 else make_variant(source, shift=shift)["images"]
    for image in images:
        tokens, coords = tokenize_image(image, pool_size=2, tokenization="patch")
        assert tokens.shape == (256, 12)
        np.testing.assert_array_equal(reconstruct(tokens, coords, True), image)


def test_permutation_split_has_disjoint_appearances_and_preserves_task(source):
    training_ids, held_ids = vocabulary_split()
    assert len(training_ids) == len(held_ids) == 12
    assert set(training_ids).isdisjoint(held_ids)
    assert sorted(training_ids + held_ids) == list(range(24))
    train = make_view(source, training_ids, mapping_seed=10)
    held = make_view(source, held_ids, mapping_seed=10)
    checks = data_checks(train, {"held": held}, training_ids, held_ids, False)
    assert checks["datasets"]["held"]["oracle_accuracy"] == 1
    assert len(checks["datasets"]["held"]["observed_types_outside_training_vocabulary"]) == 12
    assert all(Counter(t.tobytes() for t in sample.numpy()) ==
               Counter(t.tobytes() for t in train["x"][0].numpy()) for sample in held["x"])


def test_augmented_training_phases_are_label_balanced_and_holdout_is_disjoint(source):
    training_ids, held_ids = vocabulary_split()
    train = make_view(source, training_ids, mapping_seed=10, phases=(0, 1, 2),
                      phase_seed=20, flattened=True)
    held = make_view(source, training_ids, mapping_seed=10, phases=(3,), flattened=True)
    assert set(train["phases"]) == {0, 1, 2}
    assert set(held["phases"]) == {3}
    positive = Counter(phase for phase, label in zip(train["phases"], train["y"].tolist(), strict=True)
                       if label == 1)
    negative = Counter(phase for phase, label in zip(train["phases"], train["y"].tolist(), strict=True)
                       if label == 0)
    assert positive == negative
    checks = data_checks(train, {"held": held}, training_ids, held_ids, True)
    assert checks["datasets"]["held"]["pixel_reconstruction_exact"]
    assert checks["datasets"]["held"]["oracle_accuracy"] == 1
