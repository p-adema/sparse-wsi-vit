"""Pixel-level invariants for new richer-appearance benchmark controls."""

from collections import Counter

import numpy as np
import pytest

from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    audit_splits,
    complete_object_view,
    dataset_checks,
    distance_diagnostics,
    make_appearance_pools,
    make_controlled_dataset,
    pattern_from_id,
    pattern_id,
    render_scene,
    strict_pair_label,
    vocabulary_size,
)


@pytest.mark.parametrize("variant,total", [("quadrant24", 24), ("cells8", 2520),
                                            ("pixels16", 63063000)])
def test_pattern_count_rank_and_histogram(variant, total):
    assert vocabulary_size(variant) == total
    ids = range(total) if total <= 2520 else [0, 1, 100, total // 2, total - 1]
    patterns = []
    for identity in ids:
        pattern = pattern_from_id(identity, variant)
        assert pattern_id(pattern, variant) == identity
        np.testing.assert_array_equal(np.bincount(pattern.ravel()), [4] * 4)
        patterns.append(pattern.tobytes())
    assert len(set(patterns)) == len(patterns)


@pytest.mark.parametrize("ids,label", [([0, 0, 1, 2], 1), ([0, 0, 1, 2, 3, 4], 1),
                                       ([0, 1, 2, 3, 4], 0), ([0, 0, 1, 1], 0),
                                       ([0, 0, 0, 1], 0), ([0, 0, 0, 0], 0),
                                       ([0, 0, 1, 1, 2, 3], 0)])
def test_variable_count_strict_rule(ids, label):
    assert strict_pair_label(ids) == label


def test_disjoint_pools_are_deterministic_and_have_distance_control():
    kwargs = {"sizes": {"train": 24, "validation": 12, "test": 12},
              "seed": 5, "min_distance": 6}
    first, second = make_appearance_pools(**kwargs), make_appearance_pools(**kwargs)
    union = []
    for split in first:
        np.testing.assert_array_equal(first[split].ids, second[split].ids)
        np.testing.assert_array_equal(first[split].patterns, second[split].patterns)
        assert not first[split].patterns.flags.writeable
        union.extend(first[split].patterns)
    diagnostics = distance_diagnostics(np.array(union))
    assert diagnostics["exhaustive"]
    assert diagnostics["minimum"] >= 6
    assert len({p.tobytes() for p in union}) == len(union)
    with pytest.raises(ValueError, match="Requested"):
        make_appearance_pools("quadrant24", sizes={"train": 25})
    with pytest.raises(ValueError, match="Could build"):
        make_appearance_pools("quadrant24", sizes={"train": 24}, min_distance=16,
                              max_attempts=100)


def test_known_vocabulary_control_explicitly_shares_all24():
    pools = make_appearance_pools("quadrant24", sizes={"train": 24, "validation": 24,
                                                      "test": 24}, disjoint=False)
    assert set(pools["train"].ids) == set(range(24))
    np.testing.assert_array_equal(pools["train"].ids, pools["test"].ids)


@pytest.mark.parametrize("variant", ["quadrant24", "cells8", "pixels16"])
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("palette_mode", ["fixed", "shared"])
def test_pixels_and_balanced_nuisances(variant, aligned, palette_mode):
    pools = make_appearance_pools(variant)
    data = make_controlled_dataset(pools["train"], 48, seed=13,
                                   object_counts=(3, 4, 6), aligned=aligned,
                                   palette_mode=palette_mode)
    checks = dataset_checks(data)
    assert checks["pixel_label_oracle_accuracy"] == 1
    assert checks["layouts_palettes_counts_exactly_label_balanced"]
    assert checks["equal_within_image_object_histograms_and_means"]
    assert data["x"].shape == (48, 64, 48)
    for label in [0, 1]:
        assert Counter(data["object_counts"][data["y"] == label]) == {3: 8, 4: 8, 6: 8}
    if not aligned:
        assert any(x % 4 or y % 4 for ps in data["positions"] for y, x in ps)
    crops = complete_object_view(data)
    assert crops["x"].shape == (48, 6, 48)
    assert np.array_equal(crops["mask"].sum(axis=1), data["object_counts"])
    for i, positions in enumerate(data["positions"]):
        for j, (y, x) in enumerate(positions):
            np.testing.assert_array_equal(crops["x"][i, j].reshape(4, 4, 3),
                                          data["images"][i, y:y + 4, x:x + 4])
        assert np.all(crops["x"][i, len(positions):] == -1)


def test_disjoint_scene_splits_reject_image_and_appearance_leakage():
    pools = make_appearance_pools()
    datasets = {split: make_controlled_dataset(pool, 32, seed=i)
                for i, (split, pool) in enumerate(pools.items())}
    overlap = audit_splits(datasets)
    assert all(v["identical_images"] == v["observed_appearance_overlap"] ==
               v["pool_id_overlap"] == 0 for v in overlap.values())
    with pytest.raises(ValueError, match="Identical images"):
        audit_splits({"train": datasets["train"], "test": datasets["train"]})
    another_train = make_controlled_dataset(pools["train"], 32, seed=99)
    with pytest.raises(ValueError, match="Appearance pools overlap"):
        audit_splits({"train": datasets["train"], "test": another_train})
    audit_splits({"train": datasets["train"], "test": another_train},
                 require_disjoint_appearances=False)


def test_generation_is_reproducible_and_does_not_touch_global_random_state():
    np.random.seed(100)
    before = np.random.get_state()
    pool = make_appearance_pools()["train"]
    first = make_controlled_dataset(pool, 16, seed=88, palette_mode="shared")
    second = make_controlled_dataset(pool, 16, seed=88, palette_mode="shared")
    np.testing.assert_array_equal(first["images"], second["images"])
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_checks_detect_broken_labels_layouts_and_lossy_patches():
    data = make_controlled_dataset(make_appearance_pools()["train"], 16)
    broken = {**data, "y": data["y"].copy()}
    broken["y"][0] = 1 - broken["y"][0]
    with pytest.raises(ValueError, match="strict-pair"):
        dataset_checks(broken)
    broken = {**data, "x": data["x"].copy()}
    broken["x"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="lost pixels"):
        dataset_checks(broken)


def test_explicit_palette_rerender_preserves_matching_and_equal_histograms():
    data = make_controlled_dataset(make_appearance_pools()["train"], 8)
    palette = np.array([[0.2, 0.3, 0.4], [0.4, 0.7, 0.1],
                        [0.8, 0.2, 0.6], [0.9, 0.5, 0.4]], dtype=np.float32)
    for i in range(8):
        image = render_scene(data["patterns"][i], data["positions"][i], palette)
        objects = [image[y:y + 4, x:x + 4] for y, x in data["positions"][i]]
        assert strict_pair_label([p.tobytes() for p in objects]) == data["y"][i]
        for obj in objects:
            assert [np.all(obj == c, axis=-1).sum() for c in palette] == [4] * 4


def test_invalid_configuration_fails_explicitly():
    pool = make_appearance_pools()["train"]
    with pytest.raises(ValueError, match="divisible"):
        make_controlled_dataset(pool, 10, object_counts=(4, 6))
    with pytest.raises(ValueError, match="three objects"):
        make_controlled_dataset(pool, 8, object_counts=(2,))
    with pytest.raises(ValueError, match="structure"):
        pattern_id(pattern_from_id(100, "pixels16"), "quadrant24")
