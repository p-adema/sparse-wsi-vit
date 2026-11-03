"""Composition contracts: strict local labels, split provenance and matched nuisances."""

from collections import Counter
import itertools

import numpy as np
import pytest

from sparse_wsi_vit.datasets.mondriaan_dataset.composite import (
    FIXED_PALETTE,
    aggregate_local_labels,
    build_balanced_composite,
    build_composite_splits,
    composite_subgroup_metrics,
    lattice_origins,
    local_label_oracle,
    local_predictions_to_canvas,
    opposite_label_global_count_fixture,
    predict_local_then_aggregate,
    render_index_scene,
    render_palette,
    scene_pool_from_data,
    verify_composite,
    verify_split_pools,
)


def small_pool(split="train", seed=12):
    rng = np.random.default_rng(seed)
    all_patterns, all_positions, labels, images = [], [], [], []
    cases = [(1, 1, 1, 1), (2, 2), (3, 1), (4,)]
    for i, (dy, dx) in enumerate(itertools.product([0, 4, 8], repeat=2)):
        positions = np.array([[0, 0], [0, 12], [12, 0], [12, 12]]) + [dy, dx]
        for label in [0, 1]:
            multiplicities = (2, 1, 1) if label else cases[i % 4]
            identities = [rng.permutation(np.repeat(np.arange(4), 4)).reshape(4, 4)
                          for _ in multiplicities]
            patterns = np.array([p for p, count in zip(identities, multiplicities, strict=True)
                                 for _ in range(count)], dtype=np.uint8)
            rng.shuffle(patterns)
            all_patterns.append(patterns)
            all_positions.append(positions)
            labels.append(label)
            images.append(render_palette(render_index_scene(patterns, positions), FIXED_PALETTE))
    return scene_pool_from_data({"split": split, "patterns": np.array(all_patterns),
                                "positions": np.array(all_positions), "y": np.array(labels),
                                "images": np.array(images),
                                "palettes": np.repeat(FIXED_PALETTE[None], len(labels), axis=0)})


@pytest.mark.parametrize("scene_count", [1, 2, 3, 4, 5])
def test_aggregation_exhaustive_truth_table(scene_count):
    labels = np.array(list(itertools.product([0, 1], repeat=scene_count)))
    counts = labels.sum(axis=1)
    np.testing.assert_array_equal(aggregate_local_labels(labels), counts > 0)
    np.testing.assert_array_equal(aggregate_local_labels(labels, rule="majority"), counts > scene_count / 2)
    np.testing.assert_array_equal(aggregate_local_labels(labels, rule="parity"), counts % 2)
    for k in range(scene_count + 1):
        np.testing.assert_array_equal(aggregate_local_labels(labels, rule="exact_k", exact_k=k), counts == k)


@pytest.mark.parametrize("rule", ["or", "majority", "exact_k", "parity"])
@pytest.mark.parametrize("palette_mode", ["fixed", "shared_per_canvas"])
def test_canvas_labels_nuisances_tokens_and_local_hook(rule, palette_mode):
    pool = small_pool()
    data = build_balanced_composite(pool, n_samples=16, canvas_size=80, scene_count=4,
                                    rule=rule, palette_mode=palette_mode)
    checks = verify_composite(data, pool)
    assert checks["token_count"] == 400
    assert checks["foreground_fraction"] == 256 / 6400
    assert checks["canvas_class_counts"] == {0: 8, 1: 8}
    np.testing.assert_array_equal(local_label_oracle(data), data["y"])
    assert composite_subgroup_metrics(local_label_oracle(data), data)["worst_subgroup_accuracy"] == 1

    def pixel_matching(x, coords):
        # These test scenes are aligned; complete objects are four active tokens.
        assert coords.shape[1:] == (64, 2)
        results = []
        for sample in x:
            counts = Counter(t.tobytes() for t in sample if not np.all(t == -1))
            results.append(int(sorted(counts.values()) == [1, 1, 2]))
        return np.array(results)

    local, predicted = predict_local_then_aggregate(data, pixel_matching, batch_size=3)
    np.testing.assert_array_equal(local, data["local_labels"])
    np.testing.assert_array_equal(predicted, data["y"])
    for label in [0, 1]:
        count_histogram = Counter(data["positive_scene_counts"][data["y"] == label])
        assert max(count_histogram.values()) - min(count_histogram.values()) <= 1
    np.testing.assert_array_equal(data["pair_ids"], data["layout_pair_ids"])
    with pytest.raises(ValueError, match="One binary"):
        local_predictions_to_canvas(np.zeros((16, 3)), data)
    with pytest.raises(ValueError, match="binary"):
        local_predictions_to_canvas(np.full((16, 4), 2), data)


def test_or_control_is_balanced_and_reports_failure_subgroups():
    data = build_balanced_composite(small_pool(), n_samples=16, canvas_size=80, scene_count=4)
    assert set(data["positive_scene_counts"][data["y"] == 0]) == {0}
    assert Counter(data["positive_scene_counts"][data["y"] == 1]) == {1: 2, 2: 2, 3: 2, 4: 2}
    result = composite_subgroup_metrics(np.ones(16, dtype=int), data)
    assert result["accuracy"] == 0.5
    assert result["worst_subgroup_accuracy"] == 0
    assert result["subgroups"]["positive_scene_count"]["0"]["accuracy"] == 0


def test_canonical_scene_split_check_catches_recolored_reuse():
    train, validation, test = small_pool(), small_pool("validation", 13), small_pool("test", 14)
    assert len(verify_split_pools([train, validation, test])) == 3
    shared = small_pool("test")
    with pytest.raises(ValueError, match="Split leakage"):
        verify_split_pools([train, shared], appearance_holdout=False)
    # The canonical scene check also survives changed RGB hash metadata.
    from dataclasses import replace
    recolored = replace(shared, image_hashes=tuple("different" + h for h in shared.image_hashes))
    with pytest.raises(ValueError, match="canonical scenes"):
        verify_split_pools([train, recolored], appearance_holdout=False)


def test_split_appearance_check_independent_of_scene_overlap():
    from dataclasses import replace
    train, test = small_pool(), small_pool("test", 14)
    shared_appearance = replace(test, appearance_ids=train.appearance_ids)
    with pytest.raises(ValueError, match="appearances"):
        verify_split_pools([train, shared_appearance])
    verify_split_pools([train, shared_appearance], appearance_holdout=False)


def test_split_builder_checks_banks_before_rendering_and_real_generator_adapter():
    from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
        make_appearance_pools,
        make_controlled_dataset,
    )
    appearances = make_appearance_pools(sizes={"train":16, "validation":16, "test":16})
    pools = {split: scene_pool_from_data(make_controlled_dataset(pool, 16, seed=i))
             for i, (split, pool) in enumerate(appearances.items())}
    datasets, overlaps = build_composite_splits(pools, n_samples=4, canvas_size=80, scene_count=4)
    assert all(d["split"] == name for name, d in datasets.items())
    assert all(row["appearance_overlap"] == 0 for row in overlaps)
    with pytest.raises(ValueError, match="Split leakage"):
        build_composite_splits({"train": small_pool(), "test": small_pool("test")}, n_samples=4)


def test_opposite_label_fixture_has_identical_global_patch_bags():
    data = opposite_label_global_count_fixture()
    np.testing.assert_array_equal(data["y"], [0, 1])
    np.testing.assert_array_equal(data["local_labels"], [[0, 0], [1, 1]])
    np.testing.assert_array_equal(np.any(data["images"][0] != 0, axis=-1),
                                  np.any(data["images"][1] != 0, axis=-1))
    assert not np.array_equal(data["images"][0], data["images"][1])
    assert Counter(t.tobytes() for t in data["x"][0]) == Counter(t.tobytes() for t in data["x"][1])
    assert Counter(data["assignments"][0].ravel()) == Counter(data["assignments"][1].ravel())


@pytest.mark.parametrize("size,m,tokens", [(80, 4, 400), (128, 4, 1024),
                                           (128, 9, 1024), (256, 4, 4096)])
def test_canvas_and_scene_sweeps_keep_scaling_contract(size, m, tokens):
    data = build_balanced_composite(small_pool(), n_samples=2, canvas_size=size, scene_count=m)
    assert data["x"].shape == (2, tokens, 48)
    assert data["local_x"].shape == (2, m, 64, 48)
    assert data["recommended_rope_high"] == size
    assert data["coords"].min() == 0
    assert data["coords"].max() == size - 4


def test_fixed_origins_isolate_black_padding_and_preserve_pair_groups():
    pool = small_pool()
    origins = lattice_origins(80, 4)
    small = build_balanced_composite(pool, n_samples=8, canvas_size=80, scene_count=4,
                                    fixed_origins=origins)
    large = build_balanced_composite(pool, n_samples=8, canvas_size=128, scene_count=4,
                                    fixed_origins=origins)
    np.testing.assert_array_equal(small["images"], large["images"][:, :80, :80])
    np.testing.assert_array_equal(small["scene_ids"], large["scene_ids"])
    np.testing.assert_array_equal(small["layout_pair_ids"], large["layout_pair_ids"])
    assert not large["images"][:, 80:].any()
    assert not large["images"][:, :, 80:].any()
    with pytest.raises(ValueError, match="fixed_origins"):
        build_balanced_composite(pool, n_samples=8, fixed_origins=[[0, 0]] * 4)


def test_rejects_unbalanced_bank_corruption_and_invalid_geometry():
    pool = small_pool()
    data = build_balanced_composite(pool, n_samples=2, canvas_size=80, scene_count=4)
    data["images"][0, 79, 79] = [1, 1, 1]
    with pytest.raises(ValueError):
        verify_composite(data, pool)
    with pytest.raises(ValueError):
        lattice_origins(64, 4)
    with pytest.raises(ValueError):
        lattice_origins(81, 1)
    with pytest.raises(ValueError):
        build_balanced_composite(pool, n_samples=3)
    with pytest.raises(ValueError):
        scene_pool_from_data({"split": "train", "patterns": pool.patterns[:1],
                             "positions": pool.positions[:1], "y": pool.labels[:1]})


def test_local_diagnostic_inputs_are_verified_against_canvas_pixels():
    pool = small_pool()
    data = build_balanced_composite(pool, n_samples=2, canvas_size=80, scene_count=4)
    data["local_x"][0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="Local model diagnostic"):
        verify_composite(data, pool)
