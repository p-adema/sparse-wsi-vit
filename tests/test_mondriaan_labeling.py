"""Thorough tests for Mondriaan label correctness and data integrity.

Tests cover:
  - Label matches target for every generated sample
  - Exactly-one-pair constraint holds for label=1
  - All four negative configurations are sampled uniformly for label=0
  - Rendered objects satisfy the complete strict-two truth table
  - Mean preservation holds per object for fixed colours
  - No global statistic leaks the label for patterned fixed colours
  - Repeated object types are pixel-identical in all colour modes
  - Dataset balance is 50/50
  - Patterned permutation structures have pairwise Hamming distance >= 3
  - Resolution ablation: ps4 objects collapse to the same grey token
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
import numpy as np
import pytest

from src.sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import (
    MondriaanGenerator,
    PATTERNED_VOCAB,
    _PATTERNED_PERMS,
)

N_LARGE = 2_000   # samples for statistical tests
IMAGE_SIZE = 32
MIN_SEP = 8
RNG_SEED = 0


def generate_batch(n, object_type="patterned", random_colors=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        import random; random.seed(seed)
    samples = []
    for i in range(n):
        img, label, obj_types, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE,
            object_type=object_type,
            target_label=i % 2,
            min_sep=MIN_SEP,
            random_colors=random_colors,
        )
        samples.append((img, label, obj_types, positions))
    return samples


# ── Hamming distance of permutation vocabulary ────────────────────────────────

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def test_patterned_vocab_hamming_distance():
    """All pairs of patterned types differ in >= 3 of 4 positions."""
    for i, p1 in enumerate(_PATTERNED_PERMS):
        for j, p2 in enumerate(_PATTERNED_PERMS):
            if i >= j:
                continue
            d = hamming(p1, p2)
            assert d >= 3, f"Types {i} and {j} have Hamming distance {d} < 3"


def test_patterned_vocab_fixed_colours_match_perms():
    """Fixed PATTERNED_VOCAB matches _PATTERNED_PERMS applied to {R,C,G,M}."""
    from src.sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import (
        RED, CYAN, GREEN, MAGENTA,
    )
    base = [RED, CYAN, GREEN, MAGENTA]
    for i, perm in enumerate(_PATTERNED_PERMS):
        expected = [base[k] for k in perm]
        actual = PATTERNED_VOCAB[i]
        for j, (e, a) in enumerate(zip(expected, actual)):
            assert np.allclose(e, a), f"Vocab type {i} pos {j}: expected {e} got {a}"


# ── Label correctness ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("random_colors", [False, True])
@pytest.mark.parametrize("object_type", ["patterned", "solid"])
def test_label_matches_target(object_type, random_colors):
    """Generated label always equals target_label."""
    samples = generate_batch(N_LARGE, object_type=object_type,
                             random_colors=random_colors, seed=RNG_SEED)
    for i, (_, label, obj_types, _) in enumerate(samples):
        target = i % 2
        assert label == target, (
            f"Sample {i}: target={target} but got label={label}, types={obj_types}"
        )


@pytest.mark.parametrize("random_colors", [False, True])
def test_label1_exactly_one_pair(random_colors):
    """label=1 samples have exactly one type appearing exactly twice."""
    samples = generate_batch(N_LARGE, random_colors=random_colors, seed=RNG_SEED)
    for i, (_, label, obj_types, _) in enumerate(samples):
        if label != 1:
            continue
        counts = Counter(obj_types)
        vals = list(counts.values())
        assert max(vals) == 2, f"Sample {i}: max count={max(vals)}, types={obj_types}"
        assert sum(1 for v in vals if v >= 2) == 1, (
            f"Sample {i}: multiple repeated types, types={obj_types}"
        )
        assert len(counts) == 3, f"Sample {i}: expected 3 distinct types, got {len(counts)}"


@pytest.mark.parametrize("random_colors", [False, True])
@pytest.mark.parametrize("object_type", ["patterned", "solid"])
def test_label0_covers_all_negative_configurations(object_type, random_colors):
    """All four negative configurations appear with approximately equal frequency."""
    samples = generate_batch(N_LARGE, object_type=object_type,
                             random_colors=random_colors, seed=RNG_SEED)
    expected = {(1, 1, 1, 1), (2, 2), (1, 3), (4,)}
    observed = Counter()
    for i, (_, label, obj_types, _) in enumerate(samples):
        if label != 0:
            continue
        configuration = tuple(sorted(Counter(obj_types).values()))
        assert configuration in expected, (
            f"Sample {i}: invalid negative configuration: {obj_types}"
        )
        observed[configuration] += 1
    assert set(observed) == expected
    for count in observed.values():
        assert abs(count / sum(observed.values()) - 0.25) < 0.05


# ── Balance ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("random_colors", [False, True])
def test_dataset_balance(random_colors):
    """Dataset produced by target_label=i%2 is exactly 50/50."""
    samples = generate_batch(N_LARGE, random_colors=random_colors, seed=RNG_SEED)
    labels = [s[1] for s in samples]
    n_pos = sum(labels)
    assert n_pos == N_LARGE // 2, f"Expected {N_LARGE//2} positives, got {n_pos}"


# ── Mean preservation ─────────────────────────────────────────────────────────

def test_mean_preservation_per_object_fixed_colors():
    """Fixed-colour patterned: all four object means equal (c0+c1+c2+c3)/4."""
    np.random.seed(RNG_SEED)
    import random; random.seed(RNG_SEED)
    for _ in range(200):
        img, _, _, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE, min_sep=MIN_SEP,
            object_type="patterned", random_colors=False,
        )
        means = [img[y:y+4, x:x+4].reshape(-1, 3).mean(axis=0) for y, x in positions]
        for j in range(1, 4):
            np.testing.assert_allclose(
                means[0], means[j], atol=1e-5,
                err_msg=f"Object means differ: {means[0]} vs {means[j]}"
            )


def test_random_colors_patterned_breaks_mean_preservation():
    """Only all-matching random-colour objects share every per-object mean."""
    np.random.seed(RNG_SEED)
    import random; random.seed(RNG_SEED)
    n_unequal = 0
    for _ in range(200):
        img, _, obj_types, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE, min_sep=MIN_SEP,
            object_type="patterned", random_colors=True,
        )
        means = [img[y:y+4, x:x+4].reshape(-1, 3).mean(axis=0) for y, x in positions]
        all_equal = all(np.allclose(means[0], m, atol=1e-5) for m in means[1:])
        assert all_equal == (len(set(obj_types)) == 1)
        if not all_equal:
            n_unequal += 1
    assert 0 < n_unequal < 200, "Expected both all-matching and distinct-type samples"


def test_global_mean_label_invariant_fixed_colors():
    """Fixed-colour patterned: global coloured-token mean is label-invariant."""
    samples = generate_batch(N_LARGE, random_colors=False, seed=RNG_SEED)
    means_0, means_1 = [], []
    for img, label, _, positions in samples:
        coloured = np.array([img[y:y+4, x:x+4].reshape(-1, 3)
                             for y, x in positions]).reshape(-1, 3)
        (means_1 if label == 1 else means_0).append(coloured.mean(axis=0))
    means_0, means_1 = np.array(means_0), np.array(means_1)
    for ch in range(3):
        diff = abs(means_0[:, ch].mean() - means_1[:, ch].mean())
        assert diff < 0.01, f"Channel {ch}: label-mean diff {diff:.4f} — leakage"


# ── Leakage: no global statistic should distinguish labels (fixed colours only) ──

def test_no_leakage_via_pixel_statistics_fixed_colors():
    """Fixed-colour patterned: pixel std/max/min are label-invariant."""
    samples = generate_batch(N_LARGE, random_colors=False, seed=RNG_SEED)
    stats_0 = {"std": [], "max": [], "min": []}
    stats_1 = {"std": [], "max": [], "min": []}
    for img, label, _, _ in samples:
        target = stats_1 if label == 1 else stats_0
        target["std"].append(img.std())
        target["max"].append(img.max())
        target["min"].append(img.min())
    for stat in ("std", "max", "min"):
        diff = abs(np.mean(stats_0[stat]) - np.mean(stats_1[stat]))
        assert diff < 0.01, f"{stat}: label-mean diff {diff:.4f} — leakage"


# ── Random colours: matching pair is pixel-identical ─────────────────────────

def test_random_colors_matching_pair_identical():
    """For label=1 with random_colors, the two paired objects are pixel-identical."""
    np.random.seed(RNG_SEED)
    import random; random.seed(RNG_SEED)
    for _ in range(500):
        img, label, obj_types, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE, min_sep=MIN_SEP,
            object_type="patterned", target_label=1, random_colors=True,
        )
        counts = Counter(obj_types)
        paired_type = next(t for t, c in counts.items() if c == 2)
        pair_indices = [i for i, t in enumerate(obj_types) if t == paired_type]
        assert len(pair_indices) == 2
        y0, x0 = positions[pair_indices[0]]
        y1, x1 = positions[pair_indices[1]]
        obj0 = img[y0:y0+4, x0:x0+4]
        obj1 = img[y1:y1+4, x1:x1+4]
        np.testing.assert_array_equal(
            obj0, obj1,
            err_msg="Paired objects are not pixel-identical with random_colors=True"
        )


@pytest.mark.parametrize("random_colors", [False, True])
@pytest.mark.parametrize("object_type", ["patterned", "solid"])
def test_rendered_objects_follow_truth_table(object_type, random_colors):
    """Labels agree with the multiplicities of the actual rendered pixel patterns."""
    truth_table = {
        (1, 1, 1, 1): 0,
        (1, 1, 2): 1,
        (2, 2): 0,
        (1, 3): 0,
        (4,): 0,
    }
    samples = generate_batch(500, object_type=object_type,
                             random_colors=random_colors, seed=RNG_SEED)
    observed = set()
    for img, label, obj_types, positions in samples:
        patches = [img[y:y+4, x:x+4].tobytes() for y, x in positions]
        configuration = tuple(sorted(Counter(patches).values()))
        assert configuration == tuple(sorted(Counter(obj_types).values()))
        for i in range(4):
            for j in range(i + 1, 4):
                assert (patches[i] == patches[j]) == (obj_types[i] == obj_types[j])
        assert label == truth_table[configuration]
        observed.add(configuration)
    assert observed == set(truth_table)


# ── Resolution ablation ───────────────────────────────────────────────────────

def test_ps4_objects_collapse_to_same_colour_fixed():
    """Fixed-colour patterned: ps4 collapses all objects to the same grey token."""
    np.random.seed(RNG_SEED)
    import random; random.seed(RNG_SEED)
    for _ in range(200):
        img, _, _, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE, min_sep=MIN_SEP,
            object_type="patterned", random_colors=False,
        )
        tokens = [img[y:y+4, x:x+4].mean(axis=(0, 1)) for y, x in positions]
        for j in range(1, 4):
            np.testing.assert_allclose(
                tokens[0], tokens[j], atol=1e-5,
                err_msg=f"ps4 tokens differ: {tokens[0]} vs {tokens[j]}"
            )


def test_ps4_random_colors_preserves_object_type_multiplicities():
    """Random-colour patterned: repeated types produce identical pooled tokens."""
    np.random.seed(RNG_SEED)
    import random; random.seed(RNG_SEED)
    for _ in range(200):
        img, _, obj_types, positions = MondriaanGenerator.generate_single(
            image_size=IMAGE_SIZE, min_sep=MIN_SEP,
            object_type="patterned", random_colors=True,
        )
        tokens = [img[y:y+4, x:x+4].mean(axis=(0, 1)) for y, x in positions]
        pooled_counts = Counter(t.tobytes() for t in tokens)
        assert sorted(pooled_counts.values()) == sorted(Counter(obj_types).values())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
