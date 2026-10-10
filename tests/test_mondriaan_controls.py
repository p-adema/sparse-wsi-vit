"""Robustness views preserve labels and pixels while removing training patterns/alignment."""

from collections import Counter

import numpy as np
import pytest

from scripts.probe_mondriaan_controls import (
    foreground_view,
    make_variant,
    novel_vocabulary,
)
from scripts.probe_mondriaan_vit5 import make_data
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import PATTERNED_VOCAB


@pytest.fixture(scope="module")
def source():
    return make_data(40, 42, 1, False, tokenization="object", retain_images=True,
                     interior_only=True)


@pytest.mark.parametrize("shift,novel", [(0, True), (1, False), (2, False)])
def test_robustness_views_preserve_information_and_truth_table(source, shift, novel):
    variant = make_variant(source, shift=shift, novel=novel, seed=42)
    training_patterns = {
        np.array(p).reshape(2, 2, 3).repeat(2, axis=0).repeat(2, axis=1).tobytes()
        for p in PATTERNED_VOCAB
    }
    for i, (original, altered) in enumerate(zip(source["images"], variant["images"], strict=True)):
        np.testing.assert_array_equal(np.sort(original.reshape(-1, 3), axis=0),
                                      np.sort(altered.reshape(-1, 3), axis=0))
        patterns = [altered[y + shift:y + shift + 4, x + shift:x + shift + 4].tobytes()
                    for y, x in source["positions"][i]]
        counts = Counter(patterns)
        assert sorted(counts.values()) == sorted(Counter(source["object_types"][i]).values())
        label = int(max(counts.values()) == 2 and list(counts.values()).count(2) == 1)
        assert label == variant["y"][i].item()
        if novel:
            assert all(p not in training_patterns for p in patterns)
        else:
            expected = np.zeros_like(original)
            expected[shift:, shift:] = original[:-shift, :-shift]
            np.testing.assert_array_equal(altered, expected)
        reconstructed = np.zeros_like(altered)
        for token, (x, y) in zip(variant["x"][i].numpy(), variant["coords"][i].int().tolist(),
                                 strict=True):
            if not np.all(token == -1):
                reconstructed[y:y + 4, x:x + 4] = token.reshape(4, 4, 3)
        np.testing.assert_array_equal(altered, reconstructed)
    assert foreground_view(source)["x"].shape == (40, 4, 48)
    assert foreground_view(variant)["x"].shape == (40, 16 if shift else 4, 48)


def test_novel_vocabulary_is_complete_fixed_palette_complement():
    novel = novel_vocabulary()
    assert len(novel) == 18
    patterns = list(np.array(PATTERNED_VOCAB)) + novel
    assert len({p.tobytes() for p in patterns}) == 24
    for pattern in patterns:
        np.testing.assert_array_equal(pattern.mean(axis=0), [0.5, 0.5, 0.5])
        assert len({color.tobytes() for color in pattern}) == 4
