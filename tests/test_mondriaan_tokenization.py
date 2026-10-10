"""Object tokens must preserve the complete rendered arrangement and coordinates."""

from collections import Counter
import random

import numpy as np
import pytest

from scripts.probe_mondriaan_vit5 import tokenize_image
from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator


@pytest.mark.parametrize("random_colors", [False, True])
@pytest.mark.parametrize("label", [0, 1])
def test_object_tokens_preserve_rendered_pixels_and_repetitions(random_colors, label):
    random.seed(42)
    np.random.seed(42)
    image, _, types, _ = MondriaanGenerator.generate_single(
        image_size=32, min_sep=8, target_label=label, random_colors=random_colors,
    )
    tokens, coords = tokenize_image(image, foreground_only=True, tokenization="object")
    assert tokens.shape == (4, 48)
    assert coords.shape == (4, 2)
    assert (coords % 4 == 0).all()
    for token, (x, y) in zip(tokens, coords):
        np.testing.assert_array_equal(token.reshape(4, 4, 3), image[y:y+4, x:x+4])
    token_counts = Counter(token.tobytes() for token in tokens)
    assert sorted(token_counts.values()) == sorted(Counter(types).values())


def test_object_grid_preserves_pattern_order_even_when_means_are_equal():
    random.seed(42)
    image, _, _, _ = MondriaanGenerator.generate_single(
        image_size=32, min_sep=8, target_label=1, random_colors=False,
    )
    tokens, coords = tokenize_image(image, tokenization="object")
    assert tokens.shape == (64, 48)
    reconstructed = np.zeros_like(image)
    objects = []
    for token, (x, y) in zip(tokens, coords):
        if np.all(token == -1):
            continue
        patch = token.reshape(4, 4, 3)
        reconstructed[y:y+4, x:x+4] = patch
        objects.append(token)
    np.testing.assert_array_equal(reconstructed, image)
    assert len(objects) == 4
    assert len({token.tobytes() for token in objects}) == 3
    means = [token.reshape(4, 4, 3).mean(axis=(0, 1)) for token in objects]
    np.testing.assert_array_equal(means, np.full((4, 3), 0.5))
