"""Frozen scaling preserves data; hierarchical errors arise only from local labels."""

from collections import Counter

import numpy as np
import torch

from scripts.probe_mondriaan_composite_diagnostics import pad_canvas, _local_diagnostic
from sparse_wsi_vit.datasets.mondriaan_dataset.composite import (
    build_balanced_composite, lattice_origins, scene_pool_from_data, verify_composite,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import make_appearance_pools, make_controlled_dataset


def _data():
    appearances = make_appearance_pools(sizes={"test": 16})
    bank = scene_pool_from_data(make_controlled_dataset(appearances["test"], 16, seed=123))
    data = build_balanced_composite(bank, n_samples=8, canvas_size=80,
                                    scene_count=4, fixed_origins=lattice_origins(80, 4))
    return bank, data


def test_frozen_padding_preserves_all_local_pixels_labels_and_pair_ids():
    bank, data = _data()
    padded = pad_canvas(data, 128)
    np.testing.assert_array_equal(padded["images"][:, :80, :80], data["images"])
    np.testing.assert_array_equal(padded["local_x"], data["local_x"])
    np.testing.assert_array_equal(padded["local_labels"], data["local_labels"])
    np.testing.assert_array_equal(padded["layout_pair_ids"], data["layout_pair_ids"])
    assert padded["x"].shape == (8, 1024, 48)
    assert padded["coords"].max() == 124
    assert not padded["images"][:, 80:].any()
    verify_composite(padded, bank)


def test_hierarchical_metric_pipeline_reads_local_pixels_and_aggregates_correctly():
    class ExactLocalPixelMatching(torch.nn.Module):
        def forward(self, x, coords):
            assert coords.shape[1:] == (64, 2)
            labels = []
            for sample in x.numpy():
                counts = Counter(token.tobytes() for token in sample if not np.all(token == -1))
                labels.append(int(sorted(counts.values()) == [1, 1, 2]))
            logits = torch.full((len(labels), 2), -10.0)
            logits[torch.arange(len(labels)), labels] = 10
            return {"logits": logits}

    _, data = _data()
    result = _local_diagnostic(ExactLocalPixelMatching(), data, seed=123, selected_step=0)
    assert result["local_test"]["accuracy"] == 1
    assert result["test"]["accuracy"] == 1
    assert result["wrong_local_predictions_per_canvas"] == {0: 8}
    assert result["canvas_errors_despite_all_local_correct"] == 0
