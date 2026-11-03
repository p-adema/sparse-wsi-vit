"""Variable-count minibatches must not introduce a count--label shortcut."""

from collections import Counter

import pytest
import torch

from scripts.probe_mondriaan_benchmark import balanced_batch, tensor_data
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    make_appearance_pools,
    make_controlled_dataset,
)


@pytest.fixture(scope="module")
def variable_data():
    pool = make_appearance_pools(sizes={"train": 32})["train"]
    return tensor_data(make_controlled_dataset(pool, 192, object_counts=(3, 4, 6)))


@pytest.mark.parametrize("batch_size", [2, 8, 32, 64])
def test_hundreds_of_batches_match_object_counts_across_labels_and_cover_families(
    variable_data, batch_size,
):
    generator = torch.Generator().manual_seed(57)
    observed_cases = set()
    count_totals = Counter()
    for _ in range(400):
        indices = balanced_batch(variable_data, batch_size, generator)
        assert len(indices) == batch_size
        selected_labels = variable_data["y"][indices]
        counts = variable_data["object_counts"][indices.numpy()]
        positive = Counter(map(int, counts[selected_labels.numpy() == 1]))
        negative = Counter(map(int, counts[selected_labels.numpy() == 0]))
        assert positive == negative
        assert sum(positive.values()) == batch_size // 2
        observed_cases.update(variable_data["cases"][i] for i in indices.tolist())
        count_totals.update(positive)
    assert observed_cases == set(variable_data["cases"])
    assert set(count_totals) == {3, 4, 6}
    # Random remainder allocation must not always favour one count.
    assert max(count_totals.values()) - min(count_totals.values()) < 90


def test_variable_sampling_uses_private_reproducible_rng(variable_data):
    first = torch.Generator().manual_seed(81)
    second = torch.Generator().manual_seed(81)
    before = torch.get_rng_state().clone()
    for _ in range(20):
        torch.testing.assert_close(balanced_batch(variable_data, 32, first),
                                   balanced_batch(variable_data, 32, second), atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
