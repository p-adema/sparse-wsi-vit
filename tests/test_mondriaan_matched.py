"""Challenge rejection must retain whole nuisance pairs and a train-only policy."""

from collections import Counter

import numpy as np
import pytest

from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    audit_splits,
    make_appearance_pools,
    make_controlled_dataset,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.matched import (
    build_matched_subcase_challenge,
    challenge_checks,
    fit_matching_policy,
)


@pytest.fixture(scope="module")
def source():
    pools = make_appearance_pools(sizes={"train": 64, "test": 64}, seed=7)
    training = make_controlled_dataset(pools["train"], 192, object_counts=(3, 4), split="train")
    return pools, training, fit_matching_policy(training, bins=4)


def test_challenge_retains_pairs_and_count_quotas_without_changing_policy(source):
    pools, training, policy = source
    frozen = policy.as_dict()
    data = build_matched_subcase_challenge(pools["test"], policy, 16, seed=91,
                                          object_counts=(3, 4), palette_mode="shared",
                                          proposal_pairs=128, max_rounds=40)
    assert policy.as_dict() == frozen
    assert data["diagnostic"] == "train_bin_matched_subcases"
    assert Counter(zip(data["object_counts"], data["y"], strict=True)) == {
        (3, 0): 4, (3, 1): 4, (4, 0): 4, (4, 1): 4,
    }
    assert set(data["cases"]) == {"one_pair", "all_distinct"}
    assert len(set(data["image_hashes"])) == len(data["y"])
    assert challenge_checks(data, policy)["whole_nuisance_pairs_retained"]
    assert not data["matching_audit"]["within_bin_continuous_scores_matched"]
    assert audit_splits({"train": training, "test": data})["train:test"]["pool_id_overlap"] == 0
    for pair in np.unique(data["pair_ids"]):
        bins = data["matching_bins"][data["pair_ids"] == pair]
        assert len(bins) == 2 and bins[0] == bins[1] and bins[0] > 0


def test_policy_rejects_evaluation_and_diagnostic_fitting(source):
    _, training, _ = source
    with pytest.raises(ValueError, match="primary training"):
        fit_matching_policy({**training, "split": "test"})
    with pytest.raises(ValueError, match="primary training"):
        fit_matching_policy({**training, "diagnostic": "challenge"})


def test_impossible_strata_and_exhausted_budget_never_relax_policy(source):
    pools, _, policy = source
    with pytest.raises(ValueError, match="zero-diversity"):
        build_matched_subcase_challenge(pools["test"], policy, 16, negative_cases=("all_matching",))
    with pytest.raises(ValueError, match="divisible"):
        build_matched_subcase_challenge(pools["test"], policy, 10, object_counts=(3, 4))
    with pytest.raises(ValueError, match="budget exhausted"):
        build_matched_subcase_challenge(pools["test"], policy, 1000,
                                       proposal_pairs=1, max_rounds=1)


def test_zero_atom_is_kept_separate(source):
    _, _, policy = source
    assert policy.score_bin(0, 4) == 0
    assert policy.score_bin(1e-12, 4) > 0
    with pytest.raises(ValueError, match="not fitted"):
        policy.score_bin(.5, 6)
