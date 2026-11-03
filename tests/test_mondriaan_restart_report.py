"""Paired uncertainty treats shared layouts as the independent sampling units."""

import numpy as np
import pytest

from scripts.mondriaan_metrics import CASE_NAMES
from scripts.report_mondriaan_restart import layout_groups, paired_layout_bootstrap


def test_layout_bootstrap_identical_arms_have_zero_paired_uncertainty():
    negative_cases = [case for case in CASE_NAMES if case != "one_pair"]
    cases = np.array([case for i in range(128) for case in ("one_pair", negative_cases[i % 4])])
    labels = (cases == "one_pair").astype(int)
    positions = [[(i, 0), (i, 8)] for i in range(128) for _ in range(2)]
    predictions = {
        f"{arm}/{mode}": labels.copy()
        for arm in ("uninterrupted", "restart") for mode in ("selected", "final")
    }
    result, groups = paired_layout_bootstrap(labels, cases, predictions, positions, seed=42, repeats=100)
    assert result["groups"] == 128
    assert np.array_equal(groups[::2], groups[1::2])
    for mode in ("selected", "final"):
        assert all(interval == [0.0, 0.0] for interval in
                   result["paired_restart_minus_uninterrupted_intervals"][mode].values())
    for intervals in result["intervals"].values():
        assert all(interval == [1.0, 1.0] for interval in intervals.values())


def test_layout_groups_ignore_object_order_and_bootstrap_exposes_failed_case():
    negative_cases = [case for case in CASE_NAMES if case != "one_pair"]
    cases = np.array([case for i in range(128) for case in ("one_pair", negative_cases[i % 4])])
    labels = (cases == "one_pair").astype(int)
    positions = [corners for i in range(128) for corners in ([(i, 0), (i, 8)], [(i, 8), (i, 0)])]
    assert len(layout_groups(positions)) == 128
    failed = labels.copy()
    failed[cases == "all_distinct"] = 1
    result, _ = paired_layout_bootstrap(labels, cases,
                                       {"uninterrupted/final": labels, "restart/final": failed},
                                       positions, seed=7, repeats=100)
    assert result["intervals"]["restart/final"]["worst_case_accuracy"] == [0.0, 0.0]
    assert result["paired_restart_minus_uninterrupted_intervals"]["final"]["worst_case_accuracy"] == [-1.0, -1.0]
    assert result["paired_restart_minus_uninterrupted_intervals"]["final"]["accuracy"][1] < 0


def test_bootstrap_rejects_incomplete_task():
    with pytest.raises(ValueError, match="all five observed"):
        paired_layout_bootstrap([0, 1], ["all_distinct", "one_pair"],
                                {"restart/final": [0, 1]}, [[(0, 0)], [(0, 0)]], seed=42)


def test_independent_pairs_with_coincident_layouts_remain_separate():
    negative_cases = [case for case in CASE_NAMES if case != "one_pair"]
    cases = np.array([case for i in range(128) for case in ("one_pair", negative_cases[i % 4])])
    labels = (cases == "one_pair").astype(int)
    positions = [[(0, 0), (0, 8)]] * len(labels)
    pairs = np.repeat(np.arange(128), 2)
    result, ids = paired_layout_bootstrap(labels, cases, {"restart/final": labels}, positions,
                                         seed=42, repeats=100, pair_ids=pairs)
    assert result["groups"] == 128
    np.testing.assert_array_equal(ids, pairs)
