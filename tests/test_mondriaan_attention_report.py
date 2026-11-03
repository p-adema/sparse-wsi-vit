"""Reporting must reject altered evidence and respect clustered uncertainty."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.mondriaan_metrics import CASE_NAMES, metrics_from_case_counts
from scripts.report_mondriaan_attention import (
    aligned_patch_equality_oracle, appearance_orbit_checks,
    compatible_protocols, paired_nuisance_bootstrap, verify_checkpoint,
    verify_manifest, verify_metrics, verify_selection,
)


def test_full_patch_oracle_respects_exact_equality_and_detects_corruption():
    # Each of the five partitions is presented as a full 64-token input.
    patterns = np.arange(4 * 48, dtype=np.float32).reshape(4, 48)
    partitions = ((0, 0, 1, 2), (0, 1, 2, 3), (0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 0, 0))
    full = np.full((5, 64, 48), -1, dtype=np.float32)
    for scene, identities in zip(full, partitions, strict=True):
        scene[[1, 15, 36, 60]] = patterns[list(identities)]
    np.testing.assert_array_equal(aligned_patch_equality_oracle(full), [1, 0, 0, 0, 0])
    # Oracle needs no spatial ordering; arbitrary joint token permutations agree.
    np.testing.assert_array_equal(aligned_patch_equality_oracle(full[:, ::-1]), [1, 0, 0, 0, 0])
    changed = full[:1].copy()
    changed[0, 15, 10] = np.nextafter(changed[0, 15, 10], np.float32(np.inf))
    assert aligned_patch_equality_oracle(changed).tolist() == [0]
    changed[0, 0, 0] = 0  # A corrupted background now creates a fifth foreground token.
    with pytest.raises(ValueError, match="exactly four"):
        aligned_patch_equality_oracle(changed)


def test_orbit_audit_detects_geometry_shared_under_colour_renaming():
    first = np.array([[0, 1, 2, 3]] * 4)
    renamed = np.array([2, 3, 1, 0])[first]
    other = first.T
    pools = {"train": SimpleNamespace(patterns=np.stack([first, renamed])),
             "validation": SimpleNamespace(patterns=np.stack([other])),
             "test": SimpleNamespace(patterns=np.stack([renamed]))}
    audit = appearance_orbit_checks(pools)
    assert audit["unique_geometry_orbits"] == {"train": 1, "validation": 1, "test": 1}
    assert audit["overlaps"] == {"train:validation": 0, "train:test": 1, "validation:test": 0}


def protocol(seed):
    return {"task": "local", "variant": "pixels16", "palette": "shared", "placement": "aligned",
            "representation": "scene", "object_counts": [4], "appearance_pool_size": 256,
            "train_size": 4096, "validation_size": 512, "test_size": 1024, "steps": 4000,
            "eval_every": 400, "batch_size": 32, "lr": 2e-4, "seeds": [seed],
            "models": ["vit5", "transmil", "mean"], "transmil_landmarks": 32,
            "appearance_seed_offset": 103000, "test_seed_offset": 366000,
            "output_dir": f"seed{seed}", "coordinate_scale": None, "width": None, "threads": 1}


def metrics(correct):
    return metrics_from_case_counts({case: {"count": 512 if case == "one_pair" else 128,
                                           "correct": value}
                                    for case, value in zip(CASE_NAMES, correct, strict=True)})


def test_protocol_audit_accepts_distinct_paths_but_rejects_seed_or_budget_changes():
    protocols = [protocol(seed) for seed in (42, 7, 2026)]
    assert compatible_protocols(protocols)["seeds"] == [42, 7, 2026]
    changed = deepcopy(protocols)
    changed[1]["steps"] = 2000
    with pytest.raises(ValueError, match="protocols differ"):
        compatible_protocols(changed)
    with pytest.raises(ValueError, match="three independent"):
        compatible_protocols([protocol(42)] * 3)


def test_final_manifest_audit_rejects_changed_pixels_order_and_count():
    images = np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3)
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    manifest = {"samples": 2, "split": "test", "image_sha256": hashes}
    assert len(verify_manifest({"images": images}, manifest)) == 64
    for wrong in (images[::-1], images + 1, images[:1]):
        with pytest.raises(ValueError, match="frozen image manifest"):
            verify_manifest({"images": wrong}, manifest)


def test_selection_checks_worst_case_priority_and_earliest_tie():
    # The 400-update checkpoint has higher overall accuracy but fails all-distinct.
    balanced = metrics([90, 360, 90, 90, 90])
    shortcut = metrics([0, 512, 128, 128, 128])
    history = [{"step": step, "validation": balanced if step != 400 else shortcut}
               for step in range(0, 4001, 400)]
    report = {"history": history, "selected_step": 0,
              "selected_rank": [balanced["worst_case_accuracy"], balanced["binary_macro_f1"], balanced["accuracy"]]}
    assert verify_selection(report, protocol(42))["step"] == 0
    report["selected_step"] = 800
    with pytest.raises(ValueError, match="checkpoint policy"):
        verify_selection(report, protocol(42))
    report["history"] = history[:-1]
    with pytest.raises(ValueError, match="complete the frozen"):
        verify_selection(report, protocol(42))


def test_checkpoint_audit_rejects_changed_architecture_and_coordinate_scale():
    frozen = protocol(42)
    state = {"step": 4000, "settings": {**frozen, "seed": 42, "model": "transmil", "coordinate_scale": 32}}
    verify_checkpoint(state, frozen, 42, "transmil", expected_step=4000)
    for key, value in (("transmil_landmarks", 256), ("appearance_seed_offset", 3000), ("coordinate_scale", 64)):
        changed = deepcopy(state)
        changed["settings"][key] = value
        with pytest.raises(ValueError, match="changed"):
            verify_checkpoint(changed, frozen, 42, "transmil", expected_step=4000)


def test_checkpoint_reproduction_rejects_a_hidden_failed_subcase():
    actual = metrics([90, 360, 90, 90, 90])
    recorded = deepcopy(actual)
    verify_metrics(actual, recorded)
    recorded["by_case"]["all_distinct"]["correct"] -= 1
    with pytest.raises(ValueError, match="by_case"):
        verify_metrics(actual, recorded)


def nuisance_data():
    negatives = [case for case in CASE_NAMES if case != "one_pair"]
    cases = np.array([case for index in range(128) for case in ("one_pair", negatives[index % 4])])
    labels = (cases == "one_pair").astype(int)
    return labels, cases, np.repeat(np.arange(128), 2)


def test_paired_bootstrap_identical_models_have_zero_difference_despite_sampling_variation():
    labels, cases, pair_ids = nuisance_data()
    imperfect = labels.copy()
    imperfect[:50] = 1 - imperfect[:50]
    predictions = {f"{model}/{mode}": imperfect for model in ("mean", "vit5", "transmil")
                   for mode in ("selected", "final")}
    result = paired_nuisance_bootstrap(labels, cases, predictions, pair_ids, seed=42, repeats=100)
    assert result["groups"] == 128
    low, high = result["intervals"]["vit5/selected"]["accuracy"]
    assert low < high
    for comparison in result["paired_model_difference_intervals"].values():
        for mode in comparison.values():
            assert all(interval == [0, 0] for interval in mode.values())


def test_paired_bootstrap_preserves_failed_cases_and_rejects_broken_nuisance_pairs():
    labels, cases, pair_ids = nuisance_data()
    failed = labels.copy()
    failed[cases == "all_distinct"] = 1
    predictions = {"vit5/final": labels, "transmil/final": failed}
    result = paired_nuisance_bootstrap(labels, cases, predictions, pair_ids, seed=7, repeats=100)
    assert result["intervals"]["transmil/final"]["worst_case_accuracy"] == [0, 0]
    difference = result["paired_model_difference_intervals"]["transmil_minus_vit5"]["final"]
    assert difference["worst_case_accuracy"] == [-1, -1]
    assert difference["accuracy"][1] < 0
    broken = pair_ids.copy()
    broken[0] = broken[-1]
    with pytest.raises(ValueError, match="opposite-label"):
        paired_nuisance_bootstrap(labels, cases, predictions, broken, seed=7, repeats=100)
