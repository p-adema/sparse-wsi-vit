"""Metrics expose failed repetition cases and both binary classes."""

import json
import hashlib

import numpy as np
import pytest
import torch

from scripts.mondriaan_metrics import (
    CASE_NAMES,
    classification_metrics,
    metrics_from_case_accuracies,
    metrics_from_logits,
)


@pytest.fixture
def full_task():
    cases = np.repeat(CASE_NAMES, [128, 512, 128, 128, 128])
    return (cases == "one_pair").astype(int), cases


def test_always_negative(full_task):
    labels, cases = full_task
    metrics = classification_metrics(labels, np.zeros_like(labels), cases)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_case_accuracy"] == 0.8
    assert metrics["worst_case_accuracy"] == 0
    assert metrics["positive_precision"] == metrics["positive_recall"] == metrics["positive_f1"] == 0
    assert metrics["binary_macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[512, 0], [512, 0]]
    assert metrics["all_case_counts"] == dict(zip(CASE_NAMES, [128, 512, 128, 128, 128]))


def test_always_positive(full_task):
    labels, cases = full_task
    metrics = classification_metrics(labels, np.ones_like(labels), cases)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_case_accuracy"] == 0.2
    assert metrics["worst_case_accuracy"] == 0
    assert metrics["positive_precision"] == 0.5
    assert metrics["positive_recall"] == 1
    assert metrics["positive_f1"] == pytest.approx(2 / 3)
    assert metrics["binary_macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[0, 512], [0, 512]]


def test_systematic_all_distinct_failure(full_task):
    labels, cases = full_task
    predictions = labels.copy()
    predictions[cases == "all_distinct"] = 1
    metrics = classification_metrics(labels, predictions, cases)
    assert metrics["accuracy"] == 0.875
    assert metrics["macro_case_accuracy"] == 0.8
    assert metrics["worst_case_accuracy"] == 0
    assert metrics["positive_precision"] == 0.8
    assert metrics["positive_recall"] == 1
    assert metrics["positive_f1"] == pytest.approx(8 / 9)
    assert metrics["binary_macro_f1"] == pytest.approx((8 / 9 + 6 / 7) / 2)
    assert metrics["confusion_matrix"] == [[384, 128], [0, 512]]


def test_partial_objective_has_explicit_missing_cases():
    metrics = classification_metrics([0, 1], [0, 1], ["all_distinct", "one_pair"])
    assert metrics["observed_worst_case_accuracy"] == 1
    assert metrics["observed_macro_case_accuracy"] == 1
    assert metrics["worst_case_accuracy"] is None
    assert metrics["macro_case_accuracy"] is None
    assert not metrics["selection_eligible"]
    assert metrics["case_coverage"] == 2
    assert metrics["by_case"]["two_pairs"] == {"count": 0, "correct": 0, "accuracy": None}
    json.dumps(metrics, allow_nan=False)
    pair = classification_metrics([0, 1], [0, 1], ["all_distinct", "one_pair"],
                                  intended_cases=("all_distinct", "one_pair"))
    assert pair["selection_eligible"] and pair["worst_case_accuracy"] == 1


def test_historical_summaries_recover_exact_metrics(full_task):
    labels, cases = full_task
    rng = np.random.default_rng(3)
    metrics = classification_metrics(labels, rng.integers(0, 2, len(labels)), cases)
    historical = {case: {"count": entry["count"], "accuracy": entry["accuracy"]}
                  for case, entry in metrics["by_case"].items()}
    assert metrics_from_case_accuracies(historical) == metrics
    historical["all_distinct"]["accuracy"] = 0.123
    with pytest.raises(ValueError, match="integer correct count"):
        metrics_from_case_accuracies(historical)


def test_canvas_subgroups_have_their_own_coverage_and_binary_confusion():
    metrics = classification_metrics(
        [0, 1, 1, 0], [0, 1, 0, 1], ["region_count_0", "region_count_1"] * 2,
        intended_cases=("region_count_0", "region_count_1"),
    )
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["case_coverage"] == 2 and metrics["selection_eligible"]
    assert set(metrics["by_case"]) == {"region_count_0", "region_count_1"}
    assert metrics["worst_case_accuracy"] == metrics["macro_case_accuracy"] == 0.5


def test_companion_audit_excludes_smoke_preserves_sources_and_checks_scores(tmp_path):
    from scripts.audit_mondriaan_metrics import audit_reports

    directory = tmp_path / "mondriaan_patch_training" / "six_aligned"
    directory.mkdir(parents=True)
    by_case = {case: {"count": 4 if case == "one_pair" else 1, "accuracy": 1.0}
               for case in CASE_NAMES}
    metrics = {"accuracy": 1.0, "by_case": by_case}
    report = {"settings": {"model": "vit5", "seed": 42, "train_size": 8, "steps": 2},
              "parameters": 10,
              "history": [{"step": 2, "seconds": 1.0, "train": metrics,
                           "evaluation": metrics}]}
    source = directory / "run.json"
    source.write_text(json.dumps(report))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    smoke = tmp_path / "mondriaan_patch_training" / "smoke"
    smoke.mkdir()
    (smoke / "run.json").write_text(json.dumps(report))
    history, final, primary, sources = audit_reports(tmp_path)
    assert len(sources) == len(primary) == 1
    assert len(history) == len(final) == 2
    assert sources[0]["sha256"] == before == hashlib.sha256(source.read_bytes()).hexdigest()
    assert primary[0]["tp"] == primary[0]["tn"] == 4
    report["history"][0]["evaluation"] = {**metrics, "accuracy": 0.9}
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Saved overall accuracy disagrees"):
        audit_reports(tmp_path)


def test_logits_and_evaluate_preserve_legacy_case_schema(full_task):
    from scripts.probe_mondriaan_vit5 import evaluate

    labels, cases = full_task
    logits = torch.stack([torch.tensor(1 - labels), torch.tensor(labels)], dim=1).float()
    expected = classification_metrics(labels, labels, cases)
    assert metrics_from_logits(logits, torch.tensor(labels), cases) == expected

    class ReadLogits(torch.nn.Module):
        def forward(self, x, coords):
            return {"logits": x[:, 0]}

    data = {"x": logits[:, None], "coords": torch.zeros(len(labels), 1, 2),
            "y": torch.tensor(labels), "cases": cases.tolist(),
            "indices": {case: torch.tensor(np.flatnonzero(cases == case)) for case in CASE_NAMES}}
    metrics = evaluate(ReadLogits(), data, 32)
    assert metrics["confusion_matrix"] == expected["confusion_matrix"]
    assert metrics["worst_case_accuracy"] == metrics["positive_f1"] == 1
    assert metrics["by_case"]["all_distinct"] == {"count": 128, "accuracy": 1.0}


@pytest.mark.parametrize("labels,predictions,cases", [
    ([], [], []), ([0], [2], ["all_distinct"]),
    ([1], [1], ["all_distinct"]), ([0], [0], ["unknown"]),
    ([0, 1], [0], ["all_distinct", "one_pair"]),
])
def test_invalid_metadata_is_rejected(labels, predictions, cases):
    with pytest.raises(ValueError):
        classification_metrics(labels, predictions, cases)
