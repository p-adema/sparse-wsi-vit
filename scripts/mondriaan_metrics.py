"""Case-sensitive metrics for the Mondriaan strict-one-pair task.

Confusion matrices use true labels as rows and predicted labels as columns,
ordered [0, 1]. Undefined precision/recall/F1 is reported as zero. Missing
repetition cases have count=0 and accuracy=None, never a fabricated score.
Full-task worst/macro-case scores are undefined unless all intended cases are
observed; the explicitly named observed-case scores remain available.
"""

from collections import Counter
import math

import numpy as np


CASE_NAMES = ("all_distinct", "one_pair", "two_pairs", "three_matching", "four_matching")
CASE_LABELS = {case: int(case == "one_pair") for case in CASE_NAMES}


def _vector(values, name):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _canonical_cases(by_case, intended_cases):
    intended_cases = tuple(intended_cases)
    if not intended_cases or len(set(intended_cases)) != len(intended_cases):
        raise ValueError("intended_cases must be a nonempty list of distinct cases")
    names = list(CASE_NAMES) if set(intended_cases) <= set(CASE_NAMES) else list(intended_cases)
    names.extend(case for case in by_case if case not in names)
    if any(not isinstance(case, str) or not case for case in names):
        raise ValueError("Case names must be nonempty strings")
    canonical = {}
    for case in names:
        entry = by_case.get(case, {"count": 0, "correct": 0})
        count, correct = entry["count"], entry["correct"]
        if (not isinstance(count, (int, np.integer)) or isinstance(count, bool)
                or not isinstance(correct, (int, np.integer)) or isinstance(correct, bool)
                or count < 0 or not 0 <= correct <= count):
            raise ValueError("Case count/correct must be integers with 0 <= correct <= count")
        canonical[case] = {"count": int(count), "correct": int(correct),
                           "accuracy": float(correct / count) if count else None}
    total = sum(entry["count"] for entry in canonical.values())
    if not total:
        raise ValueError("Cannot evaluate an empty dataset")
    return canonical, intended_cases, total


def _summarize(canonical, intended_cases, total, *, tn, fp, fn, tp):
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    positive_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _ratio(2 * tn, 2 * tn + fp + fn)
    observed = [case for case in canonical if canonical[case]["count"]]
    missing = [case for case in intended_cases if not canonical[case]["count"]]
    intended_scores = [canonical[case]["accuracy"] for case in intended_cases]
    observed_scores = [canonical[case]["accuracy"] for case in observed]
    return {
        "count": total,
        "accuracy": (tp + tn) / total,
        "by_case": canonical,
        "all_case_counts": {case: entry["count"] for case, entry in canonical.items()},
        "all_case_accuracies": {case: entry["accuracy"] for case, entry in canonical.items()},
        "intended_cases": list(intended_cases),
        "evaluated_cases": observed,
        "missing_cases": missing,
        "case_coverage": len(intended_cases) - len(missing),
        "selection_eligible": not missing,
        "worst_case_accuracy": min(intended_scores) if not missing else None,
        "macro_case_accuracy": sum(intended_scores) / len(intended_scores) if not missing else None,
        "observed_worst_case_accuracy": min(observed_scores),
        "observed_macro_case_accuracy": sum(observed_scores) / len(observed_scores),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": positive_f1,
        "binary_macro_f1": (negative_f1 + positive_f1) / 2,
        "by_label": {
            "0": {"count": tn + fp, "precision": _ratio(tn, tn + fn),
                  "recall": _ratio(tn, tn + fp), "f1": negative_f1},
            "1": {"count": tp + fn, "precision": precision, "recall": recall,
                  "f1": positive_f1},
        },
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def metrics_from_case_counts(by_case, *, intended_cases=CASE_NAMES):
    """Compute local-task metrics from {case: {count, correct}} statistics.

    Case labels are fixed by the puzzle rule, so counts determine the complete
    binary confusion matrix. For arbitrary canvas subgroups, use predictions
    with classification_metrics instead; subgroup accuracy alone does not
    generally determine the binary confusion matrix.
    """
    if set(intended_cases) - set(CASE_NAMES) or set(by_case) - set(CASE_NAMES):
        raise ValueError("Unknown repetition case")
    canonical, intended_cases, total = _canonical_cases(by_case, intended_cases)
    positive = canonical["one_pair"]
    tp, fn = positive["correct"], positive["count"] - positive["correct"]
    tn = sum(entry["correct"] for case, entry in canonical.items() if case != "one_pair")
    fp = total - positive["count"] - tn
    return _summarize(canonical, intended_cases, total, tn=tn, fp=fp, fn=fn, tp=tp)


def classification_metrics(labels, predictions, cases, *, intended_cases=CASE_NAMES):
    """Evaluate binary predictions with local cases or explicit canvas subgroups.

    Local repetition cases validate the strict-one-pair ground-truth rule.
    Arbitrary subgroup names require an explicit intended_cases list, which
    defines coverage independently of the five local repetition configurations.
    """
    labels, predictions = _vector(labels, "labels"), _vector(predictions, "predictions")
    cases = _vector(cases, "cases")
    if not len(labels) or len(labels) != len(predictions) or len(labels) != len(cases):
        raise ValueError("labels, predictions, and cases must have equal nonzero length")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("labels and predictions must contain binary hard labels")
    local_cases = set(cases.tolist()) <= set(CASE_NAMES)
    if not local_cases and set(intended_cases) <= set(CASE_NAMES):
        raise ValueError("Unknown repetition case")
    expected = np.array([CASE_LABELS[case] for case in cases]) if local_cases else None
    if local_cases and not np.array_equal(labels, expected):
        raise ValueError("Labels disagree with the strict-one-pair repetition cases")
    correct = labels == predictions
    counts = Counter(cases.tolist())
    canonical, intended_cases, total = _canonical_cases({
        case: {"count": count, "correct": int(correct[cases == case].sum())}
        for case, count in counts.items()
    }, intended_cases)
    return _summarize(canonical, intended_cases, total,
                      tn=int(((labels == 0) & (predictions == 0)).sum()),
                      fp=int(((labels == 0) & (predictions == 1)).sum()),
                      fn=int(((labels == 1) & (predictions == 0)).sum()),
                      tp=int(((labels == 1) & (predictions == 1)).sum()))


def metrics_from_logits(logits, labels, cases, *, intended_cases=CASE_NAMES):
    """Convenience wrapper for a finite (N, 2) logits array or CPU/GPU tensor."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite (N, 2) matrix")
    return classification_metrics(labels, logits.argmax(axis=1), cases,
                                  intended_cases=intended_cases)


def metrics_from_case_accuracies(by_case, *, intended_cases=CASE_NAMES):
    """Recover exact counts from historical count/accuracy summaries.

    This needs no saved predictions or checkpoint evaluation: every case has
    one binary ground-truth label. Reject summaries that cannot encode integer
    counts within float32 rounding tolerance, rather than silently approximate.
    """
    recovered = {}
    for case, entry in by_case.items():
        count, accuracy = entry["count"], entry["accuracy"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Historical case counts must be nonnegative integers")
        if not count:
            recovered[case] = {"count": 0, "correct": 0}
            continue
        if accuracy is None or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Historical observed case accuracies must be finite in [0, 1]")
        raw = count * accuracy
        correct = round(raw)
        if not math.isclose(raw, correct, rel_tol=0, abs_tol=max(1e-5, count * 1e-7)):
            raise ValueError("Historical accuracy does not correspond to an integer correct count")
        recovered[case] = {"count": count, "correct": correct}
    return metrics_from_case_counts(recovered, intended_cases=intended_cases)
