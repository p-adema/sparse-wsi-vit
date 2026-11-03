"""Train-defined, nuisance-paired ambiguous-subcase challenges.

This optional diagnostic leaves the primary five-case generator unchanged.
Complete-object grouping is supplied to calculate diversity. Matching score
bins does not match continuous scores within bins, and removes/reshapes cases.
No policy may be fitted from validation/test images. All-matching negatives
cannot be included: they uniquely have zero diversity while positives do not.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np

from .controlled import (
    AppearancePool,
    case_multiplicities,
    dataset_checks,
    make_controlled_dataset,
)


def diversity_scores(data: dict, statistic: str = "hamming") -> np.ndarray:
    if statistic not in ("hamming", "rgb", "normalized_rgb"):
        raise ValueError("Statistic must be hamming, rgb or normalized_rgb")
    scores = []
    for patterns, palette in zip(data["patterns"], data["palettes"], strict=True):
        if statistic == "hamming":
            value = np.mean([np.mean(patterns[i] != patterns[j])
                             for i in range(len(patterns)) for j in range(i)])
        else:
            pixels = palette[patterns].astype(np.float64)
            value = np.mean([np.square(pixels[i] - pixels[j]).sum(axis=-1).mean()
                             for i in range(len(patterns)) for j in range(i)])
            if statistic == "normalized_rgb":
                colours = palette.astype(np.float64)
                value /= 2 * np.square(colours - colours.mean(axis=0)).sum(axis=1).mean()
        scores.append(value)
    return np.asarray(scores)


@dataclass(frozen=True)
class MatchingPolicy:
    statistic: str
    # Immutable count-specific inner quantile boundaries.
    boundaries: tuple[tuple[int, tuple[float, ...]], ...]
    training_fingerprint: str
    requested_bins: int

    def score_bin(self, value: float, object_count: int) -> int:
        if not np.isfinite(value) or value < 0:
            raise ValueError("Diversity score must be finite and nonnegative")
        edges = dict(self.boundaries).get(int(object_count))
        if edges is None:
            raise ValueError("Policy was not fitted for this object count")
        # Keep the zero atom separate, however wide the first quantile bin.
        return 0 if value == 0 else 1 + int(np.searchsorted(edges, value, side="right"))

    def as_dict(self) -> dict:
        return {"statistic": self.statistic, "boundaries": dict(self.boundaries),
                "training_fingerprint": self.training_fingerprint,
                "requested_bins": self.requested_bins, "source_split": "train",
                "zero_atom_separate": True}


def fit_matching_policy(training: dict, *, statistic: str = "hamming", bins: int = 8) -> MatchingPolicy:
    """Freeze bins using only positives from the original training generator."""
    if training.get("split") != "train" or training.get("diagnostic"):
        raise ValueError("Policy must be fitted on the primary training split only")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("At least two score bins are required")
    dataset_checks(training)
    scores = diversity_scores(training, statistic)
    boundaries = []
    for count in sorted(set(map(int, training["object_counts"]))):
        selection = (training["object_counts"] == count) & (training["y"] == 1)
        edges = np.unique(np.quantile(scores[selection], np.arange(1, bins) / bins))
        boundaries.append((count, tuple(map(float, edges))))
    hashes = sorted(hashlib.sha256(im.tobytes()).hexdigest() for im in training["images"])
    fingerprint = hashlib.sha256("".join(hashes).encode()).hexdigest()
    return MatchingPolicy(statistic, tuple(boundaries), fingerprint, bins)


def build_matched_subcase_challenge(
    pool: AppearancePool, policy: MatchingPolicy, size: int, *, seed: int = 42,
    negative_cases: tuple[str, ...] = ("all_distinct",), object_counts: tuple[int, ...] = (4,),
    aligned: bool = True, palette_mode: str = "fixed", image_size: int = 32,
    min_sep: float = 8, proposal_pairs: int = 256, max_rounds: int = 100,
) -> dict:
    """Generate fresh pairs; retain the entire pair iff frozen score bins agree.

    Size must divide 2 * number of counts * number of negative case families.
    Each count/family stratum contributes the same number of opposite-label
    pairs. Bounded rejection may fail, and never relaxes bins or quotas. The
    supplied pool should be held out from the policy's training appearances.
    """
    if (not object_counts or len(set(object_counts)) != len(object_counts)
            or not negative_cases or len(set(negative_cases)) != len(negative_cases)):
        raise ValueError("Counts and negative case families must be nonempty and distinct")
    if "all_matching" in negative_cases:
        raise ValueError("All-matching zero-diversity negatives cannot match positive scores")
    if size <= 0 or size % (2 * len(object_counts) * len(negative_cases)):
        raise ValueError("Size must be divisible by twice the number of count/case strata")
    if proposal_pairs < 1 or max_rounds < 1:
        raise ValueError("Rejection budget must be positive")
    for count in object_counts:
        policy.score_bin(1, count)
        allowed = set(case_multiplicities(count)) - {"one_pair", "all_matching"}
        if not set(negative_cases) <= allowed:
            raise ValueError(f"Requested negative families unavailable for {count} objects")
    quota = size // (2 * len(object_counts) * len(negative_cases))
    wanted = {(count, case): quota for count in object_counts for case in negative_cases}
    accepted, candidates = Counter(), Counter()
    rejected_bins, rejected_duplicates, skipped_quota = Counter(), Counter(), Counter()
    rows, used_images, proposed_pairs = [], set(), 0
    rng = np.random.default_rng(seed)
    base = None
    for round_index in range(max_rounds):
        bank = make_controlled_dataset(pool, 2 * proposal_pairs * len(object_counts),
                                       seed=int(rng.integers(0, 2**32)), object_counts=object_counts,
                                       aligned=aligned, palette_mode=palette_mode,
                                       image_size=image_size, min_sep=min_sep)
        base = bank
        scores = diversity_scores(bank, policy.statistic)
        positive = {int(pair): i for i, pair in enumerate(bank["pair_ids"]) if bank["y"][i] == 1}
        proposed_pairs += len(positive)
        for negative_index, case in enumerate(bank["cases"]):
            if bank["y"][negative_index] != 0 or case not in negative_cases:
                continue
            count = int(bank["object_counts"][negative_index])
            stratum = (count, case)
            candidates[stratum] += 1
            if accepted[stratum] >= wanted[stratum]:
                skipped_quota[stratum] += 1
                continue
            positive_index = positive[int(bank["pair_ids"][negative_index])]
            pair = (positive_index, negative_index)
            bins = [policy.score_bin(float(scores[i]), count) for i in pair]
            if bins[0] != bins[1]:
                rejected_bins[stratum] += 1
                continue
            if any(bank["image_hashes"][i] in used_images for i in pair):
                rejected_duplicates[stratum] += 1
                continue
            pair_id = len(rows) // 2
            for index in pair:
                rows.append({"x": bank["x"][index], "coords": bank["coords"][index],
                             "y": int(bank["y"][index]), "case": bank["cases"][index],
                             "image": bank["images"][index], "positions": bank["positions"][index],
                             "object_types": bank["object_types"][index], "patterns": bank["patterns"][index],
                             "palette": bank["palettes"][index], "object_count": count,
                             "image_hash": bank["image_hashes"][index], "pair_id": pair_id,
                             "score": float(scores[index]), "bin": bins[0], "negative_case": case})
            used_images.update(bank["image_hashes"][i] for i in pair)
            accepted[stratum] += 1
        if all(accepted[key] == target for key, target in wanted.items()):
            break
    else:
        raise ValueError(f"Matched-subcase rejection budget exhausted: {dict(accepted)} of {wanted}; "
                         "policy unchanged, no partial challenge returned")
    rows = [rows[i] for i in rng.permutation(len(rows))]
    # Keep scalar configuration/provenance metadata; replace every row-dependent field.
    result = {k: v for k, v in base.items() if k not in (
        "x", "coords", "y", "cases", "images", "positions", "object_types", "patterns",
        "palettes", "object_counts", "image_hashes", "pair_ids")}
    result.update({"x": np.stack([r["x"] for r in rows]), "coords": np.stack([r["coords"] for r in rows]),
                   "y": np.array([r["y"] for r in rows], dtype=np.int64),
                   "cases": [r["case"] for r in rows], "images": np.stack([r["image"] for r in rows]),
                   "positions": [r["positions"] for r in rows],
                   "object_types": [r["object_types"] for r in rows], "patterns": [r["patterns"] for r in rows],
                   "palettes": np.stack([r["palette"] for r in rows]),
                   "object_counts": np.array([r["object_count"] for r in rows]),
                   "image_hashes": [r["image_hash"] for r in rows], "pair_ids": np.array([r["pair_id"] for r in rows]),
                   "matching_scores": np.array([r["score"] for r in rows]),
                   "matching_bins": np.array([r["bin"] for r in rows]),
                   "comparison_negative_cases": [r["negative_case"] for r in rows],
                   "diagnostic": "train_bin_matched_subcases", "matching_policy": policy.as_dict(),
                   "matching_audit": {"rounds": round_index + 1, "proposed_pairs": proposed_pairs,
                                      "retained_pairs": len(rows) // 2,
                                      "candidate_pairs_by_stratum": {f"n{n}/{c}": v for (n, c), v in candidates.items()},
                                      "retained_pairs_by_stratum": {f"n{n}/{c}": v for (n, c), v in accepted.items()},
                                      "rejected_bin_mismatch_by_stratum": {f"n{n}/{c}": v for (n, c), v in rejected_bins.items()},
                                      "rejected_duplicate_images_by_stratum": {f"n{n}/{c}": v for (n, c), v in rejected_duplicates.items()},
                                      "skipped_completed_quota_by_stratum": {f"n{n}/{c}": v for (n, c), v in skipped_quota.items()},
                                      "nonrequested_case_pairs": proposed_pairs - sum(candidates.values()),
                                      "within_bin_continuous_scores_matched": False}})
    challenge_checks(result, policy)
    return result


def challenge_checks(data: dict, policy: MatchingPolicy) -> dict:
    checks = dataset_checks(data)
    scores = diversity_scores(data, policy.statistic)
    np.testing.assert_array_equal(scores, data["matching_scores"])
    groups = {}
    for i, pair in enumerate(data["pair_ids"]):
        groups.setdefault(int(pair), []).append(i)
    for indices in groups.values():
        if len(indices) != 2 or sorted(data["y"][indices]) != [0, 1]:
            raise ValueError("Challenge must retain complete opposite-label pairs")
        first, second = indices
        if (data["positions"][first] != data["positions"][second]
                or not np.array_equal(data["palettes"][first], data["palettes"][second])
                or data["object_counts"][first] != data["object_counts"][second]):
            raise ValueError("Paired nuisances changed")
        bins = [policy.score_bin(float(scores[i]), int(data["object_counts"][i])) for i in indices]
        if bins[0] != bins[1] or bins[0] == 0:
            raise ValueError("Opposite-label pairs have different score bins")
    checks.update({"same_frozen_score_bin_per_pair": True, "whole_nuisance_pairs_retained": True,
                   "source_policy": policy.as_dict(), "diagnostic_only": True})
    return checks
