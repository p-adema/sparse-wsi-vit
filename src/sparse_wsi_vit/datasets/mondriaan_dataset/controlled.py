"""New, deterministic Mondriaan controls; historical generators are untouched.

Object identities are four-colour *arrangements*, represented by a 4x4 array
of palette indices. Every identity has four pixels of each colour. Distances
are pixel Hamming distances (0--16), not distances between sampled RGB values.
All functions use private NumPy generators and do not change global RNG state.
Dataset arrays are NumPy; training callers can convert them to torch tensors.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


FIXED_PALETTE = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1]],
                         dtype=np.float32)
VARIANT_COUNTS = {"quadrant24": (1, 1, 1, 1), "cells8": (2, 2, 2, 2),
                  "pixels16": (4, 4, 4, 4)}


@lru_cache(maxsize=None)
def _multinomial(counts: tuple[int, ...]) -> int:
    return math.factorial(sum(counts)) // math.prod(math.factorial(c) for c in counts)


def vocabulary_size(variant: str) -> int:
    """24, 2,520 or 63,063,000; no large vocabulary is materialised."""
    return _multinomial(VARIANT_COUNTS[variant])


def pattern_from_id(pattern_id: int, variant: str = "pixels16") -> np.ndarray:
    """Unrank a lexicographic balanced multiset permutation as a 4x4 pattern."""
    counts = list(VARIANT_COUNTS[variant])
    if int(pattern_id) != pattern_id or not 0 <= pattern_id < _multinomial(tuple(counts)):
        raise ValueError("Pattern ID outside the vocabulary")
    cells = []
    rank = int(pattern_id)
    for _ in range(sum(counts)):
        for colour in range(4):
            if not counts[colour]:
                continue
            counts[colour] -= 1
            completions = _multinomial(tuple(counts))
            if rank < completions:
                cells.append(colour)
                break
            rank -= completions
            counts[colour] += 1
    array = np.asarray(cells, dtype=np.uint8)
    if variant == "quadrant24":
        return array.reshape(2, 2).repeat(2, axis=0).repeat(2, axis=1)
    if variant == "cells8":
        return array.reshape(2, 4).repeat(2, axis=0)
    return array.reshape(4, 4)


def pattern_id(pattern: np.ndarray, variant: str = "pixels16") -> int:
    """Inverse of :func:`pattern_from_id`, with structural validation."""
    array = np.asarray(pattern)
    if array.shape != (4, 4) or not np.isin(array, range(4)).all():
        raise ValueError("Pattern must be 4x4 with colour indices 0--3")
    if variant == "quadrant24":
        cells = array[::2, ::2]
        expanded = cells.repeat(2, axis=0).repeat(2, axis=1)
    elif variant == "cells8":
        cells = array[::2]
        expanded = cells.repeat(2, axis=0)
    elif variant == "pixels16":
        cells, expanded = array, array
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if not np.array_equal(array, expanded):
        raise ValueError(f"Pattern does not have {variant} cell structure")
    counts = list(VARIANT_COUNTS[variant])
    if not np.array_equal(np.bincount(cells.ravel().astype(int), minlength=4), counts):
        raise ValueError("Pattern has an invalid colour histogram")
    rank = 0
    for colour in cells.ravel():
        colour = int(colour)
        for smaller in range(colour):
            if counts[smaller]:
                counts[smaller] -= 1
                rank += _multinomial(tuple(counts))
                counts[smaller] += 1
        counts[colour] -= 1
    return rank


@dataclass(frozen=True)
class AppearancePool:
    variant: str
    split: str
    ids: np.ndarray
    patterns: np.ndarray
    min_distance: int = 0

    def __len__(self) -> int:
        return len(self.ids)


def distance_diagnostics(patterns: np.ndarray, *, max_pairs: int = 100_000,
                         seed: int = 0) -> dict:
    """Bounded pairwise pixel-distance audit; state whether it is exhaustive."""
    flat = np.asarray(patterns).reshape(len(patterns), 16)
    n = len(flat)
    total = n * (n - 1) // 2
    if not total:
        return {"pairs": 0, "total_pairs": total, "exhaustive": True,
                "minimum": None, "mean": None, "maximum": None, "histogram": {}}
    rng = np.random.default_rng(seed)
    if total <= max_pairs:
        distances = np.concatenate([(flat[i + 1:] != flat[i]).sum(axis=1)
                                    for i in range(n - 1)])
    else:
        # Uniform unordered pairs, sampled with replacement; bounded memory.
        first = rng.integers(n, size=max_pairs)
        second = rng.integers(n - 1, size=max_pairs)
        second += second >= first
        distances = (flat[first] != flat[second]).sum(axis=1)
    return {"pairs": len(distances), "total_pairs": total,
            "exhaustive": total <= max_pairs, "minimum": int(distances.min()),
            "mean": float(distances.mean()), "maximum": int(distances.max()),
            "histogram": dict(sorted(Counter(map(int, distances)).items()))}


def make_appearance_pools(variant: str = "pixels16", *,
                          sizes: Mapping[str, int] | None = None, seed: int = 42,
                          min_distance: int = 0, disjoint: bool = True,
                          max_attempts: int = 100_000) -> dict[str, AppearancePool]:
    """Explicit train/validation/test arrangement pools.

    With ``disjoint=True`` a minimum distance applies across the *union* of
    pools. With ``disjoint=False`` every split receives the same dictionary;
    equal sizes are required. This is the explicit known-vocabulary control.
    A distance-restricted pool is a greedy sampled codebook, not a uniform
    sample of all feasible codebooks. Impossible/overly stringent requests
    fail with a bounded error rather than silently relaxing the constraint.
    """
    if sizes is None:
        sizes = ({"train": 12, "validation": 6, "test": 6} if variant == "quadrant24"
                 else {"train": 256, "validation": 64, "test": 64})
    sizes = dict(sizes)
    if not sizes or any(int(n) != n or n <= 0 for n in sizes.values()):
        raise ValueError("Pool sizes must be positive integers")
    if not 0 <= min_distance <= 16:
        raise ValueError("Minimum pixel distance must be between 0 and 16")
    if not disjoint and len(set(sizes.values())) != 1:
        raise ValueError("Shared appearance pools must have equal sizes")
    wanted = sum(sizes.values()) if disjoint else next(iter(sizes.values()))
    total = vocabulary_size(variant)
    if wanted > total:
        raise ValueError(f"Requested {wanted} distinct appearances from {total}")
    rng = np.random.default_rng(seed)
    if min_distance <= 1:
        selected = rng.choice(total, size=wanted, replace=False)
        patterns = np.stack([pattern_from_id(int(i), variant) for i in selected])
    else:
        selected_list, pattern_list, seen = [], [], set()
        for _ in range(max_attempts):
            candidate_id = int(rng.integers(total))
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidate = pattern_from_id(candidate_id, variant)
            if any(np.count_nonzero(candidate != p) < min_distance for p in pattern_list):
                continue
            selected_list.append(candidate_id)
            pattern_list.append(candidate)
            if len(selected_list) == wanted:
                break
        if len(selected_list) != wanted:
            raise ValueError(f"Could build only {len(selected_list)}/{wanted} appearances "
                             f"at minimum distance {min_distance} in {max_attempts} attempts")
        selected = np.asarray(selected_list, dtype=np.int64)
        patterns = np.stack(pattern_list)
    pools, cursor = {}, 0
    for split, size in sizes.items():
        indices = slice(cursor, cursor + size) if disjoint else slice(0, size)
        ids, local_patterns = selected[indices].copy(), patterns[indices].copy()
        ids.flags.writeable = local_patterns.flags.writeable = False
        pools[split] = AppearancePool(variant, split, ids, local_patterns, min_distance)
        cursor += size
    return pools


def strict_pair_label(identities: Sequence) -> int:
    """Exactly one identity occurs twice and every other identity occurs once."""
    counts = list(Counter(identities).values())
    return int(counts.count(2) == 1 and all(c in (1, 2) for c in counts))


def case_multiplicities(n: int) -> dict[str, tuple[int, ...]]:
    """Canonical case families generalized to n objects (n >= 3).

    At n=4 these are exactly AABC/ABCD/AABB/AAAB/AAAA. At larger n the
    two-pair and triple cases have additional singletons. These representative
    families do not exhaust every integer partition when n > 4.
    """
    if n < 3:
        raise ValueError("At least three objects are required")
    cases = {"one_pair": (2,) + (1,) * (n - 2),
             "all_distinct": (1,) * n}
    if n >= 4:
        cases["two_pairs"] = (2, 2) + (1,) * (n - 4)
        cases["triple"] = (3,) + (1,) * (n - 3)
    cases["all_matching"] = (n,)
    return cases


def sample_positions(rng: np.random.Generator, n: int, *, image_size: int = 32,
                     aligned: bool = True, min_sep: float = 8) -> list[tuple[int, int]]:
    """Non-overlapping corners; Euclidean separation and complete footprints."""
    if image_size < 4 or min_sep < 0:
        raise ValueError("Invalid image size or minimum separation")
    step = 4 if aligned else 1
    candidates = np.array([(y, x) for y in range(0, image_size - 3, step)
                           for x in range(0, image_size - 3, step)], dtype=np.int32)
    for _ in range(100):
        chosen = []
        for index in rng.permutation(len(candidates)):
            y, x = map(int, candidates[index])
            if any((abs(y - py) < 4 and abs(x - px) < 4)
                   or (y - py) ** 2 + (x - px) ** 2 < min_sep ** 2 for py, px in chosen):
                continue
            chosen.append((y, x))
            if len(chosen) == n:
                return chosen
    raise ValueError(f"Could not place {n} separated objects in {image_size}x{image_size}")


def shared_palette(rng: np.random.Generator, mode: str = "fixed") -> np.ndarray:
    if mode == "fixed":
        return FIXED_PALETTE.copy()
    if mode != "shared":
        raise ValueError("Palette mode must be 'fixed' or 'shared'")
    for _ in range(10_000):
        palette = rng.uniform(0.1, 0.9, size=(4, 3)).astype(np.float32)
        if min(np.linalg.norm(palette[i] - palette[j])
               for i in range(4) for j in range(i)) >= 0.2:
            return palette
    raise RuntimeError("Could not sample separated shared palette colours")


def render_scene(patterns: np.ndarray, positions: Sequence[tuple[int, int]],
                 palette: np.ndarray = FIXED_PALETTE, *, image_size: int = 32) -> np.ndarray:
    """Render explicit arrangements; palette is shared by *all* scene objects."""
    palette = np.asarray(palette, dtype=np.float32)
    if (palette.shape != (4, 3) or not np.isfinite(palette).all()
            or np.any(palette < 0) or np.any(palette > 1)
            or len({p.tobytes() for p in palette}) != 4):
        raise ValueError("Palette must contain four distinct RGB colours")
    if np.any(np.all(palette == 0, axis=1)):
        raise ValueError("Black is reserved for background")
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    occupied = np.zeros((image_size, image_size), dtype=bool)
    for pattern, (y, x) in zip(patterns, positions, strict=True):
        if not (0 <= y <= image_size - 4 and 0 <= x <= image_size - 4):
            raise ValueError("Object extends beyond image")
        if occupied[y:y + 4, x:x + 4].any():
            raise ValueError("Objects overlap")
        pattern = np.asarray(pattern)
        if (pattern.shape != (4, 4) or not np.isin(pattern, range(4)).all()
                or not np.issubdtype(pattern.dtype, np.integer) or not np.array_equal(
                np.bincount(pattern.ravel().astype(int), minlength=4), [4] * 4)):
            raise ValueError("Every object must contain four pixels per colour")
        image[y:y + 4, x:x + 4] = palette[pattern]
        occupied[y:y + 4, x:x + 4] = True
    return image


def patchify(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full row-major 4x4 grid; all-black patches use the historical -1 sentinel."""
    height, width, channels = image.shape
    if height % 4 or width % 4 or channels != 3:
        raise ValueError("Image dimensions must be multiples of four with RGB channels")
    tokens = image.reshape(height // 4, 4, width // 4, 4, 3).transpose(0, 2, 1, 3, 4)
    tokens = tokens.reshape(-1, 48).copy()
    tokens[np.all(tokens == 0, axis=1)] = -1
    coords = np.array([(x, y) for y in range(0, height, 4) for x in range(0, width, 4)],
                      dtype=np.float32)
    return tokens, coords


def unpatchify(tokens: np.ndarray, image_size: int = 32) -> np.ndarray:
    tokens = np.asarray(tokens).copy()
    if tokens.shape != ((image_size // 4) ** 2, 48) or image_size % 4:
        raise ValueError("Tokens do not represent a complete square patch grid")
    tokens[np.all(tokens == -1, axis=1)] = 0
    return tokens.reshape(image_size // 4, image_size // 4, 4, 4, 3).transpose(
        0, 2, 1, 3, 4).reshape(image_size, image_size, 3)


def make_controlled_dataset(pool: AppearancePool, size: int, *, seed: int = 42,
                            object_counts: Sequence[int] = (4,), aligned: bool = True,
                            palette_mode: str = "fixed", image_size: int = 32,
                            min_sep: float = 8, split: str | None = None,
                            shuffle: bool = True) -> dict:
    """Generate exactly label-balanced object counts, layouts, and palettes.

    Each layout/palette/count combination appears once with each binary label.
    Negative case families are cycled separately within each object count. The
    resulting size must be divisible by twice the number of object counts.
    Identities and their assignment to positions are sampled independently.
    """
    if any(int(n) != n for n in object_counts):
        raise ValueError("Object counts must be integers")
    object_counts = tuple(map(int, object_counts))
    if not object_counts or len(set(object_counts)) != len(object_counts):
        raise ValueError("Object counts must be a nonempty distinct sequence")
    if size <= 0 or size % (2 * len(object_counts)):
        raise ValueError("Size must be positive and divisible by 2 * number of object counts")
    if image_size % 4:
        raise ValueError("Image size must be divisible by four")
    if len(pool) < max(object_counts):
        raise ValueError("Appearance pool too small for an all-distinct scene")
    rng = np.random.default_rng(seed)
    rows, image_hashes = [], set()
    for count in object_counts:
        cases = case_multiplicities(count)
        negatives = [name for name in cases if name != "one_pair"]
        for pair_index in range(size // (2 * len(object_counts))):
            # Reject an accidental duplicate sample while preserving paired nuisances.
            for _ in range(100):
                positions = sample_positions(rng, count, image_size=image_size,
                                             aligned=aligned, min_sep=min_sep)
                palette = shared_palette(rng, palette_mode)
                pair = []
                for label, case in [(1, "one_pair"),
                                    (0, negatives[pair_index % len(negatives)])]:
                    multiplicities = cases[case]
                    selected = rng.choice(len(pool), len(multiplicities), replace=False)
                    indices = np.repeat(selected, multiplicities)
                    rng.shuffle(indices)
                    ids, patterns = pool.ids[indices], pool.patterns[indices]
                    image = render_scene(patterns, positions, palette, image_size=image_size)
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in image_hashes:
                        break
                    x, coords = patchify(image)
                    pair.append({"x": x, "coords": coords, "y": label, "case": case,
                                 "image": image, "positions": list(positions),
                                 "object_types": ids.tolist(), "patterns": patterns.copy(),
                                 "palette": palette.copy(), "object_count": count,
                                 "image_hash": digest, "pair_id": len(rows) // 2})
                if len(pair) == 2:
                    rows.extend(pair)
                    image_hashes.update(row["image_hash"] for row in pair)
                    break
            else:
                raise RuntimeError("Could not generate distinct image pairs")
    if shuffle:
        rows = [rows[i] for i in rng.permutation(len(rows))]
    return {"x": np.stack([r["x"] for r in rows]),
            "coords": np.stack([r["coords"] for r in rows]),
            "y": np.array([r["y"] for r in rows], dtype=np.int64),
            "cases": [r["case"] for r in rows], "images": np.stack([r["image"] for r in rows]),
            "positions": [r["positions"] for r in rows],
            "object_types": [r["object_types"] for r in rows],
            "patterns": [r["patterns"] for r in rows],
            "palettes": np.stack([r["palette"] for r in rows]),
            "object_counts": np.array([r["object_count"] for r in rows]),
            "image_hashes": [r["image_hash"] for r in rows],
            "pair_ids": np.array([r["pair_id"] for r in rows]),
            "split": split or pool.split, "variant": pool.variant,
            "palette_mode": palette_mode, "aligned": aligned,
            "image_size": image_size, "min_sep": min_sep,
            "appearance_pool_ids": pool.ids.tolist()}


def complete_object_view(data: dict) -> dict:
    """Diagnostic complete object crops; padding mask is mandatory for variable n.

    Object order matches generation metadata, not spatially sorted order. Exact
    positions remain available. Padding coordinates are zero; masked -1 tokens
    must not participate in a model's pooling/attention operation.
    """
    lengths = np.asarray(data["object_counts"], dtype=np.int64)
    x = np.full((len(lengths), int(lengths.max()), 48), -1, dtype=np.float32)
    coords = np.zeros((len(lengths), int(lengths.max()), 2), dtype=np.float32)
    mask = np.arange(int(lengths.max()))[None] < lengths[:, None]
    for i, (image, positions) in enumerate(zip(data["images"], data["positions"], strict=True)):
        for j, (y, xx) in enumerate(positions):
            x[i, j] = image[y:y + 4, xx:xx + 4].reshape(48)
            coords[i, j] = [xx, y]
    return {**data, "x": x, "coords": coords, "lengths": lengths, "mask": mask,
            "view": "complete_objects"}


def dataset_checks(data: dict) -> dict:
    """Verify actual pixels, lossless patches, strict labels and exact balance."""
    layouts = {0: Counter(), 1: Counter()}
    counts = {0: Counter(), 1: Counter()}
    palettes = {0: Counter(), 1: Counter()}
    distances, phases = [], Counter()
    for i, image in enumerate(data["images"]):
        label = int(data["y"][i])
        positions, ids = data["positions"][i], data["object_types"][i]
        if not np.array_equal(unpatchify(data["x"][i], data["image_size"]), image):
            raise ValueError("Patchification lost pixels")
        expected_coords = patchify(image)[1]
        if not np.array_equal(data["coords"][i], expected_coords):
            raise ValueError("Patch coordinates are not the complete row-major grid")
        objects = [image[y:y + 4, x:x + 4] for y, x in positions]
        rendered_keys = [o.tobytes() for o in objects]
        if strict_pair_label(rendered_keys) != label or strict_pair_label(ids) != label:
            raise ValueError("Rendered strict-pair label is incorrect")
        if sorted(Counter(rendered_keys).values()) != sorted(Counter(ids).values()):
            raise ValueError("Appearance metadata does not match rendered objects")
        palette = data["palettes"][i]
        # Float64 sums of these sixteen float32 values are exact: float32
        # reduction order can otherwise differ by a few ULPs across patterns.
        expected_mean = palette.astype(np.float64).mean(axis=0)
        for obj, pattern, identity in zip(objects, data["patterns"][i], ids, strict=True):
            if (not np.array_equal(obj, palette[pattern])
                    or pattern_id(pattern, data["variant"]) != identity):
                raise ValueError("Canonical pattern metadata disagrees with rendered pixels")
            if any(int(np.all(obj == colour, axis=-1).sum()) != 4 for colour in palette):
                raise ValueError("Unequal within-image object histograms")
            np.testing.assert_array_equal(obj.astype(np.float64).mean(axis=(0, 1)), expected_mean)
        mask = np.any(image != 0, axis=-1)
        if int(mask.sum()) != 16 * len(objects):
            raise ValueError("Foreground amount disagrees with object count")
        layouts[label][mask.tobytes()] += 1
        counts[label][len(objects)] += 1
        palettes[label][palette.tobytes()] += 1
        distinct_patterns = np.unique(data["patterns"][i].reshape(-1, 16), axis=0)
        for j, p in enumerate(distinct_patterns):
            distances.extend((distinct_patterns[j + 1:] != p).sum(axis=1).tolist())
        for j, (y, x) in enumerate(positions):
            phases[(y % 4, x % 4)] += 1
            if any((y - py) ** 2 + (x - px) ** 2 < data["min_sep"] ** 2
                   for py, px in positions[:j]):
                raise ValueError("Corner separation violated")
    if layouts[0] != layouts[1] or counts[0] != counts[1] or palettes[0] != palettes[1]:
        raise ValueError("Layouts, counts or palettes reveal the label")
    observed = {identity for ids in data["object_types"] for identity in ids}
    if not observed.issubset(set(data["appearance_pool_ids"])):
        raise ValueError("Sample uses an appearance outside its pool")
    return {"samples": len(data["y"]), "shape": list(data["x"].shape),
            "case_counts": dict(Counter(data["cases"])),
            "label_counts": dict(sorted(Counter(map(int, data["y"])).items())),
            "object_count_by_label": {str(k): dict(v) for k, v in counts.items()},
            "split": data["split"], "variant": data["variant"],
            "observed_appearances": len(observed), "pool_appearances": len(data["appearance_pool_ids"]),
            "pixel_reconstruction_exact": True, "pixel_label_oracle_accuracy": 1.0,
            "layouts_palettes_counts_exactly_label_balanced": True,
            "equal_within_image_object_histograms_and_means": True,
            "unique_images": len(set(data["image_hashes"])),
            "corner_phase_counts": {f"{y},{x}": n for (y, x), n in sorted(phases.items())},
            "distinct_object_distance": {"minimum": min(distances) if distances else None,
                                         "mean": float(np.mean(distances)) if distances else None}}


def audit_splits(datasets: Mapping[str, dict], *, require_disjoint_appearances: bool = True) -> dict:
    """Reject duplicate images and (optionally) any shared canonical arrangements."""
    result = {}
    for i, (left_name, left) in enumerate(datasets.items()):
        for right_name, right in list(datasets.items())[i + 1:]:
            # Hash actual pixels again; do not trust cached metadata.
            left_images = {hashlib.sha256(im.tobytes()).hexdigest() for im in left["images"]}
            right_images = {hashlib.sha256(im.tobytes()).hexdigest() for im in right["images"]}
            images = len(left_images & right_images)
            left_patterns = {p.tobytes() for ps in left["patterns"] for p in ps}
            right_patterns = {p.tobytes() for ps in right["patterns"] for p in ps}
            appearances = len(left_patterns & right_patterns)
            pool_ids = (len(set(left["appearance_pool_ids"]) & set(right["appearance_pool_ids"]))
                        if left["variant"] == right["variant"] else None)
            result[f"{left_name}:{right_name}"] = {"identical_images": images,
                                                   "observed_appearance_overlap": appearances,
                                                   "pool_id_overlap": pool_ids}
            if images:
                raise ValueError(f"Identical images shared by {left_name} and {right_name}")
            if require_disjoint_appearances and (appearances or pool_ids):
                raise ValueError(f"Appearance pools overlap between {left_name} and {right_name}")
    return result
