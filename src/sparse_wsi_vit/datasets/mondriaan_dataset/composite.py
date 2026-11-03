"""Balanced, split-safe compositions of four-object Mondriaan scenes.

Primary control: OR of local strict-one-pair labels. Local scene membership is
the label-independent 32-pixel lattice with a four-pixel gutter, visible through
spatial coordinates, not an extra feature. Object identities may recur across
regions. One palette is shared across the whole canvas. A hierarchical model
that reads each region then pools local labels is a valid solution; this task
does not require full-canvas pairwise attention.

Local pools must already be split. Each local layout must offer both labels;
opposite canvas labels then use exactly the same foreground layout and palette.
Counts are sampled evenly over the counts allowed by the requested canvas class,
then positive slots uniformly conditional on that count. In particular OR uses
zero positives for negatives and a uniform count from 1..M for positives, not
independent fair local labels (which would make OR highly imbalanced).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping

import numpy as np


CASES = {(1, 1, 1, 1): "all_distinct", (1, 1, 2): "one_pair",
         (2, 2): "two_pairs", (1, 3): "three_matching", (4,): "four_matching"}
FIXED_PALETTE = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1]],
                         dtype=np.float32)


def aggregate_local_labels(labels, *, rule="or", exact_k=1):
    """Apply the specified count rule to the last axis of binary local labels."""
    labels = np.asarray(labels)
    if labels.ndim == 0 or labels.shape[-1] < 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected nonempty binary local-label vectors")
    count, scene_count = labels.sum(axis=-1), labels.shape[-1]
    if rule == "or":
        result = count > 0
    elif rule == "majority":
        result = count > scene_count / 2  # ties are negative
    elif rule == "exact_k":
        if not isinstance(exact_k, (int, np.integer)) or not 0 <= exact_k <= scene_count:
            raise ValueError("exact_k must lie between zero and scene_count")
        result = count == exact_k
    elif rule == "parity":
        result = count % 2 == 1
    else:
        raise ValueError(f"Unknown aggregation rule {rule!r}")
    return np.asarray(result, dtype=np.int64)


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _pattern_case(patterns):
    multiplicities = tuple(sorted(Counter(p.tobytes() for p in patterns).values()))
    if multiplicities not in CASES:
        raise ValueError("Compositions require exactly four local objects")
    case = CASES[multiplicities]
    return int(case == "one_pair"), case


def render_index_scene(patterns, positions, scene_size=32):
    """Render color-index patterns; -1 denotes background, 0..3 palette entries."""
    image = np.full((scene_size, scene_size), -1, dtype=np.int8)
    for pattern, (y, x) in zip(patterns, positions, strict=True):
        if not (0 <= y <= scene_size - 4 and 0 <= x <= scene_size - 4):
            raise ValueError("Local object lies outside its scene")
        if np.any(image[y:y + 4, x:x + 4] >= 0):
            raise ValueError("Local objects overlap")
        image[y:y + 4, x:x + 4] = pattern
    return image


def render_palette(index_image, palette):
    palette = np.asarray(palette, dtype=np.float32)
    if (palette.shape != (4, 3) or not np.isfinite(palette).all()
            or np.any(palette < 0) or np.any(palette > 1)
            or np.any(np.all(palette == 0, axis=1))
            or len({c.tobytes() for c in palette}) != 4):
        raise ValueError("Palette requires four distinct finite nonblack RGB colors in [0,1]")
    image = np.zeros((*index_image.shape, 3), dtype=np.float32)
    foreground = index_image >= 0
    image[foreground] = palette[index_image[foreground]]
    return image


def patch_tokens(image):
    """Lossless row-major 4x4 RGB patches and top-left coordinates in (x,y)."""
    height, width, channels = image.shape
    if height % 4 or width % 4 or channels != 3:
        raise ValueError("RGB dimensions must be multiples of four")
    features = image.reshape(height // 4, 4, width // 4, 4, 3)
    features = features.transpose(0, 2, 1, 3, 4).reshape(-1, 48).copy()
    features[(features == 0).all(axis=1)] = -1
    ys, xs = np.mgrid[0:height:4, 0:width:4]
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    return features, coords


@dataclass(frozen=True)
class ScenePool:
    """Verified, already-split local scenes; identity and layout metadata stay diagnostic."""

    split: str
    index_images: np.ndarray
    positions: np.ndarray
    patterns: np.ndarray
    labels: np.ndarray
    cases: tuple[str, ...]
    scene_ids: tuple[str, ...]  # canonical, palette-independent rendered scene IDs
    image_hashes: tuple[str, ...]
    appearance_ids: frozenset[str]  # canonical index-pattern bytes, not pool offsets
    layout_buckets: Mapping[str, Mapping[int, tuple[int, ...]]]


def scene_pool_from_data(data, *, split=None):
    """Adapt the controlled generator's NumPy dictionary to a verified scene bank.

    Canonical scene hashes ignore palette, so recoloring a training scene cannot
    pass as an independent held-out constituent scene. Pixel hashes are also kept.
    """
    data_split = data.get("split")
    if split is None:
        split = data_split
    if split not in {"train", "validation", "test"} or (data_split is not None and data_split != split):
        raise ValueError("Every local scene bank needs its pre-composition split")
    patterns = np.asarray(data["patterns"])
    positions = np.asarray(data["positions"], dtype=np.int64)
    labels = np.asarray(data["y"], dtype=np.int64)
    if (patterns.shape != (len(labels), 4, 4, 4) or positions.shape != (len(labels), 4, 2)
            or not np.isin(patterns, [0, 1, 2, 3]).all()):
        raise ValueError("Expected four 4x4 color-index patterns and corners per scene")
    patterns = patterns.astype(np.uint8)
    if len(labels) == 0:
        raise ValueError("Scene bank cannot be empty")
    buckets = defaultdict(lambda: defaultdict(list))
    scenes, scene_ids, pixel_hashes, cases = [], [], [], []
    for i, (objects, corners, label) in enumerate(zip(patterns, positions, labels, strict=True)):
        actual_label, case = _pattern_case(objects)
        if actual_label != label:
            raise ValueError("Local metadata label disagrees with rendered pattern equality")
        for pattern in objects:
            if not np.array_equal(np.bincount(pattern.ravel(), minlength=4), [4] * 4):
                raise ValueError("Every object must contain four pixels of every palette color")
        index_image = render_index_scene(objects, corners)
        scene_id = _hash(index_image)
        if "images" in data:
            image = np.asarray(data["images"][i], dtype=np.float32)
            palette = data["palettes"][i]
            if not np.array_equal(image, render_palette(index_image, palette)):
                raise ValueError("Local rendered scene disagrees with canonical index patterns")
            pixel_hashes.append(_hash(image))
        else:
            pixel_hashes.append(_hash(render_palette(index_image, FIXED_PALETTE)))
        scenes.append(index_image)
        scene_ids.append(scene_id)
        cases.append(case)
        buckets[_hash(index_image >= 0)][int(label)].append(i)
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Local scene bank contains duplicate canonical scenes")
    for layout, by_label in buckets.items():
        if set(by_label) != {0, 1} or len(by_label[0]) != len(by_label[1]):
            raise ValueError(f"Local layout {layout[:12]} must be exactly balanced across labels")
    return ScenePool(
        split, np.stack(scenes), positions, patterns, labels, tuple(cases), tuple(scene_ids),
        tuple(pixel_hashes), frozenset(_hash(p) for objects in patterns for p in objects),
        {layout: {label: tuple(indices) for label, indices in by_label.items()}
         for layout, by_label in buckets.items()},
    )


def verify_split_pools(pools, *, appearance_holdout=True):
    """Reject any canonical or RGB scene reuse across splits before composition."""
    if len({p.split for p in pools}) != len(pools):
        raise ValueError("Expected one scene bank per unique split")
    rows = []
    for i, first in enumerate(pools):
        for second in pools[i + 1:]:
            canonical = len(set(first.scene_ids) & set(second.scene_ids))
            pixels = len(set(first.image_hashes) & set(second.image_hashes))
            appearances = len(first.appearance_ids & second.appearance_ids)
            if canonical or pixels or (appearance_holdout and appearances):
                raise ValueError(f"Split leakage: {first.split}/{second.split}: "
                                 f"{canonical} canonical scenes, {pixels} images, "
                                 f"{appearances} appearances")
            rows.append({"splits": [first.split, second.split], "canonical_scene_overlap": canonical,
                         "pixel_scene_overlap": pixels, "appearance_overlap": appearances})
    return rows


def lattice_origins(canvas_size, scene_count, *, scene_size=32, gutter=4):
    """Top-left (y,x) origins on the same regular lattice for both labels."""
    if (canvas_size < scene_size or canvas_size % 4 or gutter < 0 or gutter % 4
            or scene_count < 1):
        raise ValueError("Invalid size/count; canvas and gutter must align to 4-pixel patches")
    slots = [(y, x) for y in range(0, canvas_size - scene_size + 1, scene_size + gutter)
             for x in range(0, canvas_size - scene_size + 1, scene_size + gutter)]
    if len(slots) < scene_count:
        raise ValueError("Scene lattice does not fit on the canvas")
    return np.asarray(slots, dtype=np.int64)


def _allowed_counts(scene_count, rule, exact_k):
    labels = np.arange(scene_count)[None, :] < np.arange(scene_count + 1)[:, None]
    outcomes = aggregate_local_labels(labels, rule=rule, exact_k=exact_k)
    allowed = {label: np.flatnonzero(outcomes == label) for label in [0, 1]}
    if any(len(counts) == 0 for counts in allowed.values()):
        raise ValueError("This aggregation setting does not admit both canvas classes")
    return allowed


def build_balanced_composite(pool, *, n_samples, canvas_size=80, scene_count=4,
                             rule="or", exact_k=1, seed=42, palette_mode="fixed", gutter=4,
                             fixed_origins=None):
    """Compose one split, matching every positive/negative pair's nuisances exactly.

    The caller must run verify_split_pools on all local banks before composing.
    Scene choices are random within matching label/layout buckets. Layouts are
    sampled without replacement per canvas when the bank is large enough.
    Region origins, local foreground masks, palette and total color amounts are
    identical within opposite-label pairs. Region IDs/corners are metadata only.
    """
    if n_samples < 2 or n_samples % 2:
        raise ValueError("n_samples must be a positive even number")
    if palette_mode not in {"fixed", "shared_per_canvas"}:
        raise ValueError("Only globally shared palettes are supported")
    rng = np.random.default_rng(seed)
    slots = lattice_origins(canvas_size, scene_count, gutter=gutter)
    if fixed_origins is not None:
        raw_origins = np.asarray(fixed_origins)
        fixed_origins = np.asarray(fixed_origins, dtype=np.int64)
        if (fixed_origins.shape != (scene_count, 2)
                or not np.array_equal(raw_origins, fixed_origins)
                or len({tuple(origin) for origin in fixed_origins}) != scene_count
                or any(tuple(origin) not in {tuple(slot) for slot in slots} for origin in fixed_origins)):
            raise ValueError("fixed_origins must be distinct legal scene-lattice origins")
    allowed = _allowed_counts(scene_count, rule, exact_k)
    half = n_samples // 2
    schedules = {label: rng.permutation(np.resize(rng.permutation(counts), half))
                 for label, counts in allowed.items()}
    layout_ids = list(pool.layout_buckets)
    images, x, local_x, labels, local_labels, local_cases = [], [], [], [], [], []
    origins_all, selected_all, palettes, pairs = [], [], [], []
    token_coords = patch_tokens(np.zeros((canvas_size, canvas_size, 3), np.float32))[1]
    local_coords = patch_tokens(np.zeros((32, 32, 3), np.float32))[1]
    for pair in range(half):
        origins = (fixed_origins.copy() if fixed_origins is not None else
                   slots[rng.choice(len(slots), scene_count, replace=False)])
        chosen_layouts = rng.choice(layout_ids, scene_count, replace=len(layout_ids) < scene_count)
        palette = (FIXED_PALETTE.copy() if palette_mode == "fixed" else
                   rng.uniform(0.1, 0.95, size=(4, 3)).astype(np.float32))
        for target in [0, 1]:
            local = np.zeros(scene_count, dtype=np.int64)
            local[rng.choice(scene_count, int(schedules[target][pair]), replace=False)] = 1
            selected = np.array([rng.choice(pool.layout_buckets[layout][int(label)])
                                 for layout, label in zip(chosen_layouts, local, strict=True)])
            index_image = np.full((canvas_size, canvas_size), -1, dtype=np.int8)
            for scene_index, (y, z) in zip(selected, origins, strict=True):
                index_image[y:y + 32, z:z + 32] = pool.index_images[scene_index]
            image = render_palette(index_image, palette)
            local_images = [render_palette(pool.index_images[i], palette) for i in selected]
            images.append(image)
            x.append(patch_tokens(image)[0])
            local_x.append(np.stack([patch_tokens(im)[0] for im in local_images]))
            labels.append(target)
            local_labels.append(local)
            local_cases.append([pool.cases[i] for i in selected])
            selected_all.append(selected)
            origins_all.append(origins.copy())
            palettes.append(palette.copy())
            pairs.append(pair)
    order = rng.permutation(n_samples)
    local_labels = np.asarray(local_labels)[order]
    local_cases = np.asarray(local_cases)[order]
    selected_all = np.asarray(selected_all)[order]
    positive_counts = local_labels.sum(axis=1)
    data = {"images": np.stack(images)[order], "x": np.stack(x)[order],
            "coords": np.broadcast_to(token_coords, (n_samples, *token_coords.shape)).copy(),
            "y": np.asarray(labels, dtype=np.int64)[order], "local_x": np.stack(local_x)[order],
            "local_coords": np.broadcast_to(local_coords, (n_samples, scene_count, 64, 2)).copy(),
            "local_labels": local_labels, "local_cases": local_cases,
            "positive_scene_counts": positive_counts, "scene_indices": selected_all,
            "scene_ids": np.asarray(pool.scene_ids)[selected_all],
            "origins": np.asarray(origins_all)[order], "palettes": np.asarray(palettes)[order],
            "pair_ids": np.asarray(pairs)[order], "layout_pair_ids": np.asarray(pairs)[order],
            "split": pool.split,
            "canvas_size": canvas_size, "scene_count": scene_count, "rule": rule,
            "exact_k": exact_k, "gutter": gutter, "palette_mode": palette_mode,
            "appearance_ids": sorted(pool.appearance_ids),
            "cases": [";".join(f"{case}:{count}" for case, count in sorted(Counter(row).items()))
                      for row in local_cases],
            "sampling": "Canvas labels 50/50; counts evenly cycled within each allowed class; "
                        "positive scene slots uniform conditional on count; balanced local layouts",
            "grouping": "32x32 regular spatial lattice; 4-pixel-aligned gutter; "
                        "no supplied region features; identities may recur across regions",
            "recommended_rope_high": canvas_size}
    data["origin_mode"] = "fixed across canvases" if fixed_origins is not None else "random lattice slots"
    verify_composite(data, pool)
    return data


def build_composite_splits(pools, *, appearance_holdout=True, seed=42, **settings):
    """Enforce constituent/appearance separation *before* any canvases are built."""
    pools = dict(pools)
    if any(name != pool.split for name, pool in pools.items()):
        raise ValueError("Scene bank names must match their assigned splits")
    overlaps = verify_split_pools(list(pools.values()), appearance_holdout=appearance_holdout)
    datasets = {name: build_balanced_composite(pool, seed=seed + i, **settings)
                for i, (name, pool) in enumerate(pools.items())}
    for i, (name, data) in enumerate(datasets.items()):
        for other_name, other in list(datasets.items())[i + 1:]:
            if set(data["scene_ids"].ravel()) & set(other["scene_ids"].ravel()):
                raise ValueError(f"Constituent leakage in composed splits {name}/{other_name}")
            if {_hash(im) for im in data["images"]} & {_hash(im) for im in other["images"]}:
                raise ValueError(f"Identical canvas images in splits {name}/{other_name}")
    return datasets, overlaps


def local_label_oracle(data):
    return aggregate_local_labels(data["local_labels"], rule=data["rule"], exact_k=data["exact_k"])


def local_predictions_to_canvas(local_predictions, data):
    """Diagnostic hook: apply f to local binary model predictions, not metadata."""
    local_predictions = np.asarray(local_predictions)
    if local_predictions.shape != data["local_labels"].shape:
        raise ValueError("One binary prediction is required per constituent scene")
    return aggregate_local_labels(local_predictions, rule=data["rule"], exact_k=data["exact_k"])


def predict_local_then_aggregate(data, predictor: Callable, *, batch_size=32):
    """Call predictor(local_x, local_coords) on NumPy batches; return local/canvas labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    x = data["local_x"].reshape(-1, 64, 48)
    coords = data["local_coords"].reshape(-1, 64, 2)
    parts = [np.asarray(predictor(x[start:start + batch_size], coords[start:start + batch_size]))
             .reshape(-1) for start in range(0, len(x), batch_size)]
    local = np.concatenate(parts).reshape(data["local_labels"].shape)
    return local, local_predictions_to_canvas(local, data)


def composite_subgroup_metrics(predictions, data):
    """Accuracy by canvas label, positive-scene count and repetition-case mixture."""
    predictions = np.asarray(predictions)
    truth = np.asarray(data["y"])
    if predictions.shape != truth.shape or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Expected one binary prediction per canvas")
    correct = predictions == truth
    groups = {"label": truth, "positive_scene_count": data["positive_scene_counts"],
              "case_mix": np.asarray(data["cases"])}
    details = {}
    for name, values in groups.items():
        details[name] = {str(value): {"count": int(np.sum(values == value)),
                                     "accuracy": float(correct[values == value].mean())}
                         for value in np.unique(values)}
    accuracies = [row["accuracy"] for subgroup in details.values() for row in subgroup.values()]
    return {"accuracy": float(correct.mean()), "worst_subgroup_accuracy": min(accuracies),
            "macro_count_accuracy": float(np.mean([row["accuracy"]
                for row in details["positive_scene_count"].values()])), "subgroups": details}


def verify_composite(data, pool):
    """Recheck local equality on actual canvas crops, labels, tokens and nuisances."""
    size, m = data["canvas_size"], data["scene_count"]
    if data["split"] != pool.split:
        raise ValueError("Composite belongs to a different constituent split")
    masks = {label: Counter() for label in [0, 1]}
    for i, image in enumerate(data["images"]):
        actual_local, actual_cases = [], []
        for j, (index, (y, x)) in enumerate(zip(data["scene_indices"][i], data["origins"][i], strict=True)):
            if data["scene_ids"][i, j] != pool.scene_ids[index]:
                raise ValueError("Constituent provenance changed")
            region = image[y:y + 32, x:x + 32]
            expected = render_palette(pool.index_images[index], data["palettes"][i])
            if not np.array_equal(region, expected):
                raise ValueError("Rendered region disagrees with constituent scene")
            local_tokens, local_coords = patch_tokens(region)
            if (not np.array_equal(data["local_x"][i, j], local_tokens)
                    or not np.array_equal(data["local_coords"][i, j], local_coords)):
                raise ValueError("Local model diagnostic inputs disagree with rendered region")
            crops = [region[a:a + 4, b:b + 4] for a, b in pool.positions[index]]
            actual_label, actual_case = _pattern_case(crops)
            actual_local.append(actual_label)
            actual_cases.append(actual_case)
        if (not np.array_equal(actual_local, data["local_labels"][i])
                or list(data["local_cases"][i]) != actual_cases
                or int(aggregate_local_labels(actual_local, rule=data["rule"], exact_k=data["exact_k"]))
                != data["y"][i]):
            raise ValueError("Rendered local or aggregate label disagrees with metadata")
        tokens, coords = patch_tokens(image)
        if not np.array_equal(tokens, data["x"][i]) or not np.array_equal(coords, data["coords"][i]):
            raise ValueError("Composite tokenization lost pixels or changed coordinates")
        foreground = np.any(image != 0, axis=-1)
        if foreground.sum() != 64 * m:
            raise ValueError("Foreground amount changed")
        for color in data["palettes"][i]:
            if np.all(image == color, axis=-1).sum() != 16 * m:
                raise ValueError("Global palette histogram changed")
        masks[int(data["y"][i])][_hash(foreground)] += 1
    if masks[0] != masks[1]:
        raise ValueError("Canvas layouts reveal labels")
    for pair in np.unique(data["pair_ids"]):
        indices = np.flatnonzero(data["pair_ids"] == pair)
        if len(indices) != 2 or set(data["y"][indices]) != {0, 1}:
            raise ValueError("Matched label pair is incomplete")
        a, b = indices
        if (not np.array_equal(data["palettes"][a], data["palettes"][b])
                or not np.array_equal(data["origins"][a], data["origins"][b])):
            raise ValueError("Matched pair's palette or scene placement changed")
    if not np.array_equal(data["pair_ids"], data["layout_pair_ids"]):
        raise ValueError("Layout pair identifiers changed")
    return {"samples": len(data["y"]), "canvas_size": size, "scene_count": m,
            "token_count": (size // 4) ** 2, "foreground_pixels": 64 * m,
            "foreground_fraction": 64 * m / size ** 2,
            "canvas_class_counts": dict(sorted(Counter(data["y"].tolist()).items())),
            "positive_scene_count_counts": dict(sorted(Counter(data["positive_scene_counts"].tolist()).items())),
            "local_case_counts": dict(sorted(Counter(data["local_cases"].ravel()).items())),
            "constituent_scene_count": len(set(data["scene_ids"].ravel())),
            "local_and_canvas_pixel_labels_verified": True,
            "lossless_tokens_and_xy_coordinate_grid_verified": True,
            "foreground_layouts_exactly_matched_across_canvas_labels": True,
            "global_color_histograms_equal": True,
            "coordinate_max": float(data["coords"].max()), "recommended_rope_high": size,
            "rope_neighbor_normalized_delta": 4 / size,
            "note": "Coordinate scaling checked; no learned RoPE performance claim"}


def opposite_label_global_count_fixture():
    """AABB+CCDD vs AABC+BCDD: equal coordinate-free patch bags, opposite OR labels."""
    base = np.arange(4, dtype=np.uint8)
    patterns = [np.roll(base, i).reshape(2, 2).repeat(2, 0).repeat(2, 1) for i in range(4)]
    assignments = np.array([[[0, 0, 1, 1], [2, 2, 3, 3]],
                            [[0, 0, 1, 2], [1, 2, 3, 3]]])
    corners = np.array([[0, 0], [0, 12], [12, 0], [12, 12]])
    origins = np.array([[0, 0], [0, 36]])
    images, local_labels = [], []
    for canvas in assignments:
        index_image = np.full((72, 72), -1, dtype=np.int8)
        labels = []
        for types, (y, x) in zip(canvas, origins, strict=True):
            objects = np.array([patterns[t] for t in types])
            labels.append(_pattern_case(objects)[0])
            index_image[y:y + 32, x:x + 32] = render_index_scene(objects, corners)
        images.append(render_palette(index_image, FIXED_PALETTE))
        local_labels.append(labels)
    x, coords = zip(*(patch_tokens(image) for image in images), strict=True)
    data = {"images": np.stack(images), "x": np.stack(x), "coords": np.stack(coords),
            "local_labels": np.array(local_labels), "assignments": assignments,
            "y": aggregate_local_labels(local_labels), "origins": origins}
    bags = [Counter(token.tobytes() for token in tokens) for tokens in x]
    if bags[0] != bags[1] or not np.array_equal(data["y"], [0, 1]):
        raise AssertionError("Global-count grouping counterexample failed")
    return data
