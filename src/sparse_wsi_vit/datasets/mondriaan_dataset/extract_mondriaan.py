"""Extract Mondriaan pixel features to HDF5.

Generates Mondriaan images and saves one HDF5 file per sample in a single
flat output directory. Split and label are encoded in the filename — no
sub-folders, no separate label files.

Output layout
-------------
<output_dir>/
    train_000000_label0.h5
    train_000001_label1.h5
    ...
    val_000000_label1.h5
    ...
    test_000000_label0.h5
    ...
    metadata.json

Each .h5 file contains:
    features  float32  (N, 3)  — per-pixel (or per-block) RGB values
    coords    int32    (N, 2)  — (x, y) positions in original image space

For ``pool_size=1`` (full resolution, default): N = image_size².
For ``pool_size=2`` (2× downsampled): N = (image_size // 2)²; each feature
is the mean of a 2×2 pixel block and coords are the block's top-left corner.
The downsampled variant is used for the resolution ablation — at pool_size=2
each key object collapses to a single grey token, making the task
information-theoretically unsolvable.

Usage
-----
    # full resolution (patterned, default)
    uv run src/sparse_wsi_vit/datasets/mondriaan_dataset/extract_mondriaan.py \\
        --output_dir data/mondriaan/patterned_ps1

    # 2× downsampled (resolution ablation)
    uv run src/sparse_wsi_vit/datasets/mondriaan_dataset/extract_mondriaan.py \\
        --output_dir data/mondriaan/patterned_ps2 --pool_size 2

    # solid-colour control
    uv run src/sparse_wsi_vit/datasets/mondriaan_dataset/extract_mondriaan.py \\
        --output_dir data/mondriaan/solid_ps1 --object_type solid
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from sparse_wsi_vit.datasets.mondriaan_dataset.mondriaan import MondriaanGenerator


def _extract_split(
    split: str,
    n_samples: int,
    image_size: int,
    object_type: str,
    min_sep: int,
    pool_size: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(n_samples), desc=split):
        img, label, _, _ = MondriaanGenerator.generate_single(
            image_size=image_size,
            object_type=object_type,
            target_label=idx % 2,   # balanced labels within each split
            min_sep=min_sep,
        )

        if pool_size == 1:
            # full resolution: each pixel is one token
            H, W, _ = img.shape
            features = img.reshape(H * W, 3)                    # (N, 3)
            ys, xs = np.mgrid[0:H, 0:W]
            coords = np.stack([xs.ravel(), ys.ravel()], axis=1) # (N, 2) x,y
        else:
            # average-pool pool_size×pool_size blocks
            H, W, C = img.shape
            H2, W2 = H // pool_size, W // pool_size
            blocked = img[:H2 * pool_size, :W2 * pool_size].reshape(
                H2, pool_size, W2, pool_size, C
            )
            features = blocked.mean(axis=(1, 3)).reshape(H2 * W2, C) # (N, 3)
            ys, xs = np.mgrid[0:H2, 0:W2]
            # coords reference original image space (top-left corner of each block)
            coords = np.stack(
                [xs.ravel() * pool_size, ys.ravel() * pool_size], axis=1
            )                                                    # (N, 2) x,y

        fname = output_dir / f"{split}_{idx:06d}_label{label}.h5"
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("features", data=features.astype(np.float32),
                              compression="gzip")
            hf.create_dataset("coords",   data=coords.astype(np.int32),
                              compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=Path, required=True)
    parser.add_argument("--train_size",  type=int,  default=10_000)
    parser.add_argument("--val_size",    type=int,  default=2_000)
    parser.add_argument("--test_size",   type=int,  default=2_000)
    parser.add_argument("--image_size",  type=int,  default=64)
    parser.add_argument("--object_type", choices=["patterned", "solid"],
                        default="patterned")
    parser.add_argument("--min_sep",     type=int,  default=8)
    parser.add_argument("--pool_size",   type=int,  default=1,
                        help="Spatial pooling before saving (1=pixel, 2=quadrant-level, 4=object-level).")
    args = parser.parse_args()

    if args.image_size % args.pool_size != 0:
        raise ValueError(
            f"image_size {args.image_size} must be divisible by pool_size {args.pool_size}"
        )

    metadata = {
        "train_size":  args.train_size,
        "val_size":    args.val_size,
        "test_size":   args.test_size,
        "image_size":  args.image_size,
        "object_type": args.object_type,
        "min_sep":     args.min_sep,
        "pool_size":   args.pool_size,
        "n_tokens":    (args.image_size // args.pool_size) ** 2,
        "feature_dim": 3,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata}")

    for split, n in [
        ("train", args.train_size),
        ("val",   args.val_size),
        ("test",  args.test_size),
    ]:
        print(f"\n{split} ({n} samples)...")
        _extract_split(
            split=split,
            n_samples=n,
            image_size=args.image_size,
            object_type=args.object_type,
            min_sep=args.min_sep,
            pool_size=args.pool_size,
            output_dir=args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
