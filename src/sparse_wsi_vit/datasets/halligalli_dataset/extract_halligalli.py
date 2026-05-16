"""Extract ShapePatchCNN patch features from synthetic HalliGalli images.

Uses a ShapePatchCNN checkpoint trained by train_patch_encoder.py as the
frozen patch encoder. Replaces the FAST-based tissue segmentation and patch
generation from the WSI pipeline with direct HalliGalli image generation.
All patches are included — there is no tissue mask to apply.

Produces one .h5 file per sample (keys: ``features``, ``coords``) and a
``labels.csv`` per split, matching the format expected by H5FeatureBagDataset.

Output layout
-------------
<output_dir>/
    train/
        features/
            sample_000000.h5
            ...
        labels.csv
    val/
        ...
    test/
        ...

Usage
-----
    uv run src/sparse_wsi_vit/datasets/halligalli_dataset/extract_halligalli.py \\
        --output_dir data/halligalli \\
        --cnn_checkpoint checkpoints/patch_cnn/patch_cnn.pt
"""

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import HalliGalliGenerator
from sparse_wsi_vit.datasets.halligalli_dataset.patch_cnn import ShapePatchCNN


class CNNPatchExtractor(nn.Module):
    """Frozen ShapePatchCNN used as a patch feature extractor.

    Loads a checkpoint produced by train_patch_encoder.py, discards the
    classification head, and extracts embed_dim-d features from raw
    64×64 RGB patches (float32, values in [0, 1]).
    """

    def __init__(self, checkpoint_path: Path, device):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = ShapePatchCNN(
            embed_dim=ckpt["embed_dim"],
            num_classes=ckpt["num_classes"],
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)
        self.device = device

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @staticmethod
    def transform(patch_pil: Image.Image) -> torch.Tensor:
        """Convert a PIL patch to a float32 tensor in [0, 1] — no resize."""
        return to_tensor(patch_pil)  # (3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.forward_features(x)


def _extract_split(
    split: str,
    n_samples: int,
    image_size: int,
    patch_size: int,
    generator_kwargs: dict,
    model: CNNPatchExtractor,
    output_dir: Path,
    batch_size: int,
):
    feat_dir = output_dir / split / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    n_patches_per_side = image_size // patch_size
    rows = []

    for idx in tqdm(range(n_samples), desc=split):
        sample_name = f"sample_{idx:06d}"
        out_file = feat_dir / f"{sample_name}.h5"
        partial_file = feat_dir / f"{sample_name}.partial"

        if partial_file.exists():
            partial_file.unlink()  # crashed mid-write; re-extract cleanly

        if out_file.exists():
            with h5py.File(out_file, "r") as f:
                label = int(f.attrs["label"])
            rows.append({"slidename": sample_name, "label": label})
            continue

        img, label, _, _ = HalliGalliGenerator.generate_single(
            image_size=image_size,
            target_label=idx % 2,
            **generator_kwargs,
        )
        img_uint8 = (img * 255).astype(np.uint8)

        batch_imgs = []
        batch_coords = []

        for pi in range(n_patches_per_side):
            for pj in range(n_patches_per_side):
                y0, x0 = pi * patch_size, pj * patch_size
                patch = img_uint8[y0:y0 + patch_size, x0:x0 + patch_size]
                batch_imgs.append(model.transform(Image.fromarray(patch)))
                batch_coords.append([x0, y0])

        feat_ds = None
        coord_ds = None
        n_written = 0

        with h5py.File(partial_file, "w") as hf:
            hf.attrs["label"] = label

            def _write_batch(feats_np, coords_np):
                nonlocal feat_ds, coord_ds, n_written
                n = len(feats_np)
                if feat_ds is None:
                    feat_ds = hf.create_dataset(
                        "features",
                        data=feats_np,
                        maxshape=(None, feats_np.shape[1]),
                        compression="gzip",
                    )
                    coord_ds = hf.create_dataset(
                        "coords",
                        data=coords_np,
                        maxshape=(None, 2),
                        compression="gzip",
                    )
                else:
                    feat_ds.resize(n_written + n, axis=0)
                    feat_ds[n_written:] = feats_np
                    coord_ds.resize(n_written + n, axis=0)
                    coord_ds[n_written:] = coords_np
                n_written += n

            for start in range(0, len(batch_imgs), batch_size):
                chunk_imgs = batch_imgs[start:start + batch_size]
                chunk_coords = batch_coords[start:start + batch_size]
                batch_tensor = torch.stack(chunk_imgs).to(model.device)
                feats_np = model(batch_tensor).cpu().numpy()
                _write_batch(feats_np, np.array(chunk_coords, dtype=np.int32))

        partial_file.rename(out_file)
        rows.append({"slidename": sample_name, "label": label})

    csv_path = output_dir / split / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  -> {split}: {len(rows)} samples, labels saved to {csv_path}")

    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=10_000)
    parser.add_argument("--val_size", type=int, default=2_000)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--image_size", type=int, default=1792)
    parser.add_argument("--patch_size", type=int, default=224,
                        help="Patch size in pixels (224 or 112).")
    parser.add_argument("--clutter_density", type=float, default=30)
    parser.add_argument("--shape_radius", type=int, default=20,
                        help="Radius of key shapes in pixels. Should match train_patch_encoder.py.")
    parser.add_argument(
        "--randomize_key_positions",
        action="store_true",
        help="Place key shapes at random patch-interior-aligned positions (recommended).",
    )
    parser.add_argument("--cnn_checkpoint", type=Path, required=True,
                        help="Path to ShapePatchCNN checkpoint produced by train_patch_encoder.py.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.image_size % args.patch_size != 0:
        raise ValueError(
            f"image_size {args.image_size} must be divisible by patch_size {args.patch_size}"
        )

    device = torch.device(args.device)
    print(f"Loading CNN encoder from {args.cnn_checkpoint} ...")
    model = CNNPatchExtractor(args.cnn_checkpoint, device)

    generator_kwargs = {
        "clutter_density":         args.clutter_density,
        "shape_radius":            args.shape_radius,
        "randomize_key_positions": args.randomize_key_positions,
        "patch_size":              args.patch_size,
    }

    metadata = {
        "train_size":               args.train_size,
        "val_size":                 args.val_size,
        "test_size":                args.test_size,
        "image_size":               args.image_size,
        "patch_size":               args.patch_size,
        "clutter_density":          args.clutter_density,
        "shape_radius":             args.shape_radius,
        "randomize_key_positions":  args.randomize_key_positions,
        "encoder":                  f"ShapePatchCNN:{args.cnn_checkpoint.name}",
        "feature_dim":              model.embed_dim,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata written to {args.output_dir / 'metadata.json'}")

    for split, n in [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]:
        print(f"\nProcessing {split} split ({n} samples)...")
        _extract_split(
            split=split,
            n_samples=n,
            image_size=args.image_size,
            patch_size=args.patch_size,
            generator_kwargs=generator_kwargs,
            model=model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
