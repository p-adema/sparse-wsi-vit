"""Extract Virchow2 patch features from synthetic HalliGalli images.

Replaces the FAST-based tissue segmentation and patch generation from the WSI
pipeline with direct HalliGalli image generation. All patches are included —
there is no tissue mask to apply.

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
        --hf_token <YOUR_HF_TOKEN>
"""

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.layers import SwiGLUPacked
from tqdm import tqdm

from sparse_wsi_vit.datasets.halligalli_dataset.halligalli import HalliGalliGenerator


class HuggingFaceVirchowExtractor(nn.Module):
    def __init__(self, hf_token, device, concat_tokens=False):
        super().__init__()
        self.concat_tokens = concat_tokens

        if hf_token:
            print("Logging into Hugging Face...")
            from huggingface_hub import login
            login(token=hf_token)

        print("Loading Virchow2 from Hugging Face hub (paige-ai/Virchow2)...")
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def forward(self, x):
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            output = self.model(x)
            class_token = output[:, 0]

            if self.concat_tokens:
                patch_tokens = output[:, 5:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            else:
                embedding = class_token

            return embedding.to(torch.float32)


def _extract_split(
    split: str,
    n_samples: int,
    image_size: int,
    patch_size: int,
    generator_kwargs: dict,
    model: HuggingFaceVirchowExtractor,
    output_dir: Path,
    batch_size: int,
):
    feat_dir = output_dir / split / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    n_h = image_size // patch_size
    n_w = image_size // patch_size
    rows = []

    for idx in tqdm(range(n_samples), desc=split):
        sample_name = f"sample_{idx:06d}"
        out_file = feat_dir / f"{sample_name}.h5"
        partial_file = feat_dir / f"{sample_name}.partial"

        if out_file.exists() or partial_file.exists():
            # Re-read label from existing file to rebuild CSV on resume
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

        for pi in range(n_h):
            for pj in range(n_w):
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
                with torch.no_grad():
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
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Patch size in pixels. Each patch is resized to 224×224 for Virchow2.")
    parser.add_argument("--separation", type=float, default=1.0)
    parser.add_argument("--clutter_density", type=float, default=4)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--concat_tokens", action="store_true",
                        help="Use 2560-dim (CLS + mean patch) instead of 1280-dim (CLS only)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.image_size % args.patch_size != 0:
        raise ValueError(
            f"image_size {args.image_size} must be divisible by patch_size {args.patch_size}"
        )

    device = torch.device(args.device)
    model = HuggingFaceVirchowExtractor(args.hf_token, device, concat_tokens=args.concat_tokens)

    generator_kwargs = {
        "separation": args.separation,
        "clutter_density": args.clutter_density,
    }

    metadata = {
        "train_size":      args.train_size,
        "val_size":        args.val_size,
        "test_size":       args.test_size,
        "image_size":      args.image_size,
        "patch_size":      args.patch_size,
        "separation":      args.separation,
        "clutter_density": args.clutter_density,
        "concat_tokens":   args.concat_tokens,
        "feature_dim":     2560 if args.concat_tokens else 1280,
        "encoder":         "paige-ai/Virchow2",
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
