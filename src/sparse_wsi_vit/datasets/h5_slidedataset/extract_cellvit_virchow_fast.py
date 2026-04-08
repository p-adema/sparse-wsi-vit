import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import gc
import h5py
from torchvision import transforms

import fast
import timm
from timm.layers import SwiGLUPacked
from huggingface_hub import login
from PIL import Image

# Approximate conversion: target_mpp × magnification ≈ 10 for standard objectives.
MPP_TO_MAG_FACTOR = 10.0


class HuggingFaceVirchowExtractor(nn.Module):
    def __init__(self, hf_token, device, concat_tokens=False):
        super().__init__()
        self.concat_tokens = concat_tokens

        if hf_token:
            print("Logging into Hugging Face...")
            login(token=hf_token)

        print("Loading Virchow2 from Hugging Face hub (paige-ai/Virchow2)...")
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def forward(self, x):
        # Using mixed precision inference as recommended for Virchow2
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(x)
            class_token = output[:, 0]

            if self.concat_tokens:
                patch_tokens = output[:, 5:]
                # Concatenate class token and mean patch tokens for 2560-dim feature
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            else:
                # Standard 1280-dim tile-level CLS representation
                embedding = class_token

            return embedding.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides_csv", type=Path, required=True)
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for paige-ai/Virchow2")
    parser.add_argument("--concat_tokens", action="store_true", help="Return 2560-dim features (CLS + mean patch) instead of 1280-dim (CLS only)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_mpp", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.slides_csv)
    device = torch.device(args.device)

    model = HuggingFaceVirchowExtractor(args.hf_token, device, concat_tokens=args.concat_tokens)
    magnification = int(MPP_TO_MAG_FACTOR / args.target_mpp)

    for idx, row in df.iterrows():
        slide_path_val = row.get('slidepath', row.get('path'))
        if pd.isna(slide_path_val) or not str(slide_path_val).strip():
            continue

        slide_path = Path(slide_path_val)
        slide_name = row['slidename']

        out_file = args.output_dir / f"{slide_name}.h5"
        partial_file = args.output_dir / f"{slide_name}.partial"

        if out_file.exists() or partial_file.exists():
            continue

        if not slide_path.exists():
            continue

        print(f"Processing {slide_name} using FAST...")
        fast.Reporter.setGlobalReportMethod(fast.Reporter.NONE)

        batch_imgs = []
        batch_coords = []
        n_written = 0
        feat_ds = None
        coord_ds = None

        try:
            importer = fast.WholeSlideImageImporter.create(str(slide_path))
            tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
            patchGenerator = (
                fast.PatchGenerator.create(224, 224, magnification=magnification)
                .connect(0, importer)
                .connect(1, tissueSegmentation)
            )

            pbar = tqdm(desc=f"Extracting {slide_name}")

            with h5py.File(partial_file, "w") as hf:
                def _write_batch(feats_np: np.ndarray, coords_np: np.ndarray) -> None:
                    """Append one batch of features and coords to the open HDF5 file."""
                    nonlocal feat_ds, coord_ds, n_written
                    n = len(feats_np)
                    if feat_ds is None:
                        feat_ds = hf.create_dataset(
                            "features", data=feats_np,
                            maxshape=(None, feats_np.shape[1]),
                            compression="gzip",
                        )
                        coord_ds = hf.create_dataset(
                            "coords", data=coords_np,
                            maxshape=(None, 2),
                            compression="gzip",
                        )
                    else:
                        feat_ds.resize(n_written + n, axis=0)
                        feat_ds[n_written:] = feats_np
                        coord_ds.resize(n_written + n, axis=0)
                        coord_ds[n_written:] = coords_np
                    n_written += n

                for patch in fast.DataStream(patchGenerator):
                    patch_width = int(patch.getFrameData("patch-width"))
                    patch_height = int(patch.getFrameData("patch-height"))
                    x_pos = int(patch.getFrameData("patchid-x")) * patch_width
                    y_pos = int(patch.getFrameData("patchid-y")) * patch_height

                    img_np = np.asarray(patch)
                    if img_np.shape[-1] == 4:
                        img_np = img_np[:, :, :3]

                    batch_imgs.append(model.transform(Image.fromarray(img_np)))
                    batch_coords.append([x_pos, y_pos])
                    pbar.update(1)

                    if len(batch_imgs) >= args.batch_size:
                        batch_tensor = torch.stack(batch_imgs).to(device)
                        with torch.no_grad():
                            feats_np = model(batch_tensor).cpu().numpy()
                        _write_batch(feats_np, np.array(batch_coords, dtype=np.int32))
                        batch_imgs = []
                        batch_coords = []

                # Flush remaining patches
                if batch_imgs:
                    batch_tensor = torch.stack(batch_imgs).to(device)
                    with torch.no_grad():
                        feats_np = model(batch_tensor).cpu().numpy()
                    _write_batch(feats_np, np.array(batch_coords, dtype=np.int32))

            pbar.close()

            if n_written > 0:
                partial_file.rename(out_file)
                print(f"  -> Saved {n_written} features to {out_file}")
            else:
                partial_file.unlink(missing_ok=True)
                print(f"  -> No valid patches found for {slide_name}")

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {slide_name}: {e}")
            partial_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
