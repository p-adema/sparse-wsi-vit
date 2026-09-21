import argparse
import pathlib
from pathlib import Path
from typing import NamedTuple

import fast
import h5py
import numpy as np
import timm
import torch
import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

# Ignore all warnings, because libtiff has too many pointles ones
fast.Reporter.setGlobalReportMethod(fast.Reporter.NONE)

print("Loading model...")
with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    VIRCHOW2 = (
        timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        .eval()
        .cuda()
    )

    TRANSFORMS = create_transform(
        **resolve_data_config(VIRCHOW2.pretrained_cfg, model=VIRCHOW2)
    )

MAGNIFICATION = 20


@torch.compile(mode="max-autotune")
def process_region(region):
    output = VIRCHOW2(region).view(64, 261, 1280)

    # first tokens are CLS then 4 register tokens, then 16x16=256 embedding tokens.
    cls = output[:, 0]

    emb_56x56: torch.Tensor = (
        output[:, 5:]
        .view(64, 4, 4, 4, 4, 1280)  # [B, TY, Pool Y, TX, Pool X, Dim]
        .mean((2, 4))
    )

    emb_112x112: torch.Tensor = (
        output[:, 5:]
        .view(64, 2, 8, 2, 8, 1280)  # [B, TY, Pool Y, TX, Pool X, Dim]
        .mean((2, 4))
    )
    emb_224x224: torch.Tensor = output[:, 5:].view(64, 256, 1280).mean(1)

    return cls, emb_224x224, emb_112x112, emb_56x56


class Output(NamedTuple):
    coords_56x56: h5py.Dataset
    coords_112x112: h5py.Dataset
    coords_224x224: h5py.Dataset
    patches_56x56: h5py.Dataset
    patches_112x112: h5py.Dataset
    patches_224x224: h5py.Dataset
    cls_224x224: h5py.Dataset

    @classmethod
    def wrap_file(cls, file: h5py.File):
        return cls(
            *(
                file.create_dataset(
                    f"coords_{size}x{size}",
                    shape=(0, 2),
                    maxshape=(None, 2),
                    dtype="int64",
                    chunks=(64 * (224 // size) ** 2, 2),
                )
                for size in (56, 112, 224)
            ),
            *(
                file.create_dataset(
                    f"patches_{size}x{size}",
                    shape=(0, 1280),
                    maxshape=(None, 1280),
                    dtype="float16",
                    chunks=(64 * (224 // size) ** 2, 1280),
                )
                for size in (56, 112, 224)
            ),
            file.create_dataset(
                "cls_224x224",
                shape=(0, 1280),
                maxshape=(None, 1280),
                dtype="float16",
                chunks=(64, 1280),
            ),
        )

    @staticmethod
    def _append(ds: h5py.Dataset, new: torch.Tensor):
        old_size = ds.shape[0]
        addition = new.size(0)
        ds.resize(old_size + addition, axis=0)
        ds[old_size:] = new.numpy(force=True)

    def append(self, *news):
        assert len(news) == 7
        for ds, new in zip(self, news, strict=True):
            self._append(ds, new)


def verify_file(out_path: Path):
    assert out_path.stat().st_size > 0, f"Zero-size {out_path=}"
    keys = {
        "coords_56x56",
        "coords_112x112",
        "coords_224x224",
        "patches_56x56",
        "patches_112x112",
        "patches_224x224",
        "cls_224x224",
    }
    with h5py.File(out_path) as file:
        file_keys = set(file.keys())
        assert file_keys == keys, f"Incorrect {file_keys=} for {out_path=}"
        for key in keys:
            shp = file[key].shape[1:]
            assert len(shp) == 1
            if "coords" in key:
                assert shp == (2,)
            else:
                assert shp == (1280,)

        for size in ("56x56", "112x112", "224x224"):
            assert file[f"coords_{size}"].shape[0] == file[f"patches_{size}"].shape[0]
        assert file["coords_224x224"].shape[0] == file["cls_224x224"].shape[0]
        assert file["patches_224x224"].shape[0] == file["cls_224x224"].shape[0]
        assert file["patches_224x224"].shape[0] == file["patches_112x112"].shape[0] // 4
        assert file["patches_224x224"].shape[0] == file["patches_56x56"].shape[0] // 16
        assert (file["coords_224x224"][0] == file["coords_112x112"][0]).all()
        assert (file["coords_224x224"][0] == file["coords_56x56"][0]).all()


def process_batch(
    batch_img: torch.Tensor,
    batch_crd: torch.Tensor,
    tile_crd: torch.Tensor,
    output: Output,
    crop: int = 64,
):
    assert crop <= batch_img.size(0)
    cls, emb_224x224, emb_112x112, emb_56x56 = process_region(
        TRANSFORMS(batch_img).cuda()
    )
    coords_56x56 = batch_crd.view(64, 1, 1, 2) + tile_crd.view(1, 4, 4, 2)
    coords_112x112 = coords_56x56[:, ::2, ::2]
    coords_224x224 = coords_56x56[:, 0, 0]
    output.append(
        coords_56x56[:crop].view(crop * 4 * 4, 2),
        coords_112x112[:crop].reshape(crop * 2 * 2, 2),
        coords_224x224[:crop],
        emb_56x56[:crop].view(crop * 4 * 4, 1280),
        emb_112x112[:crop].view(crop * 2 * 2, 1280),
        emb_224x224[:crop],
        cls[:crop],
    )


def process_slide(slide_path: Path, out_file: h5py.File):
    assert slide_path.exists()
    importer = fast.WholeSlideImageImporter.create(str(slide_path))
    print()
    segmenter = fast.TissueSegmentation.create().connect(importer)

    patches = (
        fast.PatchGenerator.create(
            224, 224, magnification=MAGNIFICATION, overlapPercent=0
        )
        .connect(0, importer)
        .connect(1, segmenter)
    )
    batch_size = 64
    batch_img = torch.empty(batch_size, 3, 224, 224, memory_format=torch.channels_last)
    batch_crd = torch.empty(batch_size, 2, dtype=torch.int64)
    tile_crd = torch.arange(0, 4) * 56  # one axis of increasing coordinates in 56x56
    tile_crd = torch.cartesian_prod(tile_crd, tile_crd).view(4, 4, 2)
    output = Output.wrap_file(out_file)

    i = None
    for i, patch in enumerate(tqdm.tqdm(fast.DataStream(patches))):
        assert int(patch.getFrameData("patch-width")) == 224
        assert int(patch.getFrameData("patch-height")) == 224

        x_id = int(patch.getFrameData("patchid-x"))
        y_id = int(patch.getFrameData("patchid-y"))
        overlap_px = int(patch.getFrameData("patch-overlap-x"))
        assert overlap_px == int(patch.getFrameData("patch-overlap-y"))

        batch_crd[i % batch_size, 0] = y_id * (224 - overlap_px * 2) + overlap_px
        batch_crd[i % batch_size, 1] = x_id * (224 - overlap_px * 2) + overlap_px

        patch_arr = torch.from_numpy(np.asarray(patch).astype(np.float32) / 255.0)
        batch_img[i % batch_size] = patch_arr.permute(2, 0, 1)

        if i % batch_size == batch_size - 1:
            process_batch(batch_img, batch_crd, tile_crd, output)

    if i is None:
        raise ValueError(f"{slide_path=} had no valid tiles!")

    process_batch(batch_img, batch_crd, tile_crd, output, crop=i % batch_size + 1)


@torch.inference_mode
@torch.autocast("cuda", torch.bfloat16)
def main(slides: pathlib.Path, output: pathlib.Path):
    assert slides.exists() and slides.is_dir(), f"Invalid {slides=}"
    output.mkdir(parents=True, exist_ok=True)

    print("Finding slides...")
    paths: list[pathlib.Path] = sorted(slides.iterdir())

    unprocessed_paths = []
    processed_outputs = []
    for slide_path in paths:
        out_path = (output / slide_path.name).with_suffix(".h5")
        if out_path.exists():
            try:
                verify_file(out_path)
            except AssertionError as e:
                e.add_note(f"Failed at {out_path}")
                raise
            processed_outputs.append(out_path)
        else:
            unprocessed_paths.append(slide_path)

    assert len(unprocessed_paths) + len(processed_outputs) == len(paths)
    print(f"Found {len(paths)} slides,")
    print(f"    of which {len(unprocessed_paths)} still need processing")

    for slide_path in tqdm.tqdm(unprocessed_paths, desc="processing slides"):
        out_path = (output / slide_path.name).with_suffix(".h5")
        try:
            with h5py.File(out_path, "w") as out_file:
                process_slide(slide_path, out_file)
            verify_file(out_path)
        except (Exception, KeyboardInterrupt) as e:
            e.add_note(f"Failed on {slide_path=}")
            out_path.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess slides w/ Fast & Virchow2")
    parser.add_argument(
        "--slides",
        type=pathlib.Path,
        help="the directory where slides (e.g. in .svs or .tiff format) are located",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="the directory where embeddings should be placed",
        required=True,
    )
    args = parser.parse_args()
    main(args.slides, args.output)
