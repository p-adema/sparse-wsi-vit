print("Importing libraries...")
from pathlib import Path

import openslide
import timm
import torch
from openslide import OpenSlide
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
import numpy as np
import numpy.typing as npt
import pathlib
import tqdm
import h5py
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

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
    EXECUTOR = ThreadPoolExecutor()


def find_roi(
        slide: OpenSlide, mask_path: Path
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    We want to find regions of interest which are sized 112x112 when downsampled 16x, such
    that we have 8x8=64 tokens per region at patch size 224x224, so we can apply NSA even
    at the most zoomed-out level.

    We also compute an `int112`, which is a mask for which subblocks of size 56x56 are
    relevant within this patch, and an `int56`, which is the same for subblocks of 28x28.

    Using these subblocks, we can do NSA at a higher resolution, e.g. using 28x28 masks
    (which are downsampled 16x from the original image, so represent a 448x448 patch)
    and patch size 56x56 we still have 8x8=64 tokens in the block.

    Returns:
        roi_x (n_roi)
        roi_y (n_roi)
        int112 (n_roi, 2, 2)
        int56 (n_roi, 4, 4)
    """
    down16 = slide.level_dimensions[0][0] // 16, slide.level_dimensions[0][1] // 16
    mask = np.asarray(Image.open(mask_path).resize(down16)) == 0
    mask_windows = np.lib.stride_tricks.sliding_window_view(mask, (112, 112))[
        ::112, ::112
    ]
    roi_y, roi_x = np.nonzero(mask_windows.sum(axis=(2, 3)) > 112 ** 2 * 0.1)
    n_regs = len(roi_y)
    subwindows56 = mask_windows[roi_y, roi_x].reshape(n_regs, 2, 56, 2, 56)
    int112 = subwindows56.sum((2, 4)) > 56 ** 2 * 0.1
    subwindows28 = mask_windows[roi_y, roi_x].reshape(n_regs, 4, 28, 4, 28)
    int56 = subwindows28.sum((2, 4)) > 28 ** 2 * 0.1
    return roi_x, roi_y, int112, int56


def estimate_size_gb(slide_path: pathlib.Path, mask_path: pathlib.Path) -> float:
    slide = openslide.OpenSlide(slide_path)
    roi, _, _, _ = find_roi(slide, mask_path)
    # For each roi, we have (at most) 64 56x56 patch embeddings, 16 112x112 patch
    # embeddings, 1 224x224 patch embedding and 1 224x224 CLS token, all 1280dim F16.
    return len(roi) * (64 + 16 + 2) * 16 * 1280 * 2 / 1e9


def load_region(slide: OpenSlide, x: int, y: int) -> torch.Tensor:
    region = slide.read_region((x * 8 * 224, y * 8 * 224), 0, (8 * 224, 8 * 224))
    region = (
        torch.from_numpy(np.asarray(region).astype(np.float32) / 255.0)[..., :3]
        .view(8, 224, 8, 224, 3)
        .permute(4, 0, 2, 1, 3)
        .reshape(64, 3, 224, 224)
    )
    return TRANSFORMS(region)


class Result(NamedTuple):
    coords_56x56: npt.NDArray
    coords_112x112: npt.NDArray
    coords_224x224: npt.NDArray
    patches_56x56: npt.NDArray
    patches_112x112: npt.NDArray
    patches_224x224: npt.NDArray
    cls_224x224: npt.NDArray


def process_slide(slide_path: pathlib.Path, mask_path: pathlib.Path) -> Result:
    slide = OpenSlide(slide_path)
    roi_x, roi_y, int112, int56 = find_roi(slide, mask_path)

    c_glob = torch.asarray(np.stack((roi_y, roi_x), axis=-1) * 8 * 224)  # (roi, 2)
    c_tile = torch.arange(0, 8 * 4) * 4 * 14
    c_loc = torch.cartesian_prod(c_tile, c_tile)  # (32x32, 2)

    coords_56x56 = (
        (c_glob[:, None] + c_loc[None, :])
        .view(len(roi_x), 4, 8, 4, 8, 2)
        .transpose(2, 3)
        .reshape(len(roi_x), 4, 4, 64, 2)
    )
    coords_112x112 = coords_56x56[:, ::2, ::2]
    coords_224x224 = coords_56x56[:, 0, 0]

    patches_56x56 = np.empty((len(roi_x), 4, 4, 64, 1280), dtype=np.float16)
    patches_112x112 = np.empty((len(roi_x), 2, 2, 64, 1280), dtype=np.float16)
    patches_224x224 = np.empty((len(roi_x), 64, 1280), dtype=np.float16)
    cls_224x224 = np.empty((len(roi_x), 64, 1280), dtype=np.float16)

    load_handle = EXECUTOR.submit(load_region, slide, roi_x[0], roi_y[0])
    for i in tqdm.trange(0, len(roi_x), desc="processing", leave=False):
        region = load_handle.result()
        if i + 1 < len(roi_x):
            load_handle = EXECUTOR.submit(
                load_region, slide, roi_x[i + 1], roi_y[i + 1]
            )

        cls, emb_224x224, emb_112x112, emb_56x56 = process_region(region.cuda())

        cls_224x224[i] = cls.cpu().numpy()
        patches_224x224[i] = emb_224x224.cpu().numpy()
        patches_112x112[i] = emb_112x112.cpu().numpy()
        patches_56x56[i] = emb_56x56.cpu().numpy()

    return Result(
        coords_56x56=coords_56x56.numpy()[int56],
        coords_112x112=coords_112x112.numpy()[int112],
        coords_224x224=coords_224x224.numpy(),
        patches_56x56=patches_56x56[int56],
        patches_112x112=patches_112x112[int112],
        patches_224x224=patches_224x224,
        cls_224x224=cls_224x224,
    )


@torch.compile(mode="max-autotune")
def process_region(region):
    output = VIRCHOW2(region).view(64, 261, 1280)

    # first tokens are CLS then 4 register tokens, then 16x16=256 embedding tokens.
    cls = output[:, 0]

    emb_56x56: torch.Tensor = (
        output[:, 5:]
        .view(8, 8, 4, 4, 4, 4, 1280)  # [BY, BX, TY, Pool Y, TX, Pool X, Dim]
        .mean((3, 5))
        .transpose(1, 2)
        .reshape(4, 8, 4, 8, 1280)
        .transpose(1, 2)
        .reshape(4, 4, 64, 1280)
    )

    emb_112x112: torch.Tensor = (
        output[:, 5:]
        .view(8, 8, 2, 8, 2, 8, 1280)  # [BY, BX, TY, Pool Y, TX, Pool X, Dim]
        .mean((3, 5))
        .transpose(1, 2)
        .reshape(2, 8, 2, 8, 1280)
        .transpose(1, 2)
        .reshape(2, 2, 64, 1280)
    )
    emb_224x224: torch.Tensor = output[:, 5:].view(64, 256, 1280).mean(1)

    return cls, emb_224x224, emb_112x112, emb_56x56


def save_to_disk(out_path: Path, res: Result):
    try:
        with h5py.File(out_path, mode="w") as file:
            file["coords_56x56"] = res.coords_56x56
            file["coords_112x112"] = res.coords_112x112
            file["coords_224x224"] = res.coords_224x224
            file["patches_56x56"] = res.patches_56x56
            file["patches_112x112"] = res.patches_112x112
            file["patches_224x224"] = res.patches_224x224
            file["cls_224x224"] = res.cls_224x224
    except KeyboardInterrupt:
        out_path.unlink(missing_ok=True)
        raise

    verify_output(out_path)


def verify_output(out_path: Path):
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
            assert len(shp) == 2
            if "coords" in key:
                assert shp == (64, 2)
            else:
                assert shp == (64, 1280)

        for size in ("56x56", "112x112", "224x224"):
            assert file[f"coords_{size}"].shape[0] == file[f"patches_{size}"].shape[0]
        assert file["coords_224x224"].shape[0] == file["cls_224x224"].shape[0]


@torch.inference_mode
@torch.autocast("cuda", torch.bfloat16)
def main(slides: pathlib.Path, masks: pathlib.Path, output: pathlib.Path, check: bool):
    assert slides.exists() and slides.is_dir(), f"Invalid {slides=}"
    assert masks.exists() and masks.is_dir(), f"Invalid {masks=}"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[pathlib.Path, pathlib.Path]] = []

    print("Finding slides...")
    for slide_path in sorted(slides.iterdir()):
        mask_path = pathlib.Path(f"{masks}/{slide_path.name}_MASK.png")
        assert mask_path.exists(), f"No mask at {mask_path=} for {slide_path=}"
        paths.append((slide_path, mask_path))

    unprocessed_paths = []
    processed_outputs = []
    for slide_path, mask_path in paths:
        out_path = (output / slide_path.name).with_suffix(".h5")
        if out_path.exists():
            try:
                verify_output(out_path)
            except AssertionError as e:
                e.add_note(f"Failed at {out_path}")
                raise
            processed_outputs.append(out_path)
        else:
            unprocessed_paths.append((slide_path, mask_path))

    assert len(unprocessed_paths) + len(processed_outputs) == len(paths)
    print(f"Found {len(paths)} slides with masks,")
    print(f"    of which {len(unprocessed_paths)} still need processing")

    if check:
        total_size = 0
        for slide_path, mask_path in tqdm.tqdm(paths, desc="verifying slides"):
            total_size += estimate_size_gb(slide_path, mask_path)

        print("All files seem OK.")
        print(f"Total estimated output: {total_size:.1f} GB")
        return

    for slide_path, mask_path in tqdm.tqdm(unprocessed_paths, desc="processing slides"):
        try:
            res = process_slide(slide_path, mask_path)
        except (Exception, KeyboardInterrupt) as e:
            e.add_note(f"Failed on {slide_path=}")
            raise

        out_path = (output / slide_path.name).with_suffix(".h5")
        # The hard drive I'm using for this is slow, so I put this on a separate thread.
        EXECUTOR.submit(save_to_disk, out_path, res)

    EXECUTOR.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sum the integers at the command line")
    parser.add_argument(
        "--slides",
        type=pathlib.Path,
        help="the directory where slides (e.g. in .svs or .tiff format) are located",
        required=True,
    )
    parser.add_argument(
        "--masks",
        type=pathlib.Path,
        help="the directory where masks (0/1 .png files, where 0 == tissue) are located",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="the directory where embeddings should be placed",
        required=True,
    )
    parser.add_argument(
        "--check",
        help="only check whether files are correct, and provide a size estimate",
        action="store_true",
    )
    args = parser.parse_args()
    main(args.slides, args.masks, args.output, args.check)
