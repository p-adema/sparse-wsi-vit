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
import threading

print("Loading model...")
with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    model = (
        timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        .eval()
        .cuda()
    )

    transforms = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )


def find_roi(slide: OpenSlide, mask_path: Path) -> tuple[npt.NDArray, npt.NDArray]:
    down16 = slide.level_dimensions[0][0] // 16, slide.level_dimensions[0][1] // 16
    mask = np.asarray(Image.open(mask_path).resize(down16)) == 0
    mask_windows = np.lib.stride_tricks.sliding_window_view(mask, (14, 14))[::14, ::14]
    roi_y, roi_x = np.nonzero(mask_windows.sum(axis=(2, 3)) > 50)
    return roi_x, roi_y


def estimate_size(slide_path: pathlib.Path, mask_path: pathlib.Path) -> float:
    slide = openslide.OpenSlide(slide_path)
    roi, _ = find_roi(slide, mask_path)
    return len(roi) * 16 * 1280 * 2 / 1e9


def process_slide(
        slide_path: pathlib.Path, mask_path: pathlib.Path
) -> tuple[npt.NDArray, npt.NDArray]:
    slide = openslide.OpenSlide(slide_path)
    roi_x, roi_y = find_roi(slide, mask_path)

    regions = torch.empty((len(roi_x), 3, 224, 224), dtype=torch.float16)
    coords_glob = torch.asarray(np.stack((roi_x, roi_y), axis=-1) * 16)
    coords_loc = torch.cartesian_prod(torch.arange(0, 4), torch.arange(0, 4))
    coords = (coords_glob[:, None] + coords_loc[None, :]).view(len(roi_x) * 16, 2) * 14
    for i, (x, y) in enumerate(zip(roi_x, roi_y, strict=True)):
        reg = slide.read_region((x * 224, y * 224), 0, (224, 224)).convert("RGB")
        regions[i] = transforms(reg)

    patches = np.empty((len(roi_x), 16, 1280), dtype=np.float16)
    batch_size = 256
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        for i in tqdm.trange(
                0, len(roi_x), batch_size, desc="processing", disable=True
        ):
            output = model(regions[i: i + batch_size].cuda())  # size: B x 261 x 1280

            # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those
            patch = output[:, 5:].view(-1, 4, 4, 4, 4, 1280).mean((2, 4))
            patches[i: i + batch_size] = patch.cpu().view(-1, 16, 1280).numpy()

    return patches.reshape(coords.shape[0], 1280), coords.numpy()


def save_to_disk(out_path: pathlib.Path, features: npt.NDArray, coords: npt.NDArray):
    try:
        with h5py.File(out_path, mode="w") as file:
            file["features"] = features
            file["coords"] = coords
    except KeyboardInterrupt:
        out_path.unlink(missing_ok=True)

    verify_output(out_path)


def verify_output(out_path: Path):
    assert out_path.stat().st_size > 0, f"Zero-size {out_path=}"
    with h5py.File(out_path) as file:
        assert len(file["coords"].shape) == 2, f"Strange file at {out_path=}"
        assert len(file["features"].shape) == 2, f"Strange file at {out_path=}"
        coords_n, coords_2 = file["coords"].shape
        features_n, features_d = file["features"].shape
        assert coords_2 == 2, f"Strange file at {out_path=}"
        assert features_d == 1280, f"Strange file at {out_path=}"
        assert coords_n == features_n, f"Strange file at {out_path=}"


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
            verify_output(out_path)
            processed_outputs.append(out_path)
        else:
            unprocessed_paths.append((slide_path, mask_path))

    assert len(unprocessed_paths) + len(processed_outputs) == len(paths)
    print(f"Found {len(paths)} slides with masks,")
    print(f"    of which {len(unprocessed_paths)} still need processing")

    if check:
        total_size = 0
        for slide_path, mask_path in tqdm.tqdm(paths, desc="verifying slides"):
            total_size += estimate_size(slide_path, mask_path)

        print("All files seem OK.")
        print(f"Total estimated output: {total_out:.1f} GB")
        return

    save_handle = threading.Thread()
    save_handle.start()
    for slide_path, mask_path in tqdm.tqdm(unprocessed_paths, desc="processing slides"):
        features, coords = process_slide(slide_path, mask_path)

        out_path = (output / slide_path.name).with_suffix(".h5")
        # The hard drive I'm using for this is slow, so I put this on a separate thread.
        save_handle.join()
        save_handle = threading.Thread(
            target=save_to_disk, args=(out_path, features, coords)
        )
        save_handle.start()


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
