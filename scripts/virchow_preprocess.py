import matplotlib.pyplot as plt
import openslide

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import torchvision
from PIL import Image
import polars as pl
import numpy as np
import numpy.typing as npt
import pathlib
import tqdm
import h5py
import argparse

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


def process_slide(name: str):
    slide = openslide.OpenSlide(f"../tcga/{name}")
    segm = f"../grandqc/tcga-masks/tis_det_mask/{name}_MASK.png"
    assert round(slide.level_downsamples[2]) == 16
    mask = np.asarray(Image.open(segm).resize(slide.level_dimensions[2])) == 0
    mask_windows = np.lib.stride_tricks.sliding_window_view(mask, (14, 14))[::14, ::14]
    print(mask_windows.shape)
    roi_y, roi_x = np.nonzero(mask_windows.sum(axis=(2, 3)) > 50)
    out = h5py.File("tmp.h5", mode="w")

    regions = torch.empty((len(roi_x), 3, 224, 224), dtype=torch.float16)
    coords_glob = torch.asarray(np.stack((roi_x, roi_y), axis=-1) * 16)
    coords_loc = torch.cartesian_prod(torch.arange(0, 4), torch.arange(0, 4))
    coords = (coords_glob[:, None] + coords_loc[None, :]).view(len(roi_x) * 16, 2) * 14
    out["coords"] = coords.numpy()
    for i, (x, y) in tqdm.tqdm(
            enumerate(zip(roi_x, roi_y, strict=True)), total=len(roi_x), desc="loading"
    ):
        reg = slide.read_region((x * 224, y * 224), 0, (224, 224)).convert("RGB")
        regions[i] = transforms(reg)

    patches = np.empty((len(roi_x), 16, 1280), dtype=np.float16)
    batch_size = 256
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        for i in tqdm.trange(0, len(roi_x), batch_size, desc="processing"):
            output = model(regions[i: i + batch_size].cuda())  # size: B x 261 x 1280

            # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those
            patch = output[:, 5:].view(batch_size, 4, 4, 4, 4, 1280).mean((2, 4))
            patches[i: i + batch_size] = patch.cpu().view(-1, 16, 1280).numpy()

    out["features"] = patches.reshape(coords.shape[0], 1280)
    out.close()


process_slide("tumor_TCGA-BH-A0B9-01A-01-TSA.9a22384b-3086-4792-9d69-d2a5f8fdd8c7.svs")
