"""ViT-5 Dense on raw RGB pixel features from the Mondriaan synthetic benchmark.

Input: 1024 raw pixels (32×32 image, pool_size=1), each a 3-dim RGB vector.
downproj=256 up-projects 3→256 before the transformer.
rope_dynamic_high=32 calibrates RoPE for pixel coordinates 0–31.

Expected result: >90% accuracy on patterned — full pairwise attention can
compare pixel arrangements across the 4×4 objects.

Usage:
    MONDRIAAN_DATA_DIR=<path/to/patterned_ps1> \\
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/mondriaan_vit5dense.py
"""

import os
import torch

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.mondriaan_h5_datamodule import MondriaanH5DataModule

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.environ["MONDRIAAN_DATA_DIR"]
IN_FEATURES  = 3      # raw RGB pixels
OUT_FEATURES = 2      # binary, CrossEntropy

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 1024
NUM_WORKERS   = 4
PRECISION     = "bf16-mixed"

TRAINING_ITERATIONS          = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE                = 2e-4
WEIGHT_DECAY                 = 1e-4
GRAD_CLIP                    = 1.0
ACCUMULATE_GRAD_STEPS        = 1


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed  = 42

    config.dataset = LazyConfig(MondriaanH5DataModule)(
        data_dir=DATA_DIR,
        in_features=IN_FEATURES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    config.net = LazyConfig(VitDensePreEmbedded)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        downproj=256,
        depth=3,               # vit5_small defaults to 12; 3 blocks to match other experiments
        num_heads=4,           # 256 / 4 = 64
        rope_dynamic_high=32,  # pixel coords 0–31 (32×32 image)
    )

    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        training_crop_tokens=None,
        eval_crop_tokens=None,
        compile_mode=None,
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="mondriaan_vit5dense_patterned",
    )

    return config
