"""HIPT on pre-extracted ShapePatchCNN features from HalliGalli synthetic images.

ShapePatchCNN (256-dim embedding) is trained from scratch on the HalliGalli
shape vocabulary and used as a frozen patch encoder.
Features are pre-extracted by extract_halligalli.py.

Usage:
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/halligalli_hipt.py
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
from sparse_wsi_vit.models.hipt import HIPT_None_FC
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.halligalli_h5_datamodule import HalliGalliH5DataModule

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.environ["HALLIGALLI_DATA_DIR"]
IN_FEATURES  = 256    # ShapePatchCNN embed_dim
OUT_FEATURES = 2      # binary, CrossEntropy

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 1     # standard for MIL bags
NUM_WORKERS   = 16
PRECISION     = "bf16-mixed"

TRAINING_ITERATIONS          = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE                = 2e-4
WEIGHT_DECAY                 = 1e-4
GRAD_CLIP                    = 1.0
ACCUMULATE_GRAD_STEPS        = 8     # effective batch size = 8


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed  = 42

    config.dataset = LazyConfig(HalliGalliH5DataModule)(
        data_dir=DATA_DIR,
        in_features=IN_FEATURES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    config.net = LazyConfig(HIPT_None_FC)(
        in_features=IN_FEATURES,
        size_arg="big",
        out_features=OUT_FEATURES,
        #1.7M parameters
    )

    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
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
        job_group="halligalli_hipt_cnn",
    )

    return config
