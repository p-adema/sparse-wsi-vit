"""ABMIL on the HalliGalli synthetic long-range reasoning benchmark.

image_size=128, patch_size=16  →  N=64 patches, D=768 per bag.
Key shapes are 3 px radius, placed at max separation (corners), embedded
in clutter — ABMIL must attend to 4 specific patches out of 64 to solve it.

Usage:
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/halligalli_abmil.py
"""

import torch

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig
from sparse_wsi_vit.models.abmil import ABMIL
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.halligalli_datamodule import HalliGalliDataModule

# ─── Data ────────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 128   # → 8×8 = 64 patches per bag
PATCH_SIZE  = 16    # → D = 3×16² = 768
TRAIN_SIZE  = 2000
VAL_SIZE    = 400

# ─── Architecture ────────────────────────────────────────────────────────────
# in_features / out_features are injected by the runner from the datamodule;
# values here are for readability only.
IN_FEATURES  = 3 * PATCH_SIZE ** 2   # 768
OUT_FEATURES = 2                      # binary, CrossEntropy

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_WORKERS   = 4
PRECISION     = "bf16-mixed"

TRAINING_ITERATIONS          = 5_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE                = 2e-4
WEIGHT_DECAY                 = 1e-4
GRAD_CLIP                    = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = True   # set False for online W&B logging on Snellius
    config.seed  = 42

    config.dataset = LazyConfig(HalliGalliDataModule)(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    config.net = LazyConfig(ABMIL)(
        in_features=IN_FEATURES,
        hidden_dim=256,
        out_features=OUT_FEATURES,
        num_branches=1,
    )

    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),   # False → CrossEntropyLoss
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
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="halligalli_abmil",
    )

    return config
