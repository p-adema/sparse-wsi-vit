"""ABMIL on the HalliGalli synthetic long-range reasoning benchmark.

image_size=4096, patch_size=16  →  N=65 536 patches, D=768 per bag.
Only 4 of 65 536 patches carry key-shape signal (0.006% informative),
matching the extreme-context regime of real WSI MIL.

Generation note: clutter_density is kept low (4) to avoid the data
pipeline becoming the bottleneck at this image scale (~1 000 clutter
elements vs ~3 800 at the default density of 15).

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
IMAGE_SIZE       = 4096  # → 256×256 = 65 536 patches per bag; 4/65 536 = 0.006% informative
PATCH_SIZE       = 16    # → D = 3×16² = 768
CLUTTER_DENSITY  = 4     # low to prevent generation from bottlenecking the GPU
TRAIN_SIZE       = 2000
VAL_SIZE         = 400

# ─── Architecture ────────────────────────────────────────────────────────────
# in_features / out_features are injected by the runner from the datamodule;
# values here are for readability only.
IN_FEATURES  = 3 * PATCH_SIZE ** 2   # 768
OUT_FEATURES = 2                      # binary, CrossEntropy

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 8    # reduced: each bag is 65 536 × 768 × 2 bytes (bf16) ≈ 96 MB
NUM_WORKERS   = 8    # more workers to keep up with generation at 4096px
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
        clutter_density=CLUTTER_DENSITY,
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
