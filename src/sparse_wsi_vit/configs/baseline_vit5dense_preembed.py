"""AB-MIL classification config.

Usage:
    python -m wsi_classification.experiments.run --config configs/baseline_abmil.py
"""

import torch
from pathlib import Path

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig

from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import (
    WSIAttnWrapper,
)
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
CSV_BASE = "../amc-data"
FEATURES_DIR = "../amc-data"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary tasks
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(
        H5FeatureBagDataModule
    )(
        train_csv=f"{CSV_BASE}/combined_tcga_amc.csv",
        val_csv=f"{CSV_BASE}/combined_tcga_amc.csv",  # TODO: Replace with actual val split!
        features_dir=FEATURES_DIR,
        label_col_name="tmb_binary",  # Changed from 'label' to an actual column present in the CSV
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Network: The very sketchy ViT-5/Small network
    config.net = LazyConfig(VitDensePreEmbedded)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
    )

    # Lightning wrapper mappings
    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        training_crop_tokens=2**13 - 5,  # 8187 (+ CLS + 4REG = 2**13),
        # training_compile=True,
    )

    # Optimizer
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Training
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # Scheduler
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    # W&B Logging
    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="baseline_abmil",
    )

    return config
