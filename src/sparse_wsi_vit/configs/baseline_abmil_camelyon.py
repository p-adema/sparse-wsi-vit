"""AB-MIL classification config.

Usage:
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_abmil_camelyon.py
"""

import datetime

import torch
from pathlib import Path

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig

from sparse_wsi_vit.models.abmil import ABMIL
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

PIN_MEMORY = True

# ─── Data ────────────────────────────────────────────────────────────────────
CSV_TRAIN_FOLD = "/home/scur0097/splits/camelyon/full"
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = "/scratch-shared/scur0097/camelyon-emb"
FEATURES_NAME = "cls"   # "cls" or "patches"
FEATURES_SCALE = 224
FLATTEN_BLOCK = True

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 8 if FEATURES_SCALE == 224 else 1
WORKER_PREFETCH = 5
CLASS_WEIGHTS = True
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary tasks
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
MAX_DURATION = datetime.timedelta(hours=9, minutes=30)


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(
        H5FeatureBagDataModule
    )(
        train_csv=f"{CSV_TRAIN_FOLD}/train.csv",
        val_csv=f"{CSV_TRAIN_FOLD}/val.csv",
        test_csv=f"{CSV_TRAIN_FOLD}/test.csv",
        features_dir=FEATURES_DIR,
        label_col_name=TARGET_NAME,
        labels=TARGET_OPTIONS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_weights=CLASS_WEIGHTS,
        worker_prefetch=WORKER_PREFETCH,
        features_name=f"{FEATURES_NAME}_{FEATURES_SCALE}x{FEATURES_SCALE}",
        coords_name=f"coords_{FEATURES_SCALE}x{FEATURES_SCALE}",
        flatten_block=FLATTEN_BLOCK,
        pin_memory=PIN_MEMORY,
        output_channels=OUT_FEATURES,
    )

    # Network: The Standard AB-MIL baseline written natively for 1280-dim CLS tokens
    config.net = LazyConfig(ABMIL)(
        in_features=IN_FEATURES,
        hidden_dim=384,
        out_features=OUT_FEATURES,
        num_branches=2,
        # ~2.6M parameters now!
    )

    # Lightning wrapper mappings
    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1)
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
        patience=PATIENCE,
        max_duration=MAX_DURATION,
    )

    # W&B Logging
    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group=f"baseline_abmil_camelyon_{FEATURES_NAME}_{FEATURES_SCALE}",
        entity="dl2-2026"
    )

    return config
