"""NSA classification config.

Usage:
    python -m sparse_wsi_vit.experiments.run --config configs/native_sparse_attention.py
"""

# misschien eerst een wat mildere test? maar heb dit gewoon overgenomen van de laatste DSA config file

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

from sparse_wsi_vit.models.nsa import NSAViTSlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import (
    WSIAttnWrapper,
)
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

PIN_MEMORY = True

# ─── Data Details ──────────────────────────────────────────────
CSV_TRAIN_FOLD = "../splits/camelyon/full"
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = "../camelyon-emb/"
DOWNSCALE_BLOCK = 1
FEATURES_NAME = "cls"  # "patches" or "cls"
FEATURES_SCALE = 224

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"

# minimum 16 heads (GQA constraint) * 64 head_dim (Triton constraint) = 1024 embed_dim
EMBED_DIM = 1024
NUM_HEADS = 16
DEPTH = 3
NUM_CLS = 0  # Ignored by NSA (using ABMIL)

MLP_RATIO = 4.0
ATTN_DROPOUT = 0.0
PROJ_DROPOUT = 0.2

GRADIENT_CHECKPOINTING = False

WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
TRAINING_ITERATIONS = 2500
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 2
CLASS_WEIGHTS = True
WORKER_PREFETCH = 2

# NSA Specific Config ────────────────────────────────────────
BLOCK_SIZE = 64
BLOCK_COUNTS = 16
ROPE_DYNAMIC_HIGH = FEATURES_SCALE * DOWNSCALE_BLOCK

PATIENCE = 10  # Early stopping

# Set job duration to ~11h to ensure testing is not interrupted
MAX_DURATION = datetime.timedelta(hours=9, minutes=30)


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
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
        downscale_block=DOWNSCALE_BLOCK,
        pin_memory=PIN_MEMORY,
    )

    # Network: NSAViTSlideEncoder
    config.net = LazyConfig(NSAViTSlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=DEPTH,
        num_cls=NUM_CLS,
        # NSA Specific kwargs
        block_size=BLOCK_SIZE,
        block_counts=BLOCK_COUNTS,
        expansion_factor=MLP_RATIO,
        attn_dropout=ATTN_DROPOUT,
        proj_dropout=PROJ_DROPOUT,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        # Rope
        rope_coord_high=ROPE_DYNAMIC_HIGH,
    )

    # Lightning wrapper mappings
    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        compile_mode="max-autotune-no-cudagraphs",
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
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
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
        entity="dl2-2026",
        job_group="native_sparse_attention",  # Updated job group
    )

    return config
