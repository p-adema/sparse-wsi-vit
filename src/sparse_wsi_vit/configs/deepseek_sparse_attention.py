"""DSA classification config.

Usage:
    python -m sparse-wsi-vit.experiments.run --config configs/deepseek_sparse_attention.py
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

from sparse_wsi_vit.models.sparse_vit5_slide_encoder import SparseViT5SlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

PIN_MEMORY = True

# ─── Data Details ──────────────────────────────────────────────
CSV_TRAIN_FOLD = Path.home() / "splits/camelyon/full"
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = Path.home() / "camelyon-emb/"
DOWNSCALE_BLOCK = 1
FEATURES_NAME = "cls"  # "patches" or "cls"
FEATURES_SCALE = 224

SPARSE_ATTN = "dsa"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"
EMBED_DIM=256
NUM_HEADS=4            # embed_dim // num_heads=64
DEPTH=3
NUM_CLS=2

MLP_RATIO=4.0
ATTN_DROPOUT=0.0
PROJ_DROPOUT=0.2
DROP_PATH_RATE=0.1
LAYER_SCALE=True
INIT_SCALE=1e-4

GRADIENT_CHECKPOINTING=False

WARMUP_ITERATIONS_PERCENTAGE=0.05
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-2
TRAINING_ITERATIONS=1000
GRAD_CLIP=1.0
ACCUMULATE_GRAD_STEPS=2
CLASS_WEIGHTS=True
WORKER_PREFETCH=2

# DSA specific config
INDEXER_HEADS = 4
INDEXER_DIM = 32
TOP_K = 128
BLOCK_Q = 32
BLOCK_K = 32
BLOCK_D = 32   

ROPE_THETA=10_000.0
ROPE_DYNAMIC_HIGH = FEATURES_SCALE * DOWNSCALE_BLOCK

PATIENCE = 10  # Early stopping

# Training will stop after 9h30m, after which testing will start.
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

    # Network: SparseViT5SlideEncoder
    config.net=LazyConfig(SparseViT5SlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_cls=NUM_CLS,
        sparse_attn=SPARSE_ATTN,
        # DeepseekSparseAttention kwargs
        indexer_heads=INDEXER_HEADS,
        indexer_dim=INDEXER_DIM,
        top_k=TOP_K,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        rope_theta=ROPE_THETA,
        rope_coord_high=ROPE_DYNAMIC_HIGH,
        # Shared kwargs
        mlp_ratio=MLP_RATIO,
        attn_dropout=ATTN_DROPOUT,
        proj_dropout=PROJ_DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
        layer_scale=LAYER_SCALE,
        init_scale=INIT_SCALE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
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
        job_group="deepseek_sparse_attention",
    )

    return config
