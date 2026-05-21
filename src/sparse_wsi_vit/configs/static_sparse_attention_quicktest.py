"""Quick-test config for SparseViT5 static sparse attention.

Runs 10 training steps + 3 validation batches to verify NaN is fixed.
Includes P0 fixes: no nested compile, no validation_on_cpu, no sanity val.

Usage:
    PYTHONPATH=src python experiments/run.py \
        --config src/sparse_wsi_vit/configs/static_sparse_attention_quicktest.py
"""
import os
import torch
import datetime

from pathlib import Path

import pandas as pd

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
    TrainerConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig

from sparse_wsi_vit.models.sparse_vit5_slide_encoder import SparseViT5SlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

PIN_MEMORY = False

# ─── Data Details ──────────────────────────────────────────────
CSV_TRAIN_FOLD = Path("../splits/camelyon/full")
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = "../camelyon-emb"
DOWNSCALE_BLOCK = int(os.environ.get("DOWNSCALE_BLOCK", 1))
FEATURES_NAME = "patches"
FEATURES_SCALE = int(os.environ.get("FEATURES_SCALE", 224))

train_csv = str(CSV_TRAIN_FOLD / "train.csv")
val_csv = str(CSV_TRAIN_FOLD / "val.csv")
test_csv = str(CSV_TRAIN_FOLD / "test.csv")

# ─── Sparse attention type ───────────────────────────────────────────────────
SPARSE_ATTN = "static"

# ─── Hyperparameters (same as main config) ───────────────────────────────────
BATCH_SIZE = 1
NUM_WORKERS = 8 if FEATURES_SCALE == 224 else 1
WORKER_PREFETCH = 2
CLASS_WEIGHTS = True
IN_FEATURES = 1280
OUT_FEATURES = 1
PRECISION = "bf16-mixed"
GRADIENT_CHECKPOINTING = True

DEPTH = 3
HIDDEN_SIZE = 256
NUM_HEADS = 4
ROPE_DYNAMIC_HIGH = 220_000

NUM_CLS = 2
MLP_RATIO = 4.0
PROJ_DROPOUT = 0.2
DROP_PATH_RATE = 0.1
LAYER_SCALE = True
INIT_SCALE = 1e-4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

# ─── Quick-test overrides ────────────────────────────────────────────────────
TRAINING_ITERATIONS = 10
ACCUMULATE_GRAD_STEPS = 1

# ─── StaticSparseAttention-specific ──────────────────────────────────────────
WINDOW_SIZE = 4
CHUNK_SIZE = 256
FLEX_BLOCK_SIZE = 128
ROPE_THETA = 10_000.0
USE_HILBERT_SORT = True


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
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

    config.net = LazyConfig(SparseViT5SlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_cls=NUM_CLS,
        sparse_attn=SPARSE_ATTN,
        window_size=WINDOW_SIZE,
        chunk_size=CHUNK_SIZE,
        flex_block_size=FLEX_BLOCK_SIZE,
        rope_theta=ROPE_THETA,
        rope_coord_high=ROPE_DYNAMIC_HIGH,
        mlp_ratio=MLP_RATIO,
        proj_dropout=PROJ_DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
        layer_scale=LAYER_SCALE,
        init_scale=INIT_SCALE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        use_hilbert_sort=USE_HILBERT_SORT,
    )

    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        training_crop_tokens=None,
        eval_crop_tokens=50_000,
        compile_mode=None,  # Triton kernels handle their own compilation
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

    config.trainer = TrainerConfig(
        num_sanity_val_steps=0,
        val_check_interval=5,
        limit_val_batches=3,
    )

    config.scheduler = SchedulerConfig(
        name=None,
        warmup_iterations_percentage=0.0,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
        patience=999,
        max_duration=datetime.timedelta(hours=1),
    )

    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group=f"sparse_vit5_{SPARSE_ATTN}_quicktest",
        entity="dl2-2026",
    )

    return config
