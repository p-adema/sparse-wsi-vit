"""SparseViT5 with static sparse attention classification config.

Usage:
    python -m sparse-wsi-vit.experiments.run --config configs/static_sparse_attention_config.py
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
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig

from sparse_wsi_vit.models.sparse_vit5_slide_encoder import SparseViT5SlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule
from sparse_wsi_vit.experiments.callbacks import AttentionMapCallback

# ─── Data Details ──────────────────────────────────────────────
CSV_TRAIN_FOLD = "../splits/camelyon/full"
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = "../camelyon-emb"
DOWNSCALE_BLOCK = 1
FEATURES_NAME = "patches"  # "patches" or "cls"
FEATURES_SCALE = int(os.environ.get("FEATURES_SCALE", 224))

train_csv = str(CSV_TRAIN_FOLD / "train.csv")
print(f"{train_csv=}")
val_csv = str(CSV_TRAIN_FOLD / "val.csv")
print(f"{val_csv=}")
test_csv = str(CSV_TRAIN_FOLD / "test.csv")
print(f"{test_csv=}")

train_slides = set(pd.read_csv(train_csv)["slidename"])
val_slides = set(pd.read_csv(val_csv)["slidename"])

overlap_train_val = train_slides & val_slides
overlap_test_val = test_slides & val_slides
overlap_train_test = test_slides & train_slides
print(f"Train: {len(train_slides)}, Val: {len(val_slides)}, Overlap: {len(overlap)}")
if overlap_train_val or overlap_test_val or overlap_train_test:
    print("Overlapping slides:", overlap_train_val + overlap_test_val + overlap_train_test)


# ─── Sparse attention type ───────────────────────────────────────────────────
SPARSE_ATTN="static"

# ─── Shared hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE=1
NUM_WORKERS = 8 if FEATURES_SCALE == 224 else 1
WORKER_PREFETCH=2
CLASS_WEIGHTS=True
IN_FEATURES=1280
OUT_FEATURES=1
PRECISION="bf16-mixed"
GRADIENT_CHECKPOINTING=True

DEPTH=3
HIDDEN_SIZE=256
NUM_HEADS=4            # embed_dim // num_heads=64
ROPE_DYNAMIC_HIGH = FEATURES_SCALE * DOWNSCALE_BLOCK

NUM_CLS=2
MLP_RATIO=4.0
PROJ_DROPOUT=0.2
DROP_PATH_RATE=0.1
LAYER_SCALE=True
INIT_SCALE=1e-4

TRAINING_ITERATIONS=1000
WARMUP_ITERATIONS_PERCENTAGE=0.05
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-2
GRAD_CLIP=1.0
ACCUMULATE_GRAD_STEPS=5
PATIENCE = 10
MAX_DURATION = datetime.timedelta(hours=9, minutes=30)


# ─── StaticSparseAttention-specific ──────────────────────────────────────────
WINDOW_SIZE=4            # neighbouring chunks on each side
CHUNK_SIZE=256           # patches per logical chunk (must be multiple of FLEX_BLOCK_SIZE)
FLEX_BLOCK_SIZE=128      # FlexAttention kernel tile size
ROPE_THETA=10_000.0
USE_HILBERT_SORT = True

def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=f"{CSV_BASE}/train.csv",
        val_csv=f"{CSV_BASE}/val.csv",
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

    config.net=LazyConfig(SparseViT5SlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_cls=NUM_CLS,
        sparse_attn=SPARSE_ATTN,
        # StaticSparseAttention kwargs
        window_size=WINDOW_SIZE,
        chunk_size=CHUNK_SIZE,
        flex_block_size=FLEX_BLOCK_SIZE,
        rope_theta=ROPE_THETA,
        rope_coord_high=ROPE_DYNAMIC_HIGH,
        # Shared kwargs
        mlp_ratio=MLP_RATIO,
        proj_dropout=PROJ_DROPOUT,
        drop_path_rate=DROP_PATH_RATE,
        layer_scale=LAYER_SCALE,
        init_scale=INIT_SCALE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        use_hilbert_sort=USE_HILBERT_SORT
    )

    # Lightning wrapper mappings
    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        training_crop_tokens=None,
        eval_crop_tokens=None,
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
        job_group=f"sparse_vit5_{SPARSE_ATTN}",
        entity="dl2-2026"
    )

    config.callbacks = [
        LazyConfig(AttentionMapCallback)(every_n_epochs=1, layer_index=-1),
    ]

    return config