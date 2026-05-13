"""SparseViT5 with static sparse attention classification config.

Usage:
    python -m sparse-wsi-vit.experiments.run --config configs/static_sparse_attention_config.py
"""

import torch
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
CSV_BASE=Path("../splits/camelyon/0")
# CSV_BASE=Path("../splits/camelyon/full")
FEATURES_DIR="../camelyon-emb/"


train_csv = str(CSV_BASE / "train.csv")
print(f"{train_csv=}")
val_csv = str(CSV_BASE / "val.csv")
print(f"{val_csv=}")

train_slides = set(pd.read_csv(train_csv)["slidename"])
val_slides = set(pd.read_csv(val_csv)["slidename"])

overlap = train_slides & val_slides
print(f"Train: {len(train_slides)}, Val: {len(val_slides)}, Overlap: {len(overlap)}")
if overlap:
    print("Overlapping slides:", overlap)


# ─── Sparse attention type ───────────────────────────────────────────────────
SPARSE_ATTN="static"

# ─── Shared hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE=1
NUM_WORKERS=4
IN_FEATURES=1280
OUT_FEATURES=1
PRECISION="bf16-mixed"
# PRECISION="32-true"
EMBED_DIM=384
NUM_HEADS=6            # embed_dim // num_heads=64
DEPTH=6
NUM_CLS=8

MLP_RATIO=4.0
PROJ_DROPOUT=0.0
DROP_PATH_RATE=0.1
LAYER_SCALE=True
INIT_SCALE=1e-4

GRADIENT_CHECKPOINTING=True

USE_HILBERT_SORT = True

WARMUP_ITERATIONS_PERCENTAGE=0.05
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
TRAINING_ITERATIONS=2000
GRAD_CLIP=1.0
ACCUMULATE_GRAD_STEPS=8
CLASS_WEIGHTS=True
WORKER_PREFETCH=2

# ─── StaticSparseAttention-specific ──────────────────────────────────────────
WINDOW_SIZE=1            # neighbouring chunks on each side
CHUNK_SIZE=256           # patches per logical chunk (must be multiple of FLEX_BLOCK_SIZE)
FLEX_BLOCK_SIZE=128      # FlexAttention kernel tile size
ROPE_THETA=10_000.0
ROPE_COORD_HIGH=100_000.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=f"{CSV_BASE}/train.csv",
        val_csv=f"{CSV_BASE}/val.csv",
        # val_csv=f"{CSV_BASE}/test.csv",
        features_dir=FEATURES_DIR,
        label_col_name="label",
        # label_col_name="is_tumor",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_weights=CLASS_WEIGHTS,
        worker_prefetch=WORKER_PREFETCH,
        features_name="cls_224x224",
        coords_name="coords_224x224",
    )

    config.net=LazyConfig(SparseViT5SlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        num_cls=NUM_CLS,
        sparse_attn=SPARSE_ATTN,
        # StaticSparseAttention kwargs
        window_size=WINDOW_SIZE,
        chunk_size=CHUNK_SIZE,
        flex_block_size=FLEX_BLOCK_SIZE,
        rope_theta=ROPE_THETA,
        rope_coord_high=ROPE_COORD_HIGH,
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