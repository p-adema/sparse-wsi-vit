"""DSA classification config.

Usage:
    python -m sparse-wsi-vit.experiments.run --config configs/deepseek_sparse_attention.py
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

from sparse_wsi_vit.models.sparse_vit5_slide_encoder import SparseViT5SlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
CSV_BASE   = Path.home() / "splits/camelyon/0"
FEATURES_DIR = Path.home() / "camelyon-emb/"

SPARSE_ATTN = "dsa"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"
EMBED_DIM=384
NUM_HEADS=6            # embed_dim // num_heads=64
DEPTH=12
NUM_CLS=4

MLP_RATIO=4.0
ATTN_DROPOUT=0.0
PROJ_DROPOUT=0.0
DROP_PATH_RATE=0.1
LAYER_SCALE=True
INIT_SCALE=1e-4

GRADIENT_CHECKPOINTING=False

WARMUP_ITERATIONS_PERCENTAGE=0.05
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
TRAINING_ITERATIONS=2000
GRAD_CLIP=1.0
ACCUMULATE_GRAD_STEPS=1
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
ROPE_COORD_HIGH=100_000.0


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv    = str(CSV_BASE / "train.csv"),
        val_csv      = str(CSV_BASE / "val.csv"),
        features_dir = str(FEATURES_DIR),
        label_col_name = "label",
        batch_size   = BATCH_SIZE,
        num_workers  = NUM_WORKERS,
        class_weights=CLASS_WEIGHTS,
        worker_prefetch = WORKER_PREFETCH,
        features_name="cls_224x224",
        coords_name="coords_224x224",
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
        rope_coord_high=ROPE_COORD_HIGH,
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
        # entity="dl2-2026",
        job_group="deepseek_sparse_attention",
    )

    return config
