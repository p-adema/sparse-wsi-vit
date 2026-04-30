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

from sparse_wsi_vit.models.deepseek_sparse_attention import DSAViTSlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
CSV_BASE   = Path.home() / "splits/tcga-emb/0"
FEATURES_DIR = Path.home() / "tcga-emb"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 12
NUM_CLS = 2

# DSA specific config
INDEXER_HEADS = 4
INDEXER_DIM = 32
TOP_K = 512
BLOCK_Q = 16
BLOCK_K = 32
BLOCK_D = 32    

TRAINING_ITERATIONS = 10_0
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


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
    )

    # Network: DSAViTSlideEncoder
    config.net = LazyConfig(DSAViTSlideEncoder)(
        in_features    = IN_FEATURES,
        out_features   = OUT_FEATURES,
        embed_dim      = EMBED_DIM,
        num_heads      = NUM_HEADS,
        num_layers     = NUM_LAYERS,
        num_cls        = NUM_CLS,
        indexer_heads  = INDEXER_HEADS,
        indexer_dim    = INDEXER_DIM,
        top_k          = TOP_K,
        block_q        = BLOCK_Q,
        block_k        = BLOCK_K,
        block_d        = BLOCK_D,
        attn_dropout   = 0.0,
        proj_dropout   = 0.0,
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
    )

    # W&B Logging
    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="deepseek_sparse_attention",
    )

    return config
