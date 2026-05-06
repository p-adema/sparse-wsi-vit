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
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
CSV_BASE   = Path.home() / "splits/tcga-tmb/4"
FEATURES_DIR = Path.home() / "tcga-v2/"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"
EMBED_DIM = 384
NUM_HEADS = 4
NUM_LAYERS = 6
NUM_CLS = 2
CHECKPOINT_ACTIVATIONS = False
WORKER_PREFETCH = 10
CLASS_WEIGHTS = True

# DSA specific config
INDEXER_HEADS = 4
INDEXER_DIM = 32
TOP_K = 128
BLOCK_Q = 32
BLOCK_K = 32
BLOCK_D = 32   

TRAINING_ITERATIONS = 10000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 1


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
        features_name = "patches_112x112",  # low resolution!
        coords_name = "coords_112x112",
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
        gradient_checkpointing = CHECKPOINT_ACTIVATIONS,
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
        entity="dl2-2026",
        job_group="deepseek_sparse_attention",
    )

    return config
