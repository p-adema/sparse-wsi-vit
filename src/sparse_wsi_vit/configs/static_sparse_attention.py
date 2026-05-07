"""Static Sparse Attention classification config.

Usage:
    python -m sparse-wsi-vit.experiments.run --config configs/static_sparse_attention_config.py
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

from sparse_wsi_vit.models.static_sparse_attention import StaticSparseViTSlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import WSIAttnWrapper
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule
import pandas as pd
# ─── Data Details ──────────────────────────────────────────────
SPLITS_ROOT = Path("../splits/tcga-tmb")
FEATURES_DIR = "../tcga-v2/"

_fold_dirs = sorted(d for d in SPLITS_ROOT.iterdir() if d.is_dir())
TRAIN_CSVS = [str(d / "train.csv") for d in _fold_dirs]
print(f"{TRAIN_CSVS=}")
VAL_CSVS = [str(d / "val.csv") for d in _fold_dirs]
print(f"{VAL_CSVS=}")

train_slides = set(pd.concat([pd.read_csv(c) for c in TRAIN_CSVS])["slidename"])
val_slides = set(pd.concat([pd.read_csv(c) for c in VAL_CSVS])["slidename"])

overlap = train_slides & val_slides
print(f"Train: {len(train_slides)}, Val: {len(val_slides)}, Overlap: {len(overlap)}")
if overlap:
    print("Overlapping slides:", overlap)

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 4
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary task
PRECISION = "bf16-mixed"
EMBED_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 4
NUM_CLS = 2
WINDOW_SIZE = 3
DILATION = 1
WORKER_PREFETCH = 2
CLASS_WEIGHTS = True

WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
TRAINING_ITERATIONS = 2000
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 10


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=TRAIN_CSVS,
        val_csv=VAL_CSVS,
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_weights=CLASS_WEIGHTS,
        worker_prefetch=WORKER_PREFETCH,
        features_name="cls_224x224",
        coords_name="coords_224x224",
    )

    # Network: StaticSparseViTSlideEncoder
    config.net = LazyConfig(StaticSparseViTSlideEncoder)(
        in_features    = IN_FEATURES,
        out_features   = OUT_FEATURES,
        embed_dim      = EMBED_DIM,
        num_heads      = NUM_HEADS,
        num_layers     = NUM_LAYERS,
        num_cls        = NUM_CLS,
        window_size    = WINDOW_SIZE,
        dilation       = DILATION,
        attn_dropout   = 0.0,
        proj_dropout   = 0.0,
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
        job_group="static_sparse_attention",
        entity="dl2-2026"
    )

    return config