"""ViT-5 classification config for debugging, training on first split from TCGA-like embeddings

Usage:
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_tcga.py
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

from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import (
    WSIAttnWrapper,
)
from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

# ─── Data Details ──────────────────────────────────────────────
CSV_BASE = "../splits/camelyon/0"
FEATURES_DIR = "../camelyon-emb"

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1  # Standard for MIL bags
NUM_WORKERS = 1  # Better for HDD or other slow disk
WORKER_PREFETCH = 10
CLASS_WEIGHTS = True  # this TCGA dataset has more cancer than healthy
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary tasks
PRECISION = "bf16-mixed"
CHECKPOINT_ACTIVATIONS = True  # instead of cropping

TRAINING_ITERATIONS = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 10


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False  # set to False to actually train
    config.seed = 42

    # Dataset: Connects to your H5 extraction
    config.dataset = LazyConfig(H5FeatureBagDataModule)(
        train_csv=f"{CSV_BASE}/train.csv",
        val_csv=f"{CSV_BASE}/val.csv",
        features_dir=FEATURES_DIR,
        label_col_name="label",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_weights=CLASS_WEIGHTS,
        worker_prefetch=WORKER_PREFETCH,
        features_name="cls_224x224",  # low resolution!
        coords_name="coords_224x224",
    )

    # Network: The very sketchy ViT-5/Small network
    config.net = LazyConfig(VitDensePreEmbedded)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        checkpoint_activations=CHECKPOINT_ACTIVATIONS,
        downproj=768,  # this is bad! but I need it for now to keep VRAM low because I have other processes running too.
    )

    # Lightning wrapper mappings
    config.lightning_wrapper_class = LazyConfig(WSIAttnWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
        training_crop_tokens=None,  # Don't crop here!
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
        job_group="baseline_vit5",
    )

    return config
