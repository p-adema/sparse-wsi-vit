"""ViT-5 classification config for debugging, training on first split from TCGA-like embeddings

Usage:
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_tcga.py
"""

import torch

from sparse_wsi_vit.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule
from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.lightning_wrappers.wsi_attn_wrapper import (
    WSIAttnWrapper,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig
from sparse_wsi_vit.models.vit5_dense import VitDensePreEmbedded

PIN_MEMORY = True

# ─── Data Details ──────────────────────────────────────────────
CSV_TRAIN_FOLD = "../splits/camelyon/full"
TARGET_NAME = "is_tumor"
TARGET_OPTIONS = ("normal", "tumor")
FEATURES_DIR = "../camelyon-emb"
DOWNSCALE_BLOCK = 1
FEATURES_NAME = "cls"  # "patches" or "cls"
FEATURES_SCALE = 224

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 1
NUM_WORKERS = 8 if FEATURES_SCALE == 224 else 1
WORKER_PREFETCH = 5
CLASS_WEIGHTS = True
IN_FEATURES = 1280
OUT_FEATURES = 1  # Binary tasks
PRECISION = "bf16-mixed"
CHECKPOINT_ACTIVATIONS = True

DEPTH = 3
HIDDEN_SIZE = 256
NUM_HEADS = 4
ROPE_DYNAMIC_HIGH = FEATURES_SCALE * DOWNSCALE_BLOCK

TRAINING_ITERATIONS = 1_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 5
PATIENCE = 10  # Early stopping


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

    # Network: The very sketchy ViT-5/Small network
    config.net = LazyConfig(VitDensePreEmbedded)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        checkpoint_activations=CHECKPOINT_ACTIVATIONS,
        downproj=HIDDEN_SIZE,
        rope_dynamic_high=ROPE_DYNAMIC_HIGH,
        num_heads=NUM_HEADS,
        depth=DEPTH,
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
    )

    # W&B Logging
    config.wandb = WandbConfig(
        project="wsi-classification", job_group="baseline_vit5", entity="dl2-2026"
    )

    return config
