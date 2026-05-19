"""Static Sparse Attention on raw RGB pixel features from the Mondriaan patterned benchmark.

CLS tokens attend globally; patch tokens attend to CLS + local window of ±window_size
sequence neighbours. No positional encoding — sparsity pattern is index-based.

Usage:
    MONDRIAAN_DATA_DIR=<path/to/patterned_ps1> \\
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/mondriaan_ssa.py
"""

import os
import torch

from sparse_wsi_vit.experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig
from sparse_wsi_vit.models.static_sparse_attention import StaticSparseViTSlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.mondriaan_h5_datamodule import MondriaanH5DataModule

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.environ["MONDRIAAN_DATA_DIR"]
IN_FEATURES  = 3
OUT_FEATURES = 2

# ─── Model ───────────────────────────────────────────────────────────────────
EMBED_DIM   = 256
NUM_HEADS   = 4      # 256 / 4 = 64 dim per head
NUM_LAYERS  = 3
NUM_CLS     = 2
WINDOW_SIZE = 2
DILATION    = 1

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
NUM_WORKERS   = 4
PRECISION     = "bf16-mixed"

TRAINING_ITERATIONS          = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE                = 2e-4
WEIGHT_DECAY                 = 1e-4
GRAD_CLIP                    = 1.0
ACCUMULATE_GRAD_STEPS        = 1


def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    config.seed  = 42

    config.dataset = LazyConfig(MondriaanH5DataModule)(
        data_dir=DATA_DIR,
        in_features=IN_FEATURES,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    config.net = LazyConfig(StaticSparseViTSlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_cls=NUM_CLS,
        window_size=WINDOW_SIZE,
        dilation=DILATION,
    )

    config.lightning_wrapper_class = LazyConfig(MILWrapper)(
        use_bce_loss=(OUT_FEATURES == 1),
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

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations=TRAINING_ITERATIONS,
        mode="max",
    )

    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="mondriaan_ssa_patterned",
    )

    return config
