"""Static Sparse Attention on raw RGB pixel features from the Mondriaan solid-colour control.

Solid-colour control where mean colour IS discriminative.
Expected to succeed, confirming that patterned failure is arrangement-specific.

Usage:
    MONDRIAAN_DATA_DIR=<path/to/solid_ps1> \\
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/mondriaan_solid_ssa.py
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
NUM_HEADS   = 4
NUM_LAYERS  = 3
NUM_CLS     = 2
WINDOW_SIZE = 4

GRADIENT_CHECKPOINTING=True #ToDo: @Lucius, check ff wat je hiermee wil.

DEPTH=3
HIDDEN_SIZE=256
NUM_HEADS=4            # embed_dim // num_heads=64
ROPE_DYNAMIC_HIGH = 220_000 # should work across the board, since coords are in pixel space

NUM_CLS=2
MLP_RATIO=4.0
PROJ_DROPOUT=0.2
DROP_PATH_RATE=0.1
LAYER_SCALE=True
INIT_SCALE=1e-4

CHUNK_SIZE=256           # patches per logical chunk (must be multiple of FLEX_BLOCK_SIZE)
FLEX_BLOCK_SIZE=128      # FlexAttention kernel tile size
ROPE_THETA=10_000.0
USE_HILBERT_SORT = True

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

    config.net=LazyConfig(SparseViT5SlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        depth=NUM_LAYERS,
        num_cls=NUM_CLS,
        sparse_attn="static",
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
        job_group="mondriaan_ssa_solid",
    )

    return config
