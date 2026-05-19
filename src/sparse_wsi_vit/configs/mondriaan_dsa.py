"""DeepSeek Sparse Attention on raw RGB pixel features from the Mondriaan patterned benchmark.

Content-based top-k routing: lightweight indexer selects top_k keys per query.
No positional encoding — routing is fully content-driven.

Note: top_k=64 approaches N at ps4 (N=64); if the Triton kernel errors reduce to top_k=32.

Usage:
    MONDRIAAN_DATA_DIR=<path/to/patterned_ps1> \\
    uv run experiments/run.py --config src/sparse_wsi_vit/configs/mondriaan_dsa.py
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
from sparse_wsi_vit.models.deepseek_sparse_attention import DSAViTSlideEncoder
from sparse_wsi_vit.experiments.lightning_wrappers.mil_wrapper import MILWrapper
from sparse_wsi_vit.experiments.datamodules.mondriaan_h5_datamodule import MondriaanH5DataModule

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.environ["MONDRIAAN_DATA_DIR"]
IN_FEATURES  = 3
OUT_FEATURES = 2

# ─── Model ───────────────────────────────────────────────────────────────────
EMBED_DIM     = 256
NUM_HEADS     = 4      # 256 / 4 = 64 dim per head
NUM_LAYERS    = 3
NUM_CLS       = 2

INDEXER_HEADS = 4
INDEXER_DIM   = 32
TOP_K         = 64     # ~6% at ps1 N=1024; ~25% at ps2 N=256
BLOCK_Q       = 16
BLOCK_K       = 32
BLOCK_D       = 32

# ─── Optimisation ────────────────────────────────────────────────────────────
BATCH_SIZE    = 64     # conservative; Triton kernels at small N untested
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

    config.net = LazyConfig(DSAViTSlideEncoder)(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_cls=NUM_CLS,
        indexer_heads=INDEXER_HEADS,
        indexer_dim=INDEXER_DIM,
        top_k=TOP_K,
        block_q=BLOCK_Q,
        block_k=BLOCK_K,
        block_d=BLOCK_D,
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
        job_group="mondriaan_dsa_patterned",
    )

    return config
