"""Example classification config — demonstrates the get_config() pattern.

This is a minimal placeholder config. Replace the dataset and network
with real WSI data modules and models as they are implemented.

Usage:
    python -m wsi_classification.experiments.run --config configs/example_classification.py
"""

import torch

from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from sparse_wsi_vit.experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig


PLACEHOLDER = None

# ─── Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 2  # e.g., MSI-high vs MSS
PRECISION = "bf16-mixed"

TRAINING_ITERATIONS = 10_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def get_config() -> ExperimentConfig:
    """Return an example WSI classification configuration.

    NOTE: dataset and net are PLACEHOLDER — replace with real modules.
    """
    config = ExperimentConfig()
    config.debug = True
    config.seed = 42

    # Dataset — PLACEHOLDER: replace with your WSI datamodule
    config.dataset = PLACEHOLDER

    # Network — PLACEHOLDER: replace with your classification model
    config.net = PLACEHOLDER

    # Lightning wrapper
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    # Optimizer
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
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

    # W&B
    config.wandb = WandbConfig(
        project="wsi-classification",
        job_group="example",
    )

    return config
