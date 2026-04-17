# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Default configuration for experiments with wsi_classification."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig


PLACEHOLDER = None


@dataclass
class TrainConfig:
    """Train configuration."""

    do: bool = True
    precision: str = "32-true"
    iterations: int = -1
    batch_size: int = -1
    grad_clip: float = 0.0
    track_grad_norm: int = -1  # -1 for no tracking
    accumulate_grad_steps: int = 1  # Accumulate gradient over different batches


@dataclass
class TrainerConfig:
    """Lightning Trainer configuration overrides."""

    # Check once every epoch by default.
    val_check_interval: float = 1.0

    # Run through all validation batches every epoch by default.
    limit_val_batches: Union[int, float] = 1.0


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    name: str = PLACEHOLDER
    warmup_iterations_percentage: float = 0.0
    stable_iterations_percentage: float = 0.0
    total_iterations: int = PLACEHOLDER
    mode: str = "max"
    monitor: Optional[str] = None  # in case we'd like to track e.g. val/iou


@dataclass
class WandbConfig:
    """Wandb configuration."""

    project: str = "wsi-classification"
    entity: str = ""

    job_group: str = ""


@dataclass
class AutoResumeConfig:
    """Auto-resume configuration via Weights & Biases run name.

    If enabled, the launcher will:
    - compute a stable run name (no timestamp; optionally includes username),
    - look up an existing W&B run with that exact name under the configured entity/project,
    - assert there is at most one such run,
    - download the checkpoint artifact for `alias` and resume Trainer from it.
    """

    enabled: bool = False
    alias: Literal["best", "latest"] = "latest"
    run_name: str | None = None


@dataclass
class ResumeFromCheckpointConfig:
    """Configuration to specify whether to start training from a previously saved checkpoint."""

    load: bool = False
    alias: Literal["best", "latest"] = "latest"
    strict: bool = True
    partial_load: bool = False
    run_path: str = ""
    output_dir: str = ".artifacts/{run_id}/{alias}"


@dataclass
class ExperimentConfig:
    """Default configuration for experiments with wsi_classification."""

    device: str = "cuda"
    debug: bool = True
    deterministic: bool = False
    seed: int = 0
    comment: str = ""
    compile: bool = False  # Whether to compile the model with torch.compile
    experiment_dir: Optional[str] = None
    num_nodes: int = 1

    dataset: LazyConfig = PLACEHOLDER
    net: LazyConfig = PLACEHOLDER
    lightning_wrapper_class: LazyConfig = PLACEHOLDER
    optimizer: LazyConfig = PLACEHOLDER

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    resume_from_checkpoint: ResumeFromCheckpointConfig = field(
        default_factory=ResumeFromCheckpointConfig
    )
    autoresume: AutoResumeConfig = field(default_factory=AutoResumeConfig)
    callbacks: list[LazyConfig] = field(default_factory=list)
