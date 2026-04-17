"""
Basic tests for sparse_wsi_vit package.

These tests verify that the package can be imported and basic functionality
works.
"""

import pytest


def test_package_import() -> None:
    """Test that the package can be imported successfully."""
    import sparse_wsi_vit

    assert sparse_wsi_vit is not None


def test_version() -> None:
    """Test that the package has a version string."""
    import sparse_wsi_vit

    assert hasattr(sparse_wsi_vit, "__version__")
    assert isinstance(sparse_wsi_vit.__version__, str)


def test_torch_import() -> None:
    """Test that PyTorch is available (dependency check)."""
    import torch

    assert torch is not None
    assert hasattr(torch, "__version__")


def test_lazy_config_import() -> None:
    """Test that LazyConfig can be imported."""
    from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig, instantiate

    assert LazyConfig is not None
    assert instantiate is not None


def test_lazy_config_instantiate() -> None:
    """Test that LazyConfig can instantiate a simple torch module."""
    import torch
    from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig, instantiate

    cfg = LazyConfig(torch.nn.Linear)(in_features=10, out_features=5)
    layer = instantiate(cfg)
    assert isinstance(layer, torch.nn.Linear)
    assert layer.in_features == 10
    assert layer.out_features == 5


def test_schedulers_import() -> None:
    """Test that ChainedScheduler can be imported."""
    from sparse_wsi_vit.experiments.utils.schedulers import ChainedScheduler

    assert ChainedScheduler is not None


def test_default_cfg_import() -> None:
    """Test that experiment config classes can be imported."""
    from sparse_wsi_vit.experiments.default_cfg import (
        ExperimentConfig,
        SchedulerConfig,
        TrainConfig,
        WandbConfig,
    )

    assert ExperimentConfig is not None
    assert SchedulerConfig is not None
    assert TrainConfig is not None
    assert WandbConfig is not None


def test_experiment_config_creation() -> None:
    """Test that ExperimentConfig can be instantiated with defaults."""
    from sparse_wsi_vit.experiments.default_cfg import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.device == "cuda"
    assert cfg.seed == 0
    assert cfg.debug is True


def test_callbacks_import() -> None:
    """Test that callbacks can be imported."""
    from sparse_wsi_vit.experiments.callbacks.wandb_cache_cleanup import (
        WandbCacheCleanupCallback,
    )

    assert WandbCacheCleanupCallback is not None


def test_checkpointing_import() -> None:
    """Test that checkpointing utilities can be imported."""
    from sparse_wsi_vit.experiments.utils.checkpointing import (
        WandbSelectiveCheckpointUploader,
        load_checkpoint_state_dict,
    )

    assert WandbSelectiveCheckpointUploader is not None
    assert load_checkpoint_state_dict is not None


def test_cli_import() -> None:
    """Test that CLI utilities can be imported."""
    from sparse_wsi_vit.experiments.utils.cli import (
        apply_config_overrides,
        get_deterministic_run_name,
        load_config_from_file,
    )

    assert apply_config_overrides is not None
    assert get_deterministic_run_name is not None
    assert load_config_from_file is not None


def test_lightning_wrappers_import() -> None:
    """Test that lightning wrappers can be imported."""
    from sparse_wsi_vit.experiments.lightning_wrappers.base_lightning_wrapper import (
        LightningWrapperBase,
    )
    from sparse_wsi_vit.experiments.lightning_wrappers.classification_wrapper import (
        ClassificationWrapper,
    )

    assert LightningWrapperBase is not None
    assert ClassificationWrapper is not None


def test_always_passes() -> None:
    """A test that always passes - useful for debugging test infrastructure."""
    assert True
