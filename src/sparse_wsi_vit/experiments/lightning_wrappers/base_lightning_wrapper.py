# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lightning wrappers for the Classification and Regression experiments."""

import warnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import grad_norm

import wandb
from sparse_wsi_vit.experiments.default_cfg import (
    PLACEHOLDER,
    ExperimentConfig,
    SchedulerConfig,
)
from sparse_wsi_vit.experiments.utils.lazy_config import LazyConfig
from sparse_wsi_vit.experiments.utils.schedulers import ChainedScheduler
from sparse_wsi_vit.experiments.utils.checkpointing import align_compiled_keys




def _get_layer_index(name: str, num_blocks: int) -> int:
    """Map a parameter name to a layer index for LLRD.

    Convention:
      - 0: patch_embed, cls_token, pos_embed, reg_token (embedding layer)
      - 1..num_blocks: blocks.0 .. blocks.<num_blocks-1>
      - num_blocks + 1: out_norm, out_proj (classification head)
    """
    if any(name.startswith(f"network.{prefix}") for prefix in ("patch_embed", "cls_token", "pos_embed", "reg_token")):
        return 0
    if "network.blocks." in name:
        block_str = name.split("network.blocks.")[1].split(".")[0]
        return int(block_str) + 1
    if any(name.startswith(f"network.{prefix}") for prefix in ("out_norm", "out_proj")):
        return num_blocks + 1
    # Fallback: treat unknown params as head-level (full LR)
    return num_blocks + 1


def _build_param_groups(
    model,
    default_weight_decay: float,
    layer_decay: float | None = None,
    num_blocks: int | None = None,
) -> list[dict]:
    """Partition model parameters into optimizer groups.

    Supports weight-decay grouping and optional layer-wise learning rate
    decay (LLRD).

    Weight-decay modes per parameter (set via custom attributes):
      - ``_no_weight_decay = True``  -> weight_decay = 0
      - ``_weight_decay = <float>``  -> weight_decay = <float> (custom group)
      - neither                      -> weight_decay = *default_weight_decay*

    A warning is emitted for any parameter with ``ndim <= 1`` (biases, norm
    weights, scales) that ends up with non-zero weight decay without an
    explicit ``_no_weight_decay`` flag.  This helps catch modules that
    forgot to mark their bias/norm parameters.

    When *layer_decay* is not None, each group also receives a per-layer
    ``lr`` scale factor so that deeper layers (closer to the head) get
    higher learning rates.  The scale for layer *i* out of *N* total
    layers is ``layer_decay ** (N - i)``.
    """
    if layer_decay is not None and num_blocks is None:
        raise ValueError("num_blocks must be set in the config when layer_decay (LLRD) is enabled.")
    seen_param_ids: set[int] = set()
    # group key = (wd_value, lr_scale) -> list of params
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}

    for name, param in model.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)

        # Determine weight decay
        custom_wd = getattr(param, "_weight_decay", None)
        if custom_wd is not None:
            wd = custom_wd
        elif getattr(param, "_no_weight_decay", False):
            wd = 0.0
        else:
            wd = default_weight_decay

        if param.ndim <= 1 and wd > 0 and not getattr(param, "_no_weight_decay", False):
            warnings.warn(
                f"Parameter '{name}' (shape {tuple(param.shape)}) has ndim <= 1 "
                f"and receives weight_decay={wd} without an explicit "
                f"_no_weight_decay flag. Set _no_weight_decay=True on this "
                f"parameter if it should be excluded from weight decay.",
                stacklevel=2,
            )

        # Determine LR scale
        if layer_decay is not None:
            num_layers = num_blocks + 2  # embedding + blocks + head
            layer_idx = _get_layer_index(name, num_blocks)
            lr_scale = layer_decay ** (num_layers - 1 - layer_idx)
        else:
            lr_scale = 1.0

        key = (wd, lr_scale)
        groups.setdefault(key, []).append(param)

    total_grouped = sum(len(ps) for ps in groups.values())
    assert len(seen_param_ids) == total_grouped, (
        "Optimizer param group mismatch: duplicate parameters across groups or some trainable "
        "parameters were not assigned. Every requires_grad=True parameter must appear in exactly one group."
    )

    parameters = []
    for wd, lr_scale in sorted(groups.keys(), key=lambda k: (k[1], k[0])):
        group = {"params": groups[(wd, lr_scale)], "weight_decay": wd}
        if layer_decay is not None:
            group["lr_scale"] = lr_scale
        parameters.append(group)

    return parameters



def construct_optimizer(
    model,
    optimizer_cfg: LazyConfig,
):
    """Constructs an optimizer for a given model given a configuration.

    Args:
        model: a list of parameters to be trained
        optimizer_cfg (LazyConfig): The optimizer configuration.

    Returns:
        torch.optim.Optimizer: The constructed optimizer.
    """
    # Create parameter groups based on weight decay flag
    # IMPORTANT: Avoid duplicates by iterating parameters ONCE at the top level
    # and tracking by object identity (id(param)).
    wd_params: list[torch.nn.Parameter] = []
    no_wd_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for name, param in model.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)
        if getattr(param, "_no_weight_decay", False):
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    # Safety: ensure no overlaps and no duplicates
    assert len(seen_param_ids) == len(set(map(id, wd_params))) + len(set(map(id, no_wd_params))), (
        "Optimizer param group mismatch: duplicate parameters across groups or some trainable "
        "parameters were not assigned. Every requires_grad=True parameter must appear in exactly one group."
    )

    # Create parameter groups with appropriate weight decay
    parameters = [
        {"params": wd_params, "weight_decay": optimizer_cfg.weight_decay},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]

    # OmegaConf has problems with non-serializable objects. To instantiate the optimizer, we need to do the following:
    # 1. Convert the optimizer config to a dictionary
    # 2. Import the optimizer class
    # 3. Instantiate the optimizer

    # 1. Convert the optimizer config to a dictionary
    _optim_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)

    # 2. Import the optimizer class
    _optimizer_cls = _optim_cfg.pop("__target__")
    module_path, class_name = _optimizer_cls.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    _optimizer_cls = getattr(module, class_name)

    # 3. Instantiate the optimizer with wd=0. Weight decay is calculated over the generated kernels.
    _optim_cfg["params"] = parameters
    optimizer = _optimizer_cls(**_optim_cfg)

    return optimizer


def construct_scheduler(
    optimizer,
    scheduler_cfg: SchedulerConfig,
):
    """Creates a learning rate scheduler for a given optimizer given a configuration.

    Args:
        optimizer: the optimizer to be used
        scheduler_cfg (SchedulerConfig): The scheduler configuration.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The constructed scheduler.
    """
    assert scheduler_cfg.name in [PLACEHOLDER, "cosine"], (
        f"scheduler_cfg.name must be either {PLACEHOLDER} or 'cosine'. Got {scheduler_cfg.name}"
    )
    if scheduler_cfg.name != PLACEHOLDER:
        assert scheduler_cfg.total_iterations != PLACEHOLDER, (
            f"scheduler_cfg.total_iterations must be set when scheduler_cfg.name is not {PLACEHOLDER}"
        )

    # Unpack values from scheduler_cfg
    scheduler_type = scheduler_cfg.name
    warmup_iterations_percentage = scheduler_cfg.warmup_iterations_percentage
    total_iterations = scheduler_cfg.total_iterations

    # Interpret fractional warmup as a percentage of total iterations
    assert warmup_iterations_percentage >= 0.0 and warmup_iterations_percentage < 1.0, (
        f"scheduler_cfg.warmup_iterations_percentage must be in [0.0, 1.0). Got {warmup_iterations_percentage}"
    )
    warmup_iterations = int(total_iterations * warmup_iterations_percentage)

    # Create warm_up scheduler
    if warmup_iterations != 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_iterations,
        )
    else:
        warmup_scheduler = None

    # Create main scheduler
    if scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            last_epoch=-warmup_iterations,
        )
    else:
        lr_scheduler = None
        warnings.warn(
            f"No scheduler will be used. cfg.train.scheduler = {scheduler_type}",
            stacklevel=2,
        )

    # Concatenate schedulers if required
    if warmup_scheduler is not None:
        # If both schedulers are defined, concatenate them
        if lr_scheduler is not None:
            lr_scheduler = ChainedScheduler(
                [
                    warmup_scheduler,
                    lr_scheduler,
                ]
            )
        # Otherwise, return only the warmup scheduler
        else:
            lr_scheduler = warmup_scheduler

    return lr_scheduler


class LightningWrapperBase(pl.LightningModule):
    """Base Lightning wrapper class."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
    ):
        """Initialize the LightningWrapperBase.

        Args:
            network: Network to wrap.
            cfg: Configuration.
        """
        super().__init__()
        # Define network
        self.network = network

        # Save optimizer & scheduler parameters
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        # Explicitly define whether we are in distributed mode.
        self.distributed = torch.cuda.device_count() > 1

        # Calculate the number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        self.num_params = num_params

        # Gradient tracking configuration
        self.should_track_grad_norm = cfg.train.track_grad_norm > 0
        self.grad_norm_interval = cfg.train.track_grad_norm

        # Placeholder for other outputs from the training and validation steps.
        self.other_outputs_train = []
        self.other_outputs_validation = []

    def forward(self, input_and_condition: dict[str, torch.Tensor]):
        """Forward pass of the network.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

        Returns:
            The output of the network.
        """
        return self.network(input_and_condition)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Patch checkpoint for cross-optimizer and compiled/non-compiled resume.

        Handles three mismatch scenarios:

        1. **state_dict key prefixes** — ``torch.compile`` wraps modules under
           ``_orig_mod``, so checkpoint keys may differ from the live model.

        2. **current_model_state key prefixes** — when EMA is active, Lightning
           saves the raw training weights under ``current_model_state``.  The
           EMA callback later calls ``pl_module.load_state_dict(...)`` with
           these keys, so they must also be remapped.

        3. **optimizer param-group keys** — resuming with a different optimizer
           (e.g. Apex FusedLAMB vs torch_optimizer.Lamb) may require injecting
           default values for keys the new optimizer expects but the old
           checkpoint lacks (like ``bias_correction``, ``adam_w_mode``, etc.).
        """
        model_keys = set(self.state_dict().keys())

        # --- 1. state_dict key remapping ----------------------------------
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = align_compiled_keys(state_dict, model_keys)

        # --- 2. current_model_state key remapping (EMA) -------------------
        current_model_state = checkpoint.get("current_model_state")
        if current_model_state is not None:
            checkpoint["current_model_state"] = align_compiled_keys(current_model_state, model_keys)

        # --- 3. optimizer param-group patching ----------------------------
        #
        # When resuming with a different optimizer (e.g. Apex FusedLAMB from
        # a torch_optimizer.Lamb checkpoint), the checkpoint's param groups
        # may be missing keys the new optimizer expects in its step().
        #
        # Strategy: construct a throwaway optimizer with the current config to
        # obtain its param groups with all keys correctly set, then inject any
        # missing keys into the checkpoint's groups.  This uses the *configured*
        # values (not just constructor defaults), so keys like max_grad_norm
        # that were explicitly overridden in the config are respected.
        #
        # Example: torch_optimizer.Lamb -> Apex FusedLAMB
        #
        #   Key              Injected value  Why correct
        #   ──────────────── ─────────────── ──────────────────────────────────
        #   bias_correction  True            Lamb applies it implicitly
        #   adam_w_mode      True            Both use decoupled weight decay
        #   max_grad_norm    0.0 (from cfg)  Avoids double-clipping with
        #                                    Lightning's grad_clip
        #   grad_averaging   True            FusedLAMB default (configured)
        #   set_grad_none    True            Memory opt, no semantic change
        #   use_nvlamb       False           Standard LAMB, not NVLAMB variant
        optimizer_states = checkpoint.get("optimizer_states")
        if optimizer_states is None:
            return

        try:
            reference_optim_dict = construct_optimizer(
                self,
                self.optimizer_cfg,
                layer_decay=self.layer_decay,
                num_blocks=self.num_blocks,
            )
            ref_group = reference_optim_dict.param_groups[0]
        except Exception:
            return

        for opt_state in optimizer_states:
            for group in opt_state.get("param_groups", []):
                for key, val in ref_group.items():
                    if key not in group and key != "params":
                        group[key] = val

    def forward(self, input_and_condition: dict[str, torch.Tensor]):
        """Forward pass of the network.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

        Returns:
            The output of the network.
        """
        return self.network(input_and_condition)

    # =========================================================================
    # Timing utilities for forward/backward pass measurement
    # =========================================================================
    def _start_timing(self):
        """Start timing for forward pass using CUDA events."""
        if self.training and torch.cuda.is_available():
            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_forward_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_start_event.record()

    def _record_forward_end(self):
        """Record the end of forward pass."""
        if self._cuda_start_event is not None:
            self._cuda_forward_end_event.record()

    def _record_backward_end_and_accumulate(self):
        """Record backward end time and accumulate timing stats."""
        if self._cuda_start_event is not None:
            self._cuda_backward_end_event = torch.cuda.Event(enable_timing=True)
            self._cuda_backward_end_event.record()
            torch.cuda.synchronize()

            # Calculate times in milliseconds
            forward_time_ms = self._cuda_start_event.elapsed_time(self._cuda_forward_end_event)
            backward_time_ms = self._cuda_forward_end_event.elapsed_time(self._cuda_backward_end_event)

            self._timing_forward_accum += forward_time_ms
            self._timing_backward_accum += backward_time_ms
            self._timing_step_count += 1

            # Reset events
            self._cuda_start_event = None

    def _log_timing_if_needed(self):
        """Log accumulated timing stats every N steps."""
        if (
            self._timing_step_count > 0
            and self._timing_step_count % self.timing_log_interval == 0
            and self.logger is not None
        ):
            avg_forward_ms = self._timing_forward_accum / self._timing_step_count
            avg_backward_ms = self._timing_backward_accum / self._timing_step_count
            avg_total_ms = avg_forward_ms + avg_backward_ms

            self.log("timing/forward_ms", avg_forward_ms, prog_bar=False, sync_dist=self.distributed)
            self.log("timing/backward_ms", avg_backward_ms, prog_bar=False, sync_dist=self.distributed)
            self.log("timing/step_total_ms", avg_total_ms, prog_bar=False, sync_dist=self.distributed)
            self.log(
                "timing/throughput_steps_per_sec", 1000.0 / avg_total_ms, prog_bar=False, sync_dist=self.distributed
            )

            # Reset accumulators after logging
            self._timing_forward_accum = 0.0
            self._timing_backward_accum = 0.0
            self._timing_step_count = 0

    def on_before_backward(self, loss: torch.Tensor) -> None:
        """Called before backward pass - record forward end time."""
        self._record_forward_end()

    def on_after_backward(self) -> None:
        """Called after backward pass - record timing and log."""
        self._record_backward_end_and_accumulate()
        self._log_timing_if_needed()


    def configure_optimizers(self):
        """Configure the optimizer and scheduler for training."""
        # Construct optimizer & scheduler
        optimizer = construct_optimizer(
            model=self,
            optimizer_cfg=self.optimizer_cfg,
        )
        scheduler = construct_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.scheduler_cfg,
        )
        # Construct output dictionary
        optim_dict = {"optimizer": optimizer}
        if scheduler is not None:
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        # Return
        return optim_dict

    def on_before_optimizer_step(self, optimizer):
        """Log the gradient norm before the optimizer step every `grad_norm_interval` steps."""
        if self.should_track_grad_norm and self.global_step % self.grad_norm_interval == 0:
            self.log_dict(grad_norm(self, norm_type=2))

    def on_fit_start(self):
        """Log the model architecture to Weights & Biases once training starts."""
        super().on_fit_start()

        if self.logger is not None:
            model_repr = str(self.network)
            # Log as HTML wrapped in <pre> to preserve formatting in the UI.
            self.logger.experiment.log(
                {
                    "model/architecture": wandb.Html(f"<pre>{model_repr}</pre>"),
                    "global_step": self.global_step,
                }
            )

            # Also send to raw logs (stdout captured by W&B) and W&B terminal log
            self.print(f"Model architecture:\n{model_repr}")
            wandb.termlog(f"Model architecture:\n{model_repr}")
