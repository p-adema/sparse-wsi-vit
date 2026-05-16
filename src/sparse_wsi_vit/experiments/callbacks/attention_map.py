"""Callback to visualise block-sparse attention patterns during validation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from sparse_wsi_vit.models.static_sparse_attention import (
    StaticSparseAttention,
    hilbert_sort,
)


class AttentionMapCallback(pl.callbacks.Callback):
    """Logs block-sparse attention visualisations to W&B.

    Two plots per invocation:

    1. **Block mask** — heatmap of which flex-blocks attend to which,
       verifying the sparse structure is correct.
    2. **CLS spatial attention** — patch coordinates coloured by the
       average CLS attention weight (requires ``coords`` in the batch).

    Args:
        every_n_epochs: Log every N validation epochs.
        layer_index: Transformer block to capture CLS attention from
            (default ``-1`` = last block).
    """

    def __init__(self, every_n_epochs: int = 1, layer_index: int = -1) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.layer_index = layer_index
        self._captured_qkv: Tensor | None = None
        self._hook_handle = None

    @staticmethod
    def _find_sparse_attn(
        network: torch.nn.Module,
    ) -> list[StaticSparseAttention]:
        return [
            m for m in network.modules() if isinstance(m, StaticSparseAttention)
        ]

    def _should_run(self, trainer: pl.Trainer, batch_idx: int) -> bool:
        return (
            batch_idx == 0
            and trainer.current_epoch % self.every_n_epochs == 0
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_run(trainer, batch_idx):
            return

        modules = self._find_sparse_attn(pl_module.network)
        if not modules:
            return

        target = modules[self.layer_index]

        def hook(_module, _input, output):
            self._captured_qkv = output.detach()

        self._hook_handle = target.qkv.register_forward_hook(hook)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_run(trainer, batch_idx):
            return

        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        modules = self._find_sparse_attn(pl_module.network)
        if not modules:
            return

        target = modules[self.layer_index]
        patch_len = batch["input"].shape[1]
        seq_len = patch_len + target.num_cls

        mask_fig = self._plot_block_mask(target, seq_len)

        cls_fig = None
        hilbert_fig = None
        coords = batch.get("coords")
        if coords is not None:
            hilbert_fig = self._plot_hilbert_curve(coords, target.chunk_size)
            if self._captured_qkv is not None:
                cls_fig = self._plot_cls_attention(target, coords)

        self._captured_qkv = None
        self._log_figures(trainer, mask_fig, cls_fig, hilbert_fig)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_figures(
        trainer: pl.Trainer,
        mask_fig: plt.Figure,
        cls_fig: plt.Figure | None,
        hilbert_fig: plt.Figure | None = None,
    ) -> None:
        figs = [mask_fig, cls_fig, hilbert_fig]
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            for f in figs:
                if f is not None:
                    plt.close(f)
            return

        try:
            import wandb

            log_dict: dict = {"val/block_mask": wandb.Image(mask_fig)}
            if cls_fig is not None:
                log_dict["val/cls_spatial_attention"] = wandb.Image(cls_fig)
            if hilbert_fig is not None:
                log_dict["val/hilbert_curve"] = wandb.Image(hilbert_fig)
            logger.experiment.log(log_dict, step=trainer.global_step)
        finally:
            for f in figs:
                if f is not None:
                    plt.close(f)

    # ------------------------------------------------------------------
    # Block mask plot
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_block_mask(
        attn: StaticSparseAttention, seq_len: int
    ) -> plt.Figure:
        num_cls = attn.num_cls
        chunk_size = attn.chunk_size
        window_size = attn.window_size
        flex_block_size = attn.flex_block_size

        num_flex_blocks = (seq_len + flex_block_size - 1) // flex_block_size
        blocks_per_chunk = chunk_size // flex_block_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        num_cls_blocks = (
            (num_cls + flex_block_size - 1) // flex_block_size
            if num_cls > 0
            else 0
        )

        mask = np.zeros((num_flex_blocks, num_flex_blocks), dtype=np.float32)
        for qb in range(num_flex_blocks):
            if qb * flex_block_size < num_cls:
                mask[qb, :] = 1.0
                continue
            for cb in range(num_cls_blocks):
                mask[qb, cb] = 1.0
            q_chunk = qb // blocks_per_chunk
            lo = max(0, q_chunk - window_size)
            hi = min(num_chunks - 1, q_chunk + window_size)
            for c in range(lo, hi + 1):
                s = c * blocks_per_chunk
                e = min((c + 1) * blocks_per_chunk, num_flex_blocks)
                mask[qb, s:e] = 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mask, cmap="Blues", interpolation="nearest", aspect="equal")
        ax.set_xlabel("KV block")
        ax.set_ylabel("Q block")
        ax.set_title(
            f"Block mask ({num_flex_blocks}×{num_flex_blocks})\n"
            f"seq={seq_len}  cls={num_cls}  chunk={chunk_size}  window={window_size}"
        )

        for c in range(1, num_chunks):
            pos = c * blocks_per_chunk - 0.5
            if pos < num_flex_blocks:
                ax.axhline(y=pos, color="red", linewidth=0.5, alpha=0.5)
                ax.axvline(x=pos, color="red", linewidth=0.5, alpha=0.5)

        if num_cls_blocks > 0:
            b = num_cls_blocks - 0.5
            ax.axhline(y=b, color="green", linewidth=1, linestyle="--")
            ax.axvline(x=b, color="green", linewidth=1, linestyle="--")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Hilbert curve plot
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_hilbert_curve(
        coords: Tensor,
        chunk_size: int,
    ) -> plt.Figure:
        coords_dev = coords[:1]
        sort_idx = hilbert_sort(coords_dev)
        orig = coords[0].cpu().numpy()
        order = sort_idx[0].cpu().numpy()
        sorted_xy = orig[order]
        num_patches = len(order)
        num_chunks = (num_patches + chunk_size - 1) // chunk_size

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: patches colored by Hilbert position
        ax = axes[0]
        scatter = ax.scatter(
            orig[:, 0],
            orig[:, 1],
            c=np.arange(num_patches)[np.argsort(order)],
            cmap="viridis",
            s=2,
            alpha=0.8,
        )
        fig.colorbar(scatter, ax=ax, label="Hilbert position")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(f"Hilbert sort order ({num_patches} patches)")
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # Right: patches colored by chunk assignment
        ax = axes[1]
        chunk_ids = np.zeros(num_patches, dtype=np.int32)
        for i, idx in enumerate(order):
            chunk_ids[idx] = i // chunk_size
        scatter = ax.scatter(
            orig[:, 0],
            orig[:, 1],
            c=chunk_ids,
            cmap="tab20",
            s=2,
            alpha=0.8,
        )
        fig.colorbar(scatter, ax=ax, label="Chunk ID")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(
            f"Chunk assignment ({num_chunks} chunks of {chunk_size})"
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()

        fig.suptitle("Hilbert space-filling curve", fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CLS spatial attention plot
    # ------------------------------------------------------------------

    def _plot_cls_attention(
        self,
        attn: StaticSparseAttention,
        coords: Tensor,
    ) -> plt.Figure:
        num_cls = attn.num_cls
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        device = self._captured_qkv.device

        qkv_out = self._captured_qkv
        batch_size, seq_len, _ = qkv_out.shape

        qkv = qkv_out.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)

        # Hilbert-sort coords (matching model forward) and apply RoPE
        coords_dev = coords[:1].to(device)
        sort_idx = hilbert_sort(coords_dev)
        sorted_coords = coords_dev.gather(
            1, sort_idx.unsqueeze(-1).expand(-1, -1, 2)
        )

        with torch.no_grad():
            q_patch = attn.rope(q[:1, :, num_cls:].transpose(1, 2), sorted_coords).transpose(1, 2)
            k_patch = attn.rope(k[:1, :, num_cls:].transpose(1, 2), sorted_coords).transpose(1, 2)

            q_full = torch.cat([q[:1, :, :num_cls], q_patch], dim=2)
            k_full = torch.cat([k[:1, :, :num_cls], k_patch], dim=2)

            scale = head_dim**-0.5
            cls_attn = torch.softmax(
                q_full[:, :, :num_cls] @ k_full.transpose(-2, -1) * scale,
                dim=-1,
            )

        # Average over heads and CLS tokens, patch positions only
        patch_attn = cls_attn[0, :, :, num_cls:]
        mean_attn = patch_attn.mean(dim=(0, 1)).cpu().numpy()

        inv_sort = sort_idx[0].argsort().cpu().numpy()
        mean_attn_orig = mean_attn[inv_sort]

        orig_coords = coords[0].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            orig_coords[:, 0],
            orig_coords[:, 1],
            c=mean_attn_orig,
            cmap="hot",
            s=2,
            alpha=0.8,
        )
        fig.colorbar(scatter, ax=ax, label="Attention weight")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(
            f"CLS spatial attention (layer {self.layer_index})\n"
            f"avg over {num_cls} CLS × {num_heads} heads"
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig
