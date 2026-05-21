"""
Block-sparse static attention for WSI slide encoding.

CLS tokens attend globally, patch tokens attend to CLS + spatially nearby
chunks (Hilbert-ordered).  Uses chunk-wise scaled_dot_product_attention
instead of FlexAttention — compatible with gradient checkpointing.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding

logger = logging.getLogger(__name__)


def _check_nan(tensor: Tensor, label: str) -> bool:
    """Log and return True when *tensor* contains NaN or Inf."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.error(
            "NaN/Inf at %s | nan=%d inf=%d total=%d shape=%s dtype=%s",
            label, nan_count, inf_count, tensor.numel(),
            tuple(tensor.shape), tensor.dtype,
        )
        return True
    return False


def hilbert_sort(coords: Tensor) -> Tensor:
    """Sort patches by 2-D Hilbert curve so spatially nearby patches are contiguous.

    Args:
        coords: ``(batch_size, num_patches, 2)`` pixel coordinates.

    Returns:
        ``(batch_size, num_patches)`` indices that reorder patches along the Hilbert curve.
    """
    batch_size, num_patches, _ = coords.shape
    bits = 16
    grid_size = 1 << bits

    cmin = coords.amin(dim=1, keepdim=True)
    cmax = coords.amax(dim=1, keepdim=True)
    span = (cmax - cmin).clamp(min=1).float()
    norm = ((coords - cmin).float() / span * (grid_size - 1)).long()

    x = norm[..., 0].clone()
    y = norm[..., 1].clone()
    d = torch.zeros(batch_size, num_patches, dtype=torch.long, device=coords.device)

    s = grid_size >> 1
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d += s * s * ((3 * rx) ^ ry)
        no_ry = ry == 0
        flip = no_ry & (rx == 1)
        x = torch.where(flip, s - 1 - x, x)
        y = torch.where(flip, s - 1 - y, y)
        x_new = torch.where(no_ry, y, x)
        y_new = torch.where(no_ry, x, y)
        x, y = x_new, y_new
        s >>= 1

    return d.argsort(dim=1)


class StaticSparseAttention(nn.Module):
    """Block-sparse static attention using chunk-wise SDPA.

    CLS tokens (first ``num_cls`` positions) attend to the full sequence.
    Patch tokens attend to CLS tokens plus ``window_size`` neighbouring
    chunks on each side.  Patches should be Hilbert-sorted beforehand so
    that sequence-local chunks correspond to spatially nearby patches.

    Chunks are defined in *full-sequence* space (position // chunk_size),
    so CLS tokens fall in chunk 0 together with the first patches.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens at the front of the sequence.
        window_size: Number of neighbouring chunks (on each side) that each
            patch chunk attends to.  ``0`` = same-chunk only + CLS.
        chunk_size: Number of Hilbert-sorted patches per logical chunk.
        flex_block_size: Accepted for config compatibility; unused.
        rope_theta: Base frequency for 2-D RoPE.
        rope_coord_high: Coordinate normalisation divisor for 2-D RoPE.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_cls: int = 1,
        window_size: int = 1,
        chunk_size: int = 256,
        flex_block_size: int = 128,  # kept for config compat, unused
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.head_dim = embed_dim // num_heads

        self.rope = VisionRotaryEmbedding(
            self.head_dim // 2,
            dynamic=True,
            coord_high=rope_coord_high,
            theta=rope_theta,
        )
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        """Block-sparse attention over a CLS-prepended patch sequence.

        Args:
            x: ``(batch_size, num_cls + patch_len, embed_dim)`` — CLS tokens
                already prepended by the slide encoder.
            coords: ``(batch_size, patch_len, 2)`` — patch pixel coordinates
                for 2-D RoPE.

        Returns:
            ``(batch_size, num_cls + patch_len, embed_dim)``
        """
        batch_size, seq_len, _ = x.shape
        num_cls = self.num_cls
        chunk_size = self.chunk_size
        window_size = self.window_size

        # --- QKV projection ---
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv.unbind(0)

        # --- 2-D RoPE on patch tokens only ---
        if coords is not None:
            q_patch = q[:, :, num_cls:].transpose(1, 2)   # (batch, patch_len, heads, head_dim)
            k_patch = k[:, :, num_cls:].transpose(1, 2)
            q_patch = self.rope(q_patch, coords).transpose(1, 2)  # back to (batch, heads, patch_len, head_dim)
            k_patch = self.rope(k_patch, coords).transpose(1, 2)
            q = torch.cat([q[:, :, :num_cls], q_patch], dim=2)
            k = torch.cat([k[:, :, :num_cls], k_patch], dim=2)

        # --- QK-norm (stabilises softmax at large scale) ---
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --- Chunk-wise SDPA ---
        # Shape: q, k, v are (batch, heads, seq_len, head_dim)
        out = torch.zeros_like(q)

        # CLS tokens: full attention over the entire sequence
        out[:, :, :num_cls] = F.scaled_dot_product_attention(
            q[:, :, :num_cls], k, v,
        )

        # Patch tokens: each chunk attends to CLS + neighbouring chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
            # Skip if this chunk only contains CLS tokens (already handled)
            patch_start = max(chunk_start, num_cls)
            if patch_start >= chunk_end:
                continue

            # KV window: neighbouring chunks
            lo = max(0, chunk_idx - window_size)
            hi = min(num_chunks - 1, chunk_idx + window_size)
            kv_start = lo * chunk_size
            kv_end = min((hi + 1) * chunk_size, seq_len)

            k_window = k[:, :, kv_start:kv_end]
            v_window = v[:, :, kv_start:kv_end]

            # Prepend CLS keys/values if not already in the window range
            if kv_start > 0:
                k_window = torch.cat([k[:, :, :num_cls], k_window], dim=2)
                v_window = torch.cat([v[:, :, :num_cls], v_window], dim=2)

            out[:, :, patch_start:chunk_end] = F.scaled_dot_product_attention(
                q[:, :, patch_start:chunk_end], k_window, v_window,
            )

        # --- Output projection ---
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class StaticSparseAttentionAdapter(nn.Module):
    """Adapts StaticSparseAttention to the ViT-5 Block ``Attention_block`` API.

    ``Block.__init__`` calls ``Attention_block(dim, num_heads=..., attn_drop=...,
    flash=..., rope_size=..., ...)`` with ViT-5-specific kwargs.  This adapter
    accepts the relevant subset and silently absorbs the rest via ``**_``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        num_cls: int = 1,
        window_size: int = 1,
        chunk_size: int = 256,
        flex_block_size: int = 128,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
        **_,
    ) -> None:
        super().__init__()
        self.attn = StaticSparseAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_cls=num_cls,
            window_size=window_size,
            chunk_size=chunk_size,
            flex_block_size=flex_block_size,
            rope_theta=rope_theta,
            rope_coord_high=rope_coord_high,
        )

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        return self.attn(x, coords)
