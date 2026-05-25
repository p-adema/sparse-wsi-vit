"""
Approach A: Block-sparse static attention for WSI slide encoding.

Uses custom Triton block-sparse FlashAttention kernels for patch tokens
and standard SDPA for CLS tokens.  Compatible with torch.utils.checkpoint
(no torch.compile dependency).

CLS tokens attend globally, patch tokens attend to CLS + spatially nearby
chunks (Hilbert-ordered).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding
from sparse_wsi_vit.models.static_sparse_kernels import (
    build_block_mask,
    block_sparse_attention,
)


def hilbert_sort(coords: Tensor) -> Tensor:
    """Sort patches by 2-D Hilbert curve so spatially nearby patches are contiguous.

    Args:
        coords: ``(B, N, 2)`` pixel coordinates.

    Returns:
        ``(B, N)`` indices that reorder patches along the Hilbert curve.
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
    """Block-sparse static attention using custom Triton kernels.

    CLS tokens (first ``num_cls`` positions) attend to the full sequence
    via standard SDPA.  Patch tokens attend to CLS tokens plus ``window_size``
    neighbouring chunks on each side via a Triton block-sparse FlashAttention
    kernel.  Compatible with ``torch.utils.checkpoint`` (no ``torch.compile``).

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens at the front of the sequence.
        window_size: Number of neighbouring chunks (on each side) that each
            patch chunk attends to.  ``0`` = same-chunk only + CLS.
        chunk_size: Number of Hilbert-sorted patches per logical chunk.
            Must be a multiple of ``block_size``.
        block_size: Triton kernel tile size.  Hardware-dependent;
            128 is optimal for H100.
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
        block_size: int = 128,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if chunk_size % block_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be a multiple of "
                f"block_size ({block_size})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.head_dim = embed_dim // num_heads

        self.rope = VisionRotaryEmbedding(
            self.head_dim // 2,
            dynamic=True,
            coord_high=rope_coord_high,
            theta=rope_theta,
        )
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Cache for block mask — same seq_len and device reuses the mask
        self._cached_mask_key: tuple[int, torch.device] | None = None
        self._cached_kv_indices: Tensor | None = None

    def _get_kv_indices(self, seq_len: int, device: torch.device) -> Tensor:
        """Get or build the KV-block index array, cached by (seq_len, device)."""
        key = (seq_len, device)
        if self._cached_mask_key != key:
            self._cached_kv_indices = build_block_mask(
                seq_len=seq_len,
                num_cls=self.num_cls,
                chunk_size=self.chunk_size,
                window_size=self.window_size,
                block_size=self.block_size,
                device=device,
            )
            self._cached_mask_key = key
        return self._cached_kv_indices

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        """Block-sparse attention over a CLS-prepended patch sequence.

        CLS queries attend to the full sequence via SDPA.  Patch queries
        attend block-sparsely to CLS + windowed chunks via the Triton kernel.

        Args:
            x: ``(batch_size, num_cls + patch_len, embed_dim)`` — CLS tokens prepended.
            coords: ``(batch_size, patch_len, 2)`` — patch pixel coordinates for 2-D RoPE.

        Returns:
            ``(batch_size, num_cls + patch_len, embed_dim)``
        """
        batch_size, seq_len, _ = x.shape
        num_cls = self.num_cls

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)  # each: (batch, heads, seq_len, head_dim), contiguous

        # --- Apply 2-D RoPE to patch tokens only ---
        if coords is not None:
            # rope expects (batch, patch_len, num_heads, head_dim)
            q_patch_for_rope = q[:, :, num_cls:].transpose(1, 2)
            k_patch_for_rope = k[:, :, num_cls:].transpose(1, 2)
            q_patch_roped = self.rope(q_patch_for_rope, coords).transpose(1, 2)
            k_patch_roped = self.rope(k_patch_for_rope, coords).transpose(1, 2)
            q = torch.cat([q[:, :, :num_cls], q_patch_roped], dim=2)
            k = torch.cat([k[:, :, :num_cls], k_patch_roped], dim=2)

        # --- CLS attention: global over full sequence via SDPA ---
        q_cls = q[:, :, :num_cls]  # (batch, heads, num_cls, head_dim)
        cls_out = F.scaled_dot_product_attention(q_cls, k, v)

        # --- Patch attention: block-sparse via Triton kernel ---
        # q_patch is a slice of q (contiguous after cat, but offset by num_cls).
        # Make it contiguous so the Triton kernel gets standard strides.
        q_patch = q[:, :, num_cls:].contiguous()  # (batch, heads, num_patches, head_dim)
        kv_indices = self._get_kv_indices(seq_len, x.device)
        patch_out = block_sparse_attention(
            q_patch, k, v, kv_indices, seq_len, self.block_size,
        )

        # --- Concatenate and project ---
        out = torch.cat([cls_out, patch_out], dim=2)  # (batch, heads, seq_len, head_dim)
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
            block_size=flex_block_size,
            rope_theta=rope_theta,
            rope_coord_high=rope_coord_high,
        )

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        return self.attn(x, coords)
