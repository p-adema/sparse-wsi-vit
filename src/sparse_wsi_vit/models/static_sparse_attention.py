"""
Block-sparse static attention for WSI slide encoding.

CLS tokens attend globally, patch tokens attend to CLS + spatially nearby
chunks (Hilbert-ordered).  Uses chunk-wise scaled_dot_product_attention
instead of FlexAttention to avoid torch.compile/checkpoint conflicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding


def hilbert_sort(coords: Tensor) -> Tensor:
    """Sort patches by 2-D Hilbert curve so spatially nearby patches are contiguous.

    Args:
        coords: ``(B, N, 2)`` pixel coordinates.

    Returns:
        ``(B, N)`` indices that reorder patches along the Hilbert curve.
    """
    B, N, _ = coords.shape
    bits = 16
    grid_size = 1 << bits

    cmin = coords.amin(dim=1, keepdim=True)
    cmax = coords.amax(dim=1, keepdim=True)
    span = (cmax - cmin).clamp(min=1).float()
    norm = ((coords - cmin).float() / span * (grid_size - 1)).long()

    x = norm[..., 0].clone()
    y = norm[..., 1].clone()
    d = torch.zeros(B, N, dtype=torch.long, device=coords.device)

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
    """Block-sparse static attention via chunk-wise SDPA.

    CLS tokens (first ``num_cls`` positions) attend to the full sequence.
    Patch tokens attend to CLS tokens plus ``window_size`` neighbouring
    chunks on each side.  Patches should be Hilbert-sorted beforehand so
    that sequence-local chunks correspond to spatially nearby patches.

    Chunks are defined in full-sequence space (position // chunk_size), so
    CLS tokens fall in chunk 0 together with the first patches.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens at the front of the sequence.
        window_size: Number of neighbouring chunks (on each side) that each
            patch chunk attends to.  ``0`` = same-chunk only + CLS.
        chunk_size: Number of positions per logical chunk.
        flex_block_size: Unused, kept for config compatibility.
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
        flex_block_size: int = 128,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) "
                f"(preferably embed_dim/num_heads = 64)"
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
            x: ``(batch_size, num_cls + patch_len, embed_dim)`` — CLS tokens prepended.
            coords: ``(batch_size, patch_len, 2)`` — patch pixel coordinates for 2-D RoPE.

        Returns:
            ``(batch_size, num_cls + patch_len, embed_dim)``
        """
        batch_size, seq_len, _ = x.shape
        num_cls = self.num_cls
        chunk_size = self.chunk_size
        window_size = self.window_size

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv.unbind(0)

        if coords is not None:
            q_patch = q[:, :, num_cls:].transpose(1, 2)
            k_patch = k[:, :, num_cls:].transpose(1, 2)
            q = torch.cat([q[:, :, :num_cls], self.rope(q_patch, coords).transpose(1, 2)], dim=2)
            k = torch.cat([k[:, :, :num_cls], self.rope(k_patch, coords).transpose(1, 2)], dim=2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        out = torch.empty_like(q)

        # CLS tokens: full attention over the entire sequence
        out[:, :, :num_cls] = F.scaled_dot_product_attention(
            q[:, :, :num_cls], k, v
        )

        # Patch tokens: chunk-wise windowed attention
        # Chunks are in full-sequence space: chunk c = positions [c*chunk_size, (c+1)*chunk_size)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for c in range(num_chunks):
            chunk_start = c * chunk_size
            chunk_end = min((c + 1) * chunk_size, seq_len)

            # Skip positions that are purely CLS (already handled above)
            patch_start = max(chunk_start, num_cls)
            if patch_start >= chunk_end:
                continue

            # KV window: chunks [lo, hi] in full-sequence space
            lo = max(0, c - window_size)
            hi = min(num_chunks - 1, c + window_size)
            kv_start = lo * chunk_size
            kv_end = min((hi + 1) * chunk_size, seq_len)

            k_window = k[:, :, kv_start:kv_end]
            v_window = v[:, :, kv_start:kv_end]

            # CLS tokens are always in the KV set; prepend if not already
            # in the window range (i.e. when the window doesn't reach chunk 0)
            if kv_start > 0:
                k_window = torch.cat([k[:, :, :num_cls], k_window], dim=2)
                v_window = torch.cat([v[:, :, :num_cls], v_window], dim=2)

            out[:, :, patch_start:chunk_end] = F.scaled_dot_product_attention(
                q[:, :, patch_start:chunk_end], k_window, v_window,
            )

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
