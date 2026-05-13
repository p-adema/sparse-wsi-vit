"""
Approach A: Block-sparse static attention for WSI slide encoding.

Uses FlexAttention with a static block mask: CLS tokens attend globally,
patch tokens attend to CLS + spatially nearby chunks (Hilbert-ordered).
"""

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, BlockMask

compiled_flex_attention = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "TRITON"}),
    dynamic=True,
)


def build_block_mask(
    seq_len: int,
    num_cls: int,
    chunk_size: int,
    window_size: int,
    flex_block_size: int,
    device: torch.device,
    mask_mod,
) -> BlockMask:
    """Construct BlockMask

    Chunks are defined from position 0 (not from num_cls), so chunk boundaries
    land on exact multiples of chunk_size — which is itself a multiple of
    flex_block_size. This guarantees perfect alignment between logical chunks
    and FlexAttention kernel tiles.

    CLS tokens occupy the first positions of chunk 0.  The mask_mod handles
    fine-grained masking: CLS attends globally, patches attend to CLS +
    chunks within the window.
    """
    num_blocks = (seq_len + flex_block_size - 1) // flex_block_size
    blocks_per_chunk = chunk_size // flex_block_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    num_cls_flex_blocks = (num_cls + flex_block_size - 1) // flex_block_size if num_cls > 0 else 0

    all_kv_lists: list[list[int]] = []
    for qb in range(num_blocks):
        q_start = qb * flex_block_size

        if q_start < num_cls:
            kv_list = list(range(num_blocks))
        else:
            kv_set: set[int] = set()
            for cb in range(num_cls_flex_blocks):
                kv_set.add(cb)

            q_chunk = qb // blocks_per_chunk
            win_lo = max(0, q_chunk - window_size)
            win_hi = min(num_chunks - 1, q_chunk + window_size)

            for c in range(win_lo, win_hi + 1):
                fb_start = c * blocks_per_chunk
                fb_end = min((c + 1) * blocks_per_chunk, num_blocks)
                for fb in range(fb_start, fb_end):
                    kv_set.add(fb)

            kv_list = sorted(kv_set)

        all_kv_lists.append(kv_list)

    max_kv = max(len(kv) for kv in all_kv_lists)

    kv_num_blocks = torch.zeros(1, 1, num_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(1, 1, num_blocks, max_kv, dtype=torch.int32, device=device)

    for qb, kv_list in enumerate(all_kv_lists):
        kv_num_blocks[0, 0, qb] = len(kv_list)
        for i, kvb in enumerate(kv_list):
            kv_indices[0, 0, qb, i] = kvb

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=(flex_block_size, flex_block_size),
        mask_mod=mask_mod,
        seq_lengths=(seq_len, seq_len),
    )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent pairs: [..., x1, x2, ...] → [..., -x2, x1, ...]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


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



class Rope2D(nn.Module):
    """2-D Rotary Position Embedding for patch tokens.

    Applies 1-D RoPE independently along the x and y coordinate axes.
    ``head_dim`` is split evenly: the first half encodes x, the second
    half encodes y.

    Args:
        head_dim: Per-head feature dimension. Must be divisible by 4.
        theta: RoPE base frequency.
        coord_high: Coordinate normalisation divisor.  Raw pixel coords
            are divided by this value before computing frequencies.
            Default: 100_000 — matches ViT-5 ``rope_dynamic_high`` and
            is suitable for WSI pixel-level coordinates.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 10_000.0,
        coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
            )
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer("inv_freq", inv_freq)  # (half // 2,)
        self.coord_high = coord_high

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        """Rotate Q or K using 2-D spatial coordinates.

        Args:
            x: ``(batch_size, num_heads, patch_len, head_dim)`` — patch queries or keys.
            coords: ``(batch_size, patch_len, 2)`` — (x, y) pixel coordinates for each patch.

        Returns:
            Rotated tensor, same shape as ``x``.
        """
        xy = coords.float() / self.coord_high  # (batch_size, patch_len, 2)

        # Per-token frequency vectors for each axis: (batch_size, patch_len, half//2)
        freq_x = torch.einsum("bl, f -> blf", xy[..., 0], self.inv_freq)
        freq_y = torch.einsum("bl, f -> blf", xy[..., 1], self.inv_freq)

        # Repeat each freq to align with rotate_half on adjacent pairs
        freq_x = freq_x.repeat_interleave(2, dim=-1)  # (batch_size, patch_len, half)
        freq_y = freq_y.repeat_interleave(2, dim=-1)  # (batch_size, patch_len, half)

        freqs = torch.cat([freq_x, freq_y], dim=-1)   # (batch_size, patch_len, head_dim)
        cos = freqs.cos().unsqueeze(1)  # (batch_size, 1, patch_len, head_dim)
        sin = freqs.sin().unsqueeze(1)

        return x * cos + _rotate_half(x) * sin


class StaticSparseAttention(nn.Module):
    """Block-sparse static attention using FlexAttention.

    CLS tokens (first ``num_cls`` positions) attend to the full sequence.
    Patch tokens attend to CLS tokens plus ``window_size`` neighbouring
    chunks on each side.  Patches should be Hilbert-sorted beforehand so
    that sequence-local chunks correspond to spatially nearby patches.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens at the front of the sequence.
        window_size: Number of neighbouring chunks (on each side) that each
            patch chunk attends to.  ``0`` = same-chunk only + CLS.
        chunk_size: Number of Hilbert-sorted patches per logical chunk.
            Must be a multiple of ``flex_block_size``.
        flex_block_size: FlexAttention kernel tile size. Hardware-dependent;
            128 is optimal for H100.
        rope_theta: Base frequency for :class:`Rope2D`.
        rope_coord_high: Coordinate normalisation divisor for :class:`Rope2D`.
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
        if chunk_size % flex_block_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be a multiple of "
                f"flex_block_size ({flex_block_size})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.flex_block_size = flex_block_size
        self.head_dim = embed_dim // num_heads

        self.rope = Rope2D(self.head_dim, theta=rope_theta, coord_high=rope_coord_high)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        """Block-sparse attention over a CLS-prepended patch sequence.

        Args:
            x: ``(batch_size, num_cls + patch_len, embed_dim)`` — CLS tokens prepended.
            coords: ``(batch_size, patch_len, 2)`` — patch pixel coordinates for 2-D RoPE.

        Returns:
            ``(batch_size, num_cls + patch_len, embed_dim)``
        """
        batch_size, seq_len, embed_dim = x.shape
        num_cls = self.num_cls

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)

        if coords is not None:
            q = torch.cat([q[:, :, :num_cls], self.rope(q[:, :, num_cls:], coords)], dim=2)
            k = torch.cat([k[:, :, :num_cls], self.rope(k[:, :, num_cls:], coords)], dim=2)

        chunk_size = self.chunk_size
        window_size = self.window_size

        def mask_mod(b, h, q_idx, kv_idx):
            q_is_cls = q_idx < num_cls
            kv_is_cls = kv_idx < num_cls
            q_chunk = q_idx // chunk_size
            kv_chunk = kv_idx // chunk_size
            in_window = (q_chunk - kv_chunk).abs() <= window_size
            return q_is_cls | kv_is_cls | in_window

        block_mask = build_block_mask(
            seq_len, num_cls, chunk_size, window_size,
            self.flex_block_size, x.device, mask_mod,
        )
        out = compiled_flex_attention(q, k, v, block_mask=block_mask)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.out_proj(out)


class StaticSparseViTBlock(nn.Module):
    """Transformer block with StaticSparseAttention and an MLP, using pre-norm.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens.
        window_size: Neighbouring chunks on each side for patch attention.
        chunk_size: Patches per logical chunk.
        flex_block_size: FlexAttention kernel tile size.
        expansion_factor: Hidden-dim expansion factor in the MLP.
        proj_dropout: Dropout after attention and MLP projections.
        rope_theta: RoPE base frequency forwarded to :class:`StaticSparseAttention`.
        rope_coord_high: RoPE coordinate normalisation forwarded to
            :class:`StaticSparseAttention`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_cls: int = 1,
        window_size: int = 1,
        chunk_size: int = 256,
        flex_block_size: int = 128,
        expansion_factor: float = 4.0,
        proj_dropout: float = 0.0,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = StaticSparseAttention(
            embed_dim, num_heads, num_cls, window_size, chunk_size,
            flex_block_size, rope_theta, rope_coord_high,
        )
        self.drop1 = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * expansion_factor)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), coords))
        x = x + self.mlp(self.norm2(x))
        return x


class StaticSparseViTSlideEncoder(nn.Module):
    """Slide-level ViT encoder using block-sparse static attention.

    Takes a bag of pre-extracted patch embeddings, Hilbert-sorts them for
    spatial locality, and produces a slide-level classification.

    Args:
        in_features: Patch embedding dimension (1280 for Virchow2 CLS token).
        out_features: Number of output classes.
        embed_dim: Internal transformer dimension.
        num_heads: Number of attention heads (must divide ``embed_dim``).
        num_layers: Number of transformer blocks.
        num_cls: Number of global CLS tokens.
        window_size: Neighbouring chunks on each side for patch attention.
        chunk_size: Patches per logical chunk.
        flex_block_size: FlexAttention kernel tile size.
        expansion_factor: MLP hidden-dim expansion factor.
        proj_dropout: Dropout after projections.
        rope_theta: Base frequency for 2-D RoPE. Default: 10_000.
        rope_coord_high: Coordinate normalisation divisor for RoPE. Default: 100_000.
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        num_cls: int = 2,
        window_size: int = 1,
        chunk_size: int = 256,
        flex_block_size: int = 128,
        expansion_factor: float = 4.0,
        proj_dropout: float = 0.0,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.embed_dim = embed_dim
        self.out_features = out_features

        self.input_proj = nn.Linear(in_features, embed_dim)

        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.blocks = nn.ModuleList([
            StaticSparseViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_cls=num_cls,
                window_size=window_size,
                chunk_size=chunk_size,
                flex_block_size=flex_block_size,
                expansion_factor=expansion_factor,
                proj_dropout=proj_dropout,
                rope_theta=rope_theta,
                rope_coord_high=rope_coord_high,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_pool = nn.Linear(embed_dim, 1, bias=False)
        self.head = nn.Linear(embed_dim, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> dict[str, Tensor]:
        """Encode a bag of patch embeddings and return slide-level logits.

        Args:
            x: Patch embeddings ``(batch_size, patch_len, in_features)``.
            coords: Patch pixel coordinates ``(batch_size, patch_len, 2)``.

        Returns:
            Dict with ``"logits"`` of shape ``(batch_size, out_features)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        x = self.input_proj(x)

        if coords is not None:
            sort_idx = hilbert_sort(coords)
            x = x.gather(1, sort_idx.unsqueeze(-1).expand_as(x))
            coords = coords.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))

        cls = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x, coords)

        x = self.norm(x)

        cls_tokens = x[:, :self.num_cls]
        cls_pool_weights = torch.softmax(self.cls_pool(cls_tokens), dim=1)
        cls_out = (cls_pool_weights * cls_tokens).sum(dim=1)

        return {
            "logits": self.head(cls_out),
            "cls_pool_weights": cls_pool_weights,
        }

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
