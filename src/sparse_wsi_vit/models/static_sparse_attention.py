"""
Approach A: Static sparse attention for WSI slide encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent pairs: [..., x1, x2, ...] → [..., -x2, x1, ...]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


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
    """Longformer-style static sparse attention with optional local window.

    The first ``num_cls`` positions are global CLS tokens that attend to all
    tokens. Patch tokens attend to CLS tokens and, if ``window_size > 0``,
    also to the ``2 * window_size + 1`` sequence neighbours around them.

    The two attention patterns are combined into a single softmax per patch
    query over the concatenated key set ``[CLS_1 ... CLS_K | patch_{i-W} ...
    patch_{i+W}]``.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens at the front of the sequence. (num_cls + patch_len >= 8 for MMA)
        window_size: One-sided local window radius for patch-to-patch attention.
            Patch i attends to patches [i-W, i+W].
            Set to 0 to disable local patch attention (CLS-only).
        dilation: Step size between attended patches inside the local window.
            ``dilation=1`` gives a standard consecutive window.
            ``dilation=d`` attends to every d-th patch, covering a span of
            ``2 * window_size * dilation`` at the same key-count cost.
        attn_dropout: Dropout probability on attention weights.
        chunk_size: Number of patch queries to process per SDPA call in the windowed
            path. Each call uses a block-diagonal mask of shape
            ``(chunk_size, chunk_size * context_len)``, keeping the query sequence long enough for
            Flash Attention / MMA (requires chunk_size >= 8).
        rope_theta: Base frequency for :class:`Rope2D`. Default: 10_000.
        rope_coord_high: Coordinate normalisation divisor for :class:`Rope2D`.
            Raw pixel coordinates are divided by this value. Default: 100_000.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_cls: int = 1,
        window_size: int = 0,
        dilation: int = 1,
        attn_dropout: float = 0.0,
        chunk_size: int = 512,
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
        self.dilation = dilation
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size

        self.rope = Rope2D(self.head_dim, theta=rope_theta, coord_high=rope_coord_high)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = attn_dropout

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        """Static sparse attention.

        Args:
            x: Token sequence ``(batch_size, num_cls + patch_len, embed_dim)``,
               CLS tokens prepended.
            coords: Patch pixel coordinates ``(batch_size, patch_len, 2)``.
               When provided, 2-D RoPE is applied to patch Q and K.

        Returns:
            Output of shape ``(batch_size, num_cls + patch_len, embed_dim)``.
        """
        batch_size, seq_len, embed_dim = x.shape
        num_cls = self.num_cls
        patch_len = seq_len - num_cls

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)            # each (batch_size, num_heads, seq_len, head_dim)

        # Apply 2-D RoPE to patch Q and K — CLS tokens have no spatial position
        if coords is not None:
            q = torch.cat([q[:, :, :num_cls], self.rope(q[:, :, num_cls:], coords)], dim=2)
            k = torch.cat([k[:, :, :num_cls], self.rope(k[:, :, num_cls:], coords)], dim=2)

        dropout_p = self.attn_dropout if self.training else 0.0

        q_cls, k_cls, v_cls = q[:, :, :num_cls], k[:, :, :num_cls], v[:, :, :num_cls]
        q_patch              = q[:, :, num_cls:]

        # CLS tokens: global attention over the full sequence
        cls_out = F.scaled_dot_product_attention(q_cls, k, v, dropout_p=dropout_p)

        # Patch tokens: CLS only, or CLS + dilated local window
        if self.window_size == 0:
            patch_out = F.scaled_dot_product_attention(
                q_patch, k_cls, v_cls, dropout_p=dropout_p
            )
        else:
            patch_out = self._windowed_patch_attention(
                q_patch, k_cls, v_cls,
                k[:, :, num_cls:], v[:, :, num_cls:],
                patch_len, dropout_p,
            )
        out = torch.cat([cls_out, patch_out], dim=2)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.out_proj(out)

    def _windowed_patch_attention(
        self,
        q_patch: Tensor,  # (batch_size, num_heads, patch_len, head_dim)
        k_cls:   Tensor,  # (batch_size, num_heads, num_cls,   head_dim)
        v_cls:   Tensor,  # (batch_size, num_heads, num_cls,   head_dim)
        k_patch: Tensor,  # (batch_size, num_heads, patch_len, head_dim)
        v_patch: Tensor,  # (batch_size, num_heads, patch_len, head_dim)
        patch_len: int,
        dropout_p: float,
    ) -> Tensor:
        """Each patch attends to its CLS tokens plus a dilated local window.

        The gather and SDPA are fused inside a chunk loop so that at most
        ``chunk_size`` patches' context keys/values are in memory at once.
        Peak memory per chunk: O(chunk_size * context_len * head_dim).

        Boundary patches attend to fewer neighbours (masked, not wrapped).

        Returns: (batch_size, num_heads, patch_len, head_dim)
        """
        batch_size, num_heads, _, head_dim = q_patch.shape
        num_cls = self.num_cls
        window_size = self.window_size
        dilation = self.dilation
        chunk_size = self.chunk_size
        window_len = 2 * window_size + 1
        context_len = num_cls + window_len
        pad = window_size * dilation

        # Pad k/v along the sequence axis so boundary patches can use the same
        # gather logic as interior patches (padded positions are masked out below).
        k_padded = F.pad(k_patch, (0, 0, pad, pad))  # (batch_size, num_heads, patch_len+2*pad, head_dim)
        v_padded = F.pad(v_patch, (0, 0, pad, pad))

        # Precompute window offsets and boundary window positions (shared across chunks).
        offsets    = torch.arange(window_len, device=q_patch.device) * dilation  # (window_len,)
        window_pos = (torch.arange(window_len, device=q_patch.device) - window_size) * dilation  # (window_len,)

        # Pre-expand CLS k/v — shape is small; safe to keep fully expanded.
        k_cls_exp = k_cls.unsqueeze(2)   # (batch_size, num_heads, 1, num_cls, head_dim)  — broadcast below
        v_cls_exp = v_cls.unsqueeze(2)

        chunks: list[Tensor] = []
        for start in range(0, patch_len, chunk_size):
            end   = min(start + chunk_size, patch_len)
            chunk = end - start  # number of patches in this chunk

            chunk_positions = torch.arange(start, end, device=q_patch.device)  # (chunk,)

            # Gather indices for this chunk's window neighbours.
            gather_idx = chunk_positions.unsqueeze(1) + offsets.unsqueeze(0)  # (chunk, window_len)
            gather_idx_exp = (
                gather_idx
                .unsqueeze(0).unsqueeze(0).unsqueeze(-1)              # (1, 1, chunk, window_len, 1)
                .expand(batch_size, num_heads, -1, -1, head_dim)      # (batch_size, num_heads, chunk, window_len, head_dim)
            )

            k_windows = torch.gather(
                k_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1),
                2, gather_idx_exp,
            )  # (batch_size, num_heads, chunk, window_len, head_dim)
            v_windows = torch.gather(
                v_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1),
                2, gather_idx_exp,
            )  # (batch_size, num_heads, chunk, window_len, head_dim)

            # Prepend CLS tokens: (batch_size, num_heads, chunk, context_len, head_dim)
            k_context = torch.cat(
                [k_cls_exp.expand(-1, -1, chunk, -1, -1), k_windows], dim=3
            )
            v_context = torch.cat(
                [v_cls_exp.expand(-1, -1, chunk, -1, -1), v_windows], dim=3
            )

            # Boundary mask for this chunk: (chunk, context_len)
            in_bounds   = (chunk_positions.unsqueeze(1) + window_pos >= 0) & \
                          (chunk_positions.unsqueeze(1) + window_pos < patch_len)
            window_mask = q_patch.new_zeros(chunk, window_len)
            window_mask[~in_bounds] = float("-inf")
            attn_mask   = torch.cat(
                [q_patch.new_zeros(chunk, num_cls), window_mask], dim=1
            )  # (chunk, context_len)

            # Flatten for SDPA: (batch_size*num_heads*chunk, 1, head_dim) queries
            num_queries_flat = batch_size * num_heads * chunk
            q_flat = q_patch[:, :, start:end].reshape(num_queries_flat, 1, head_dim)
            k_flat = k_context.reshape(num_queries_flat, context_len, head_dim)
            v_flat = v_context.reshape(num_queries_flat, context_len, head_dim)

            # Mask: (chunk, context_len) → (batch_size*num_heads*chunk, 1, context_len)
            mask_flat = (
                attn_mask
                .unsqueeze(1)                                  # (chunk, 1, context_len)
                .unsqueeze(0)                                  # (1, chunk, 1, context_len)
                .expand(batch_size * num_heads, -1, -1, -1)   # (batch_size*num_heads, chunk, 1, context_len)
                .reshape(num_queries_flat, 1, context_len)
            )

            out_chunk = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                attn_mask=mask_flat,
                dropout_p=dropout_p,
            )  # (batch_size*num_heads*chunk, 1, head_dim)

            chunks.append(out_chunk.reshape(batch_size, num_heads, chunk, head_dim))

        return torch.cat(chunks, dim=2)  # (batch_size, num_heads, patch_len, head_dim)


class StaticSparseViTBlock(nn.Module):
    """Transformer block with StaticSparseAttention and an MLP, using pre-norm.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_cls: Number of global CLS tokens.
        window_size: Local window radius for patch-to-patch attention (0 = disabled).
        dilation: Step size between attended window patches (1 = consecutive).
        expansion_factor: Hidden-dim expansion factor in the MLP.
        attn_dropout: Dropout in attention weights.
        proj_dropout: Dropout after attention and MLP projections.
        chunk_size: Patch chunk size forwarded to :class:`StaticSparseAttention`.
        rope_theta: RoPE base frequency forwarded to :class:`StaticSparseAttention`.
        rope_coord_high: RoPE coordinate normalisation forwarded to
            :class:`StaticSparseAttention`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_cls: int = 1,
        window_size: int = 0,
        dilation: int = 1,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        chunk_size: int = 512,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = StaticSparseAttention(
            embed_dim, num_heads, num_cls, window_size, dilation,
            attn_dropout, chunk_size, rope_theta, rope_coord_high,
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
    """Slide-level ViT encoder using Longformer-style static sparse attention.

    Takes a bag of pre-extracted patch embeddings and produces a slide-level
    classification.

    Args:
        in_features: Patch embedding dimension (1280 for Virchow2 CLS token).
        out_features: Number of output classes.
        embed_dim: Internal transformer dimension.
        num_heads: Number of attention heads (must divide ``embed_dim``).
        num_layers: Number of transformer blocks.
        num_cls: Number of global CLS tokens.
        window_size: One-sided local window radius for patch attention (0 = disabled).
        dilation: Step size between attended window patches (1 = consecutive).
        expansion_factor: MLP hidden-dim expansion factor.
        attn_dropout: Dropout on attention weights.
        proj_dropout: Dropout after projections.
        chunk_size: Number of patch queries processed per SDPA call in the windowed
            attention path. Must be >= 8 for Flash Attention / MMA.
        rope_theta: Base frequency for 2-D RoPE. Default: 10_000.
        rope_coord_high: Coordinate normalisation divisor for RoPE. Raw pixel
            coordinates are divided by this value. Default: 100_000.
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        num_cls: int = 2,
        window_size: int = 0,
        dilation: int = 1,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        chunk_size: int = 512,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.embed_dim = embed_dim
        self.out_features = out_features

        self.input_proj = nn.Linear(in_features, embed_dim)

        # Learned global CLS tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            StaticSparseViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_cls=num_cls,
                window_size=window_size,
                dilation=dilation,
                expansion_factor=expansion_factor,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                chunk_size=chunk_size,
                rope_theta=rope_theta,
                rope_coord_high=rope_coord_high,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.cls_pool = nn.Linear(embed_dim, 1, bias=False)

        # Classification head
        self.head = nn.Linear(embed_dim, out_features)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> dict[str, Tensor]:
        """Encode a bag of patch embeddings and return slide-level logits.

        Args:
            x: Patch embeddings ``(B, patch_len, in_features)``.
            coords: Patch pixel coordinates ``(B, patch_len, 2)``.
                When provided, 2-D RoPE is applied to patch Q and K inside
                every attention layer. CLS tokens are never rotated.

        Returns:
            Dict with ``"logits"`` of shape ``(B, out_features)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Project patch embeddings into the transformer's working dimension
        x = self.input_proj(x)  # (batch_size, patch_len, embed_dim)

        # Prepend CLS tokens
        cls = self.cls_tokens.expand(batch_size, -1, -1)  # (batch_size, num_cls, embed_dim)
        x = torch.cat([cls, x], dim=1)                    # (batch_size, num_cls + patch_len, embed_dim)

        for block in self.blocks:
            x = block(x, coords)

        x = self.norm(x)

        cls_tokens = x[:, :self.num_cls]                            # (batch_size, num_cls, embed_dim)
        weights = torch.softmax(self.cls_pool(cls_tokens), dim=1)   # (batch_size, num_cls, 1)
        cls_out = (weights * cls_tokens).sum(dim=1)                 # (batch_size, embed_dim)

        logits = self.head(cls_out)  # (batch_size, out_features)

        return {"logits": logits}

class StaticSparseAttentionAdapter(nn.Module):
    """Adapts StaticSparseAttention to the ViT-5 Block ``Attention_block`` API.

    ``Block.__init__`` calls ``Attention_block(dim, num_heads=..., attn_drop=...,
    flash=..., rope_size=..., ...)`` with ViT-5-specific kwargs. This adapter
    accepts the relevant subset and silently absorbs the rest via ``**_``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        num_cls: int = 1,
        window_size: int = 0,
        dilation: int = 1,
        chunk_size: int = 512,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
        attn_drop: float = 0.0,
        **_,
    ) -> None:
        super().__init__()
        self.attn = StaticSparseAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_cls=num_cls,
            window_size=window_size,
            dilation=dilation,
            attn_dropout=attn_drop,
            chunk_size=chunk_size,
            rope_theta=rope_theta,
            rope_coord_high=rope_coord_high,
        )

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        return self.attn(x, coords)


# Notes:
# embed_dim / num_heads should equal 64
