"""
WSI slide encoder using ViT-5 blocks with configurable sparse attention.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from sparse_wsi_vit.models.vit_5.models_vit5 import Block, RMSNorm
from sparse_wsi_vit.models.static_sparse_attention import StaticSparseAttentionAdapter


_SPARSE_ATTN_TYPES = ("static")

class SparseViT5SlideEncoder(nn.Module):
    """WSI slide encoder: ViT-5 blocks with configurable sparse attention.

    Combines:
    - The ViT-5 :class:`~sparse_wsi_vit.models.vit_5.models_vit5.Block` structure:
      pre-norm (RMSNorm), learnable layer-scale, stochastic depth, and timm Mlp.
    - Drop-in sparse attention: either Longformer-style
    - Slide-level head: input projection -> prepend CLS tokens -> transformer
      -> weighted CLS pooling -> linear classifier.

    Args:
        in_features: Patch-embedding input dimension (1280 for Virchow2 CLS).
        out_features: Number of output classes (1 = binary, >1 = multiclass).
        embed_dim: Transformer hidden dimension.
        num_heads: Number of attention heads. ``embed_dim // num_heads`` should be 64.
        depth: Number of transformer blocks.
        num_cls: Number of global CLS tokens prepended to every sequence.
        sparse_attn: Which sparse attention to use — ``"static"`` or ``"dsa"``.

        StaticSparseAttention kwargs (active when ``sparse_attn="static"``):
            window_size: One-sided local window radius (0 = CLS-only).
            dilation: Step size between attended window patches.
            chunk_size: Patch chunk size forwarded to the windowed SDPA path.
            rope_theta: RoPE base frequency.
            rope_coord_high: Coordinate normalisation divisor for 2-D RoPE.

        Shared transformer kwargs:
            mlp_ratio: MLP hidden-dim expansion factor.
            attn_dropout: Dropout on attention weights.
            proj_dropout: Dropout after projections.
            drop_path_rate: Stochastic-depth drop probability (same for all layers).

        ViT-5 Block kwargs:
            layer_scale: Enable learnable per-layer scale (ViT-5 default: True).
            init_scale: Initial layer-scale value (ViT-5 default: 1e-4).

        gradient_checkpointing: Recompute block activations during backward to
            trade compute for memory.
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 6,
        num_cls: int = 2,
        sparse_attn: str = "static",
        # StaticSparseAttention kwargs
        window_size: int = 3,
        dilation: int = 1,
        chunk_size: int = 512,
        rope_theta: float = 10_000.0,
        rope_coord_high: float = 100_000.0,
        # Shared transformer kwargs
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        # ViT-5 Block kwargs
        layer_scale: bool = True,
        init_scale: float = 1e-4,
        # Memory
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if sparse_attn not in _SPARSE_ATTN_TYPES:
            raise ValueError(
                f"sparse_attn must be one of {list(_SPARSE_ATTN_TYPES)!r}, "
                f"got {sparse_attn!r}"
            )

        self.num_cls = num_cls
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.gradient_checkpointing = gradient_checkpointing

        self.input_proj = nn.Linear(in_features, embed_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Prepend learned global CLS tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        AdapterClass = StaticSparseAttentionAdapter if sparse_attn == "static" else None
        if sparse_attn == "static":
            sparse_kwargs: dict = dict(
                num_cls=num_cls,
                window_size=window_size,
                dilation=dilation,
                chunk_size=chunk_size,
                rope_theta=rope_theta,
                rope_coord_high=rope_coord_high,
            )

        attn_block = partial(AdapterClass, **sparse_kwargs)

        norm_layer = partial(RMSNorm, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                drop=proj_dropout,
                attn_drop=attn_dropout,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                Attention_block=attn_block,
                init_values=init_scale,
                layer_scale=layer_scale,
                # Disable ViT-5 features handled inside the sparse attention modules
                flash=False,
                rope_size=0,
                rope_reg_size=0,
                num_registers=0,
                qk_norm=False,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.cls_pool = nn.Linear(embed_dim, 1, bias=False)

        # Classification head
        self.head = nn.Linear(embed_dim, out_features)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> dict[str, Tensor]:
        """Encode a bag of pre-extracted patch embeddings into a slide-level prediction.

        Args:
            x: Patch embeddings ``(B, N, in_features)`` or ``(N, in_features)``
               (batch dimension added automatically for single slides).
            coords: Pixel coordinates ``(B, N, 2)``.  Used by the static sparse
                attention variant for 2-D RoPE; passed but ignored by DSA.

        Returns:
            Dict with ``"logits"`` key, shape ``(B, out_features)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords is not None and coords.dim() == 2:
            coords = coords.unsqueeze(0)

        B = x.shape[0]

        x = self.input_proj(x)  # (B, N, embed_dim)

        cls = self.cls_tokens.expand(B, -1, -1)  # (B, num_cls, embed_dim)
        x = torch.cat([cls, x], dim=1)            # (B, num_cls + N, embed_dim)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, coords, use_reentrant=False)
            else:
                x = block(x, coords)

        x = self.norm(x)

        cls_tokens = x[:, : self.num_cls]                          # (B, num_cls, embed_dim)
        weights = torch.softmax(self.cls_pool(cls_tokens), dim=1)  # (B, num_cls, 1)
        cls_pooled = (weights * cls_tokens).sum(dim=1)             # (B, embed_dim)

        return {"logits": self.head(cls_pooled)}
