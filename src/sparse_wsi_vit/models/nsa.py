"""
Approach C: Native Sparse Attention (NSA) for WSI slide encoding.
"""

# Native Sparse Pytorch + ViT-5 adapter
import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from functools import partial  # for RMSNorm
import timm.layers  # for init weights

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# local stuff
from sparse_wsi_vit.models.nsa_kernels.flash_topk import FlashTopK
from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding
from sparse_wsi_vit.models.abmil import (ABMIL)  # used instead of CLS for aggregation; can do ablations if time permits
from sparse_wsi_vit.models.vit_5.models_vit5 import Block, RMSNorm

try:
    from flash_attn import flash_attn_func
except ImportError:
    logging.warning("FlashAttention not found, git gud scrub")

def mean_pooling(x, block_size):
    B, T, H, D = x.shape
    return x.view(B, T // block_size, block_size, H, D).mean(dim=2)

class NativeSparseAttention(nn.Module):
    """
    Core NSA module, handles projections and calls kernels.
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int = 16,
        num_kv_heads: int = 1,
        head_dim: int | None = None,
        block_size: int = 64,  # 8x8 h5 macro blocks input
        block_counts: int = 16,
        rope_coord_high: float = 100_000.0,
    ):
        super().__init__()
        assert (
            num_q_heads % (num_kv_heads * 16) == 0
        ), "NSA needs HQ to be a multiple of 16 * HKV"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        # explicit if provided, otherwise derived
        self.head_dim = head_dim if head_dim is not None else d_model // num_q_heads

        assert self.head_dim in [
            64,
            128,
        ], f"Kernels enjoy head_dim 64 or 128, got {self.head_dim}"

        self.block_size = block_size
        self.block_counts = block_counts

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.rope = VisionRotaryEmbedding(
            dim=self.head_dim // 2,
            freqs_for="pixel",  # hmm
            coord_high=rope_coord_high,
            dynamic=True,
        )

        # gates for dynamic block selection
        self.g_proj = nn.Linear(d_model, num_q_heads * 2, bias=False)
        # !!! for now just two gates for global and sparse branches
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

        self.get_blocks = FlashTopK(
            head_dim=self.head_dim,
            num_head=num_q_heads,
            top_k=block_counts, # !!! need GQA inflation
            is_causal=False
        )

    def _create_wsi_block_mask(self, block_indices, q_len, kv_len):
        B, H, _, _ = block_indices.shape
        block_size = self.block_size

        def local_or_sparse_mask(b, h, q_idx, kv_idx):
            q_blk = q_idx // block_size # macro grid
            kv_blk = kv_idx // block_size

            is_local = (q_blk == kv_blk)  # for local attention
            selected = torch.any(block_indices[b,h,q_blk] == kv_blk) # for block sparse

            return is_local | selected # either is ok; note that we treat it the same here (joint softmax so we should maybe try separating later)

        return create_block_mask(local_or_sparse_mask, B=B, H=H, Q_LEN=q_len, KV_LEN=kv_len)

    def forward(
        self, x: Tensor, coords: Tensor | None = None, cu_seqlens: Tensor | None = None
    ) -> Tensor:
        B, T, _ = x.shape

        q = rearrange(self.q_proj(x), "... (h d) -> ... h d", h=self.num_q_heads)
        k = rearrange(self.k_proj(x), "... (h d) -> ... h d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "... (h d) -> ... h d", h=self.num_kv_heads)
        # g = rearrange(self.g_proj(x), "... (h d) -> ... h d", d=3)
        # g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
        # this left commented out to emphasize we dont do this anymore
        # instead:
        gates = rearrange(self.g_proj(x), "... (h d) -> ... h d", d=2)
        g_cmp, g_slc = gates.sigmoid().unbind(-1) # gated to allow for learnable focus

        # CLS? fuggedaboudit not in my house 😤
        if coords is not None:
            q = self.rope(q, coords)
            k = self.rope(k, coords)

        # get blocks
        q_cmp = mean_pooling(q, self.block_size)
        k_cmp = mean_pooling(k, self.block_size)
        v_cmp = mean_pooling(v, self.block_size)

        # global compressed attention (dense FA2)
        out_cmp = flash_attn_func(q, k_cmp, v_cmp, causal=False)
        out_cmp = out_cmp * g_cmp.unsqueeze(-1)

        # !!! needed GQA workaround, copies memory tensor 16x
        k_cmp_timesg = k_cmp.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2) if self.num_q_heads > self.num_kv_heads else k_cmp
        v_cmp_timesg = v_cmp.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=2) if self.num_q_heads > self.num_kv_heads else v_cmp

        _out, block_indices, _, _ = self.get_blocks(q_cmp, k_cmp_timesg, v_cmp_timesg)
        topk_indices = rearrange(block_indices, 'b s h k -> b h s k')

        sparse_mask = self._create_wsi_block_mask(topk_indices, q_len=T, kv_len=T)

        # [B, H, T, D] -> [B, T, H, D] flex needs
        q_flex = q.transpose(1, 2)
        k_flex = k.transpose(1, 2)
        v_flex = v.transpose(1, 2)

        out_sparse_flex = flex_attention(q_flex, k_flex, v_flex, block_mask=sparse_mask)
        out_sparse = out_sparse_flex.transpose(1, 2) * g_slc.unsqueeze(-1)

        out = out_cmp + out_sparse

        return self.out_proj(out.reshape(B, T, -1))


class NSAViTWrapper(nn.Module):
    """Adapts NSA to ViT-5 Attention_Block API. The * ensures only keyword args for the custom ones, **_ absorbs silently."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        num_kv_heads: int = 1,
        block_size: int = 64,
        block_counts: int = 16,
        rope_coord_high: float = 100_000.0,
        **_,
    ) -> None:
        super().__init__()
        self.attn = NativeSparseAttention(
            d_model=dim,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            block_counts=block_counts,
            rope_coord_high=rope_coord_high,
        )

    def forward(
        self, x: Tensor, coords: Tensor | None = None, cu_seqlens: Tensor | None = None
    ) -> Tensor:
        return self.attn(x, coords=coords, cu_seqlens=cu_seqlens)


class NSABlock(Block):
    """Inherits from ViT5 Block, makes it easier to adapt later if necessary"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        # nsa stuff
        block_size: int = 64,
        block_counts: int = 16,
        num_kv_heads: int = 1,
        rope_coord_high: float = 100_000.0,
        **kwargs,
    ) -> None:
        nsa_wrapped_attn = partial(
            NSAViTWrapper,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            block_counts=block_counts,
            rope_coord_high=rope_coord_high
        )
        super().__init__(
            dim=dim, num_heads=num_heads, Attention_block=nsa_wrapped_attn, **kwargs
        )

    # AI written code, bewware
    def forward(
        self, x: Tensor, coords: Tensor | None = None, cu_seqlens: Tensor | None = None
    ) -> Tensor:
        if self.layer_scale:
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(self.norm1(x), coords=coords, cu_seqlens=cu_seqlens)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.attn(self.norm1(x), coords=coords, cu_seqlens=cu_seqlens)
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NSAViTSlideEncoder(nn.Module):
    """Slide-level ViT encoder using Native Sparse Attention.

    Takes a bag of pre-extracted patch embeddings and produces a slide-level classification. Mirrors API of StaticSparse/DSA SlideEncoder but uses ABMIL pooling instead of CLS tokens for global aggregation.
    """

    # (can do ablations or whatever if there's time but idk)
    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 6,
        num_cls: int = 0,  # for API parity but don't feed it any
        block_size: int = 64,
        block_counts: int = 16,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        rope_coord_high: float = 100_000.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.embed_dim = embed_dim
        self.out_features = out_features

        if num_cls > 0:
            logging.warning(
                "NSAViTSlideEncoder ignores num_cls, and uses ABMIL for pooling instead."
            )

        # project patch embeddings
        self.input_proj = nn.Linear(in_features, embed_dim)

        # NSA Blocks
        self.layers = nn.ModuleList(
            [
                NSABlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    block_counts=block_counts,
                    num_kv_heads=1,
                    mlp_ratio=expansion_factor,
                    drop=proj_dropout,
                    attn_drop=attn_dropout,
                    norm_layer=partial(
                        RMSNorm, eps=1e-6
                    ),  # apparently RMSNorm is better
                    rope_coord_high=rope_coord_high,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(embed_dim, eps=1e-6)

        # ABMIL Aggregation
        self.pooler = ABMIL(
            in_features=embed_dim,
            hidden_dim=256,
            out_features=out_features,
            num_branches=1,
            attention_dropout=proj_dropout,
        )

        self.apply(self._init_weights)

    # maybe should change init_weights here? review, compare dsa vs standard vit5 init
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # DSA logic: Preserve variance of Virchow2 features at the input
            if m is self.input_proj:
                nn.init.xavier_uniform_(m.weight)
            # ViT logic: Stabilize the deep attention and MLP layers
            else:
                timm.layers.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        true_lengths: torch.Tensor | None = None,
    ) -> dict:
        h = self.input_proj(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, coords, cu_seqlens, use_reentrant=False)
            else:
                h = layer(h, coords=coords, cu_seqlens=cu_seqlens)

        h = self.norm(h)
        # ABMIL returns a dict containing {"logits": ..., "attention": ...}
        return self.pooler(h, lengths=true_lengths, return_attention=True)
