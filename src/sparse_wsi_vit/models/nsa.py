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

from sparse_wsi_vit.models.nsa_kernels.parallel import parallel_nsa
from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding
from sparse_wsi_vit.models.abmil import ABMIL # used instead of CLS for aggregation; can do ablations if time permits
from functools import partial #for RMSNorm
from sparse_wsi_vit.models.vit_5.models_vit5 import Block, RMSNorm

import timm.layers # for init weights

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
            block_size: int = 64, # 8x8 h5 macro blocks input
            block_counts: int = 16,
            rope_coord_high: float = 100_000.0
    ):
        super().__init__()
        assert num_q_heads % (num_kv_heads * 16) == 0, "NSA needs HQ to be a multiple of 16 * HKV"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        # explicit if provided, otherwise derived
        self.head_dim = head_dim if head_dim is not None else d_model // num_q_heads

        assert self.head_dim in [64, 128], f"Triton kernels expect head_dim 64 or 128, got {self.head_dim}"

        self.block_size = block_size
        self.block_counts = block_counts

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.rope = VisionRotaryEmbedding(
            dim=self.head_dim,
            freqs_for="lang", # hmm
            coord_high=rope_coord_high,
            dynamic=True
        )

        # gates for dynamic block selection
        self.g_proj = nn.Linear(d_model, num_q_heads * 3, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: Tensor, coords: Tensor | None = None, cu_seqlens: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        if cu_seqlens is None:
            if B == 1:
                cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=x.device)
            else:
                raise ValueError("Code doesn't support B > 1 yet")

        q = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_q_heads)
        k = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), '... (h d) -> ... h d', h=self.num_kv_heads)
        g = rearrange(self.g_proj(x), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
        # use sigmoid here, mosaic used softmax which is kinda interesting

        # CLS? fuggedaboudit not in my house 😤
        if coords is not None:
                q = self.rope(q, coords)
                k = self.rope(k, coords)

        out = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.block_size,
            block_counts=self.block_counts,
            cu_seqlens=cu_seqlens,
        )

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
            **_,
    ) -> None:
        super().__init__()
        self.attn = NativeSparseAttention(
            d_model=dim,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            block_counts=block_counts,
        )

    def forward(self, x: Tensor, coords: Tensor | None = None, cu_seqlens: Tensor | None = None) -> Tensor:
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
            **kwargs
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            Attention_block=NSAViTWrapper,
            block_size=block_size,
            block_counts=block_counts,
            num_kv_heads=num_kv_heads,
            **kwargs
        )

        # could e.g. change forward pass here



class NSAViTSlideEncoder(nn.Module):
     """Slide-level ViT encoder using Native Sparse Attention.

    Takes a bag of pre-extracted patch embeddings and produces a slide-level classification. Mirrors API of StaticSparse/DSA SlideEncoder but uses ABMIL pooling instead of CLS tokens for global aggregation.
     """
     # (can do ablations or whatever if there's time but idk)
     def __init__(
               self,
               in_features: int = 1280,
               out_features: int = 1,
               embed_dim: int = 384,
               num_heads: int = 6,
               num_layers: int = 6,
               num_cls: int = 0, # for API parity but don't feed it any
               block_size: int = 64,
               block_counts: int = 16,
               expansion_factor: float = 4.0,
               attn_dropout: float = 0.0,
               proj_dropout: float = 0.0,
               gradient_checkpointing: bool = False,
               **kwargs
     ) -> None:
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.embed_dim = embed_dim
        self.out_features = out_features

        if num_cls > 0:
            logging.warning("NSAViTSlideEncoder ignores num_cls, and uses ABMIL for pooling instead.")

        # project patch embeddings
        self.input_proj = nn.Linear(in_features, embed_dim)

        # NSA Blocks
        self.layers = nn.ModuleList([
            NSABlock(
                dim=embed_dim,
                num_heads=num_heads,

                block_size=block_size,
                block_counts=block_counts,
                num_kv_heads=1,

                mlp_ratio=expansion_factor,
                drop=proj_dropout,
                attn_drop=attn_dropout,
                norm_layer=partial(RMSNorm, eps=1e-6) # apparently RMSNorm is better
            )
            for _ in range(num_layers)
        ])

        # ABMIL Aggregation
        self.pooler = ABMIL(
            in_features=embed_dim,
            hidden_dim=256,
            out_features=out_features,
            num_branches=1,
            attention_dropout=proj_dropout
        )

        self.apply(self._init_weights)

    # maybe should change init_weights here? review, compare dsa vs standard vit5 init
     def _init_weights(self, m):
        import timm.layers
        if isinstance(m, nn.Linear):
            # 1. DSA logic: Preserve variance of Virchow2 features at the input
            if m is self.input_proj:
                nn.init.xavier_uniform_(m.weight)
            # ViT logic: Stabilize the deep attention and MLP layers
            else:
                timm.layers.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 3. Standard Norm logic
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

     def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None, cu_seqlens: torch.Tensor | None = None) -> dict:
         h = self.input_proj(x)

         for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, coords, cu_seqlens, use_reentrant=False)
            else:
                h = layer(h, coords=coords, cu_seqlens=cu_seqlens)

         # ABMIL returns a dict containing {"logits": ..., "attention": ...}
         return self.pooler(h, return_attention=True)














