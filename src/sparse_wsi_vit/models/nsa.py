"""
Approach C: Native Sparse Attention (NSA) for WSI slide encoding.
"""

# Native Sparse Pytorch + ViT-5 adapter

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from sparse_wsi_vit.models.nsa_kernels.parallel import parallel_nsa
from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding

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

        q = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_q_heads)
        k = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), '... (h d) -> ... h d', h=self.num_kv_heads)
        g = rearrange(self.g_proj(x), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
        # use sigmoid here, mosaic used softmax which is kinda interesting

        # need better solution for CLS!! this is garbage
        if coords is not None:
            num_cls = T - coords.shape[1]
            if num_cls > 0:
                q_cls, q_patches = q[:, :num_cls], q[:, num_cls:]
                k_cls, k_patches = k[:, :num_cls], k[:, num_cls:]

                q_patches = self.rope(q_patches, coords)
                k_patches = self.rope(k_patches, coords)

                q = torch.cat([q_cls, q_patches], dim=1)
                k = torch.cat([k_cls, k_patches], dim=1)
            else:
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















