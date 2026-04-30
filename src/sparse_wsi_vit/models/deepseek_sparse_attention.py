"""
Approach B: DeepSeek Sparse Attention (DSA) for WSI slide encoding.
"""


import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from sparse_wsi_vit.models.deepseek_sparse_attention_kernels.kernels import (
    LightningIndexerFunction,
    DSAAttentionFunction
)


# Top-level module

class DeepSeekSparseAttention(nn.Module):
    def __init__(
        self,
        d_model,
        attention_heads,
        indexer_heads,
        indexer_dim,
        top_k,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
    ):
        super().__init__()

        self.d_model          = d_model
        self.attention_heads  = attention_heads
        self.head_dim         = self.d_model // self.attention_heads
        self.indexer_heads    = indexer_heads
        self.indexer_dim      = indexer_dim
        self.top_k            = top_k
        self.BLOCK_Q          = BLOCK_Q
        self.BLOCK_K          = BLOCK_K
        self.BLOCK_D          = BLOCK_D
        self.BLOCK_INDEXER_H  = indexer_heads
        self.BLOCK_ATTENTION_H = attention_heads

        self.indexer_head_weights = nn.Parameter(torch.randn(indexer_heads))
        self.indexer_proj = IndexerProjection(d_model, indexer_heads, indexer_dim)
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model)
        self.out_proj  = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, D)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        q_proj, k_proj = self.indexer_proj(x)  # (B, T, IH, ID)

        # LightningIndexerFunction: forward selects top-k tokens per query,
        # backward propagates gradients to q_proj, k_proj, indexer_head_weights.
        idx, valid, scores = LightningIndexerFunction.apply(
            q_proj, k_proj, self.indexer_head_weights,
            B, T, self.indexer_heads, self.indexer_dim, self.top_k,
            self.BLOCK_Q, self.BLOCK_K, self.BLOCK_D, self.BLOCK_INDEXER_H,
        )
        # idx:    (B, T, TOP_K)  int32,  always in [0, T-1]
        # valid:  (B, T, TOP_K)  float,  1.0 = real slot, 0.0 = unfilled
        # scores: (B, T, TOP_K)  float,  raw indexer scores (differentiable)

        # This keeps scores in the autograd graph so indexer weights get grads.
        routing_weights = scores * valid  # (B, T, TOP_K)

        o_ptr = torch.zeros(
            (B, self.attention_heads, T, self.head_dim), device=device, dtype=dtype
        )

        out = DSAAttentionFunction.apply(
            q, k, v,
            idx.detach(), valid.detach(), routing_weights,
            o_ptr,
            B, self.attention_heads, T, self.head_dim, self.top_k,
            self.BLOCK_Q, self.BLOCK_K, self.BLOCK_ATTENTION_H, self.BLOCK_D,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


# Indexer projection

class IndexerProjection(nn.Module):
    def __init__(self, d_model, indexer_heads, indexer_dim):
        super().__init__()
        self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim)
        self.k_proj = nn.Linear(d_model, indexer_heads * indexer_dim)
        self.indexer_heads = indexer_heads
        self.indexer_dim   = indexer_dim

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.indexer_heads, self.indexer_dim)
        K = self.k_proj(x).view(B, T, self.indexer_heads, self.indexer_dim)
        return Q.contiguous(), K.contiguous()


class DSAViTBlock(nn.Module):
    """Transformer block wrapping DeepSeekSparseAttention with an MLP, using pre-norm.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads (must divide embed_dim).
        indexer_heads: Number of heads in the lightweight indexer (4 as in DeepSeek).
        indexer_dim: Per-head dim of the indexer projection (32 as in DeepSeek).
        top_k: Number of key tokens each query attends to.
        block_q: Query tile size for Triton kernels.
        block_k: Key tile size for Triton kernels.
        block_d: Feature tile size for Triton kernels.
        expansion_factor: MLP hidden-dim expansion factor.
        attn_dropout: Unused (kept for API parity with StaticSparseViTBlock).
        proj_dropout: Dropout after attention and MLP projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        indexer_heads: int = 4,
        indexer_dim: int = 32,
        top_k: int = 512,
        block_q: int = 16,
        block_k: int = 32,
        block_d: int = 32,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.0,   # kept for API parity, not used in DSA
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = DeepSeekSparseAttention(
            d_model        = embed_dim,
            attention_heads= num_heads,
            indexer_heads  = indexer_heads,
            indexer_dim    = indexer_dim,
            top_k          = top_k,
            BLOCK_Q        = block_q,
            BLOCK_K        = block_k,
            BLOCK_D        = block_d,
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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class DSAViTSlideEncoder(nn.Module):
    """Slide-level ViT encoder using DeepSeek Native Sparse Attention.

    Takes a bag of pre-extracted patch embeddings and produces a slide-level
    classification. Mirrors the API of StaticSparseViTSlideEncoder so the two
    can be swapped as drop-in alternatives.

    Args:
        in_features: Patch embedding dimension (1280 for Virchow2 CLS token).
        out_features: Number of output classes.
        embed_dim: Internal transformer dimension.
        num_heads: Number of attention heads (must divide embed_dim,
                   embed_dim // num_heads should equal 64 or 128).
        num_layers: Number of transformer blocks.
        num_cls: Number of global CLS tokens prepended to the sequence.
        indexer_heads: Number of lightweight indexer heads (default 4).
        indexer_dim: Per-head dim of the indexer projection (default 32).
        top_k: Number of key tokens each query attends to per layer.
        block_q: Query tile size for Triton kernels.
        block_k: Key tile size for Triton kernels.
        block_d: Feature tile size for Triton kernels.
        expansion_factor: MLP hidden-dim expansion factor.
        attn_dropout: Kept for API parity (unused in DSA kernels).
        proj_dropout: Dropout after projections.
    """

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        num_cls: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 32,
        top_k: int = 512,
        block_q: int = 16,
        block_k: int = 32,
        block_d: int = 32,
        expansion_factor: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                f"Recommended: embed_dim // num_heads == 64 or 128."
            )

        self.num_cls    = num_cls
        self.embed_dim  = embed_dim
        self.out_features = out_features

        # Project patch embeddings into the transformer's working dimension
        self.input_proj = nn.Linear(in_features, embed_dim)

        # Learned global CLS tokens — prepended to every sequence
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.blocks = nn.ModuleList([
            DSAViTBlock(
                embed_dim      = embed_dim,
                num_heads      = num_heads,
                indexer_heads  = indexer_heads,
                indexer_dim    = indexer_dim,
                top_k          = top_k,
                block_q        = block_q,
                block_k        = block_k,
                block_d        = block_d,
                expansion_factor = expansion_factor,
                attn_dropout   = attn_dropout,
                proj_dropout   = proj_dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Weighted pooling over CLS tokens → single slide embedding
        self.cls_pool = nn.Linear(embed_dim, 1, bias=False)

        # Classification head
        self.head = nn.Linear(embed_dim, out_features)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Encode a bag of patch embeddings and return slide-level logits.

        Args:
            x: Patch embeddings ``(B, patch_len, in_features)`` or
               ``(patch_len, in_features)`` for a single slide (auto-unsqueezed).

        Returns:
            Dict with ``"logits"`` of shape ``(B, out_features)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B = x.shape[0]

        print(f"[forward] slide shape: {x.shape}, T={x.shape[1]}", flush=True)

        x = self.input_proj(x)  # (B, patch_len, embed_dim)

        # Prepend CLS tokens
        cls = self.cls_tokens.expand(B, -1, -1)  # (B, num_cls, embed_dim)
        x   = torch.cat([cls, x], dim=1)          # (B, num_cls + patch_len, embed_dim)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)

        # Weighted pool over CLS tokens
        cls_tokens = x[:, :self.num_cls]                           # (B, num_cls, embed_dim)
        weights    = torch.softmax(self.cls_pool(cls_tokens), dim=1)  # (B, num_cls, 1)
        cls_out    = (weights * cls_tokens).sum(dim=1)             # (B, embed_dim)

        logits = self.head(cls_out)  # (B, out_features)

        return {"logits": logits}

