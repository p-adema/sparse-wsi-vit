"""
Approach A: Static sparse attention for WSI slide encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_cls: int = 1,
        window_size: int = 0,
        dilation: int = 1,
        attn_dropout: float = 0.0,
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

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = attn_dropout

    def forward(self, x: Tensor) -> Tensor:
        """Static sparse attention.

        Args:
            x: Token sequence ``(B, num_cls + patch_len, embed_dim)``,
               CLS tokens prepended.

        Returns:
            Output of shape ``(B, num_cls + patch_len, embed_dim)``.
        """
        B, seq_len, embed_dim = x.shape
        num_cls = self.num_cls
        patch_len = seq_len - num_cls

        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)           # each (B, num_heads, seq_len, head_dim)

        q_cls,   k_cls,   v_cls = q[:, :, :num_cls],  k[:, :, :num_cls],  v[:, :, :num_cls]
        q_patch, k_patch, v_patch = q[:, :, num_cls:],  k[:, :, num_cls:],  v[:, :, num_cls:]

        dropout_p = self.attn_dropout if self.training else 0.0


        # CLS tokens: global attention over the full sequence
        cls_out = F.scaled_dot_product_attention(
            q_cls, k, v, dropout_p=dropout_p
        )  # (B, num_heads, num_cls, head_dim)

        # Patch tokens: CLS (+ dilated local window)
        if self.window_size == 0:
            patch_out = F.scaled_dot_product_attention(
                q_patch, k_cls, v_cls, dropout_p=dropout_p
            )  # (B, num_heads, patch_len, head_dim)
        else:
            patch_out = self._windowed_patch_attention(
                q_patch, k_cls, v_cls, k_patch, v_patch, patch_len, dropout_p
            )  # (B, num_heads, patch_len, head_dim)

        out = torch.cat([cls_out, patch_out], dim=2)    # (B, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(B, seq_len, embed_dim)  # (B, seq_len, embed_dim)
        return self.out_proj(out)

    def _windowed_patch_attention(
        self,
        q_patch: Tensor,  # (B, num_heads, patch_len, head_dim)
        k_cls:   Tensor,  # (B, num_heads, num_cls,   head_dim)
        v_cls:   Tensor,  # (B, num_heads, num_cls,   head_dim)
        k_patch: Tensor,  # (B, num_heads, patch_len, head_dim)
        v_patch: Tensor,  # (B, num_heads, patch_len, head_dim)
        patch_len: int,
        dropout_p: float,
    ) -> Tensor:
        """Each patch attends to its CLS tokens plus a dilated local window.

        Boundary patches attend to fewer neighbours (masked, not wrapped).

        Returns: (B, num_heads, patch_len, head_dim)
        """
        B, num_heads, _, head_dim = q_patch.shape
        num_cls = self.num_cls
        window_size = self.window_size
        dilation = self.dilation
        window_len = 2 * window_size + 1
        context_len = num_cls + window_len

        pad = window_size * dilation
        k_padded = F.pad(k_patch, (0, 0, pad, pad))  # (B, num_heads, patch_len + 2*pad, head_dim)
        v_padded = F.pad(v_patch, (0, 0, pad, pad))


        # create indexing map
        patch_positions = torch.arange(patch_len, device=q_patch.device)           # (patch_len,)
        window_offsets = torch.arange(window_len, device=q_patch.device) * dilation  # (window_len,)
        gather_idx = patch_positions.unsqueeze(1) + window_offsets.unsqueeze(0)    # (patch_len, window_len)

        # replicate gather_idx map to match across batch, heads and emb_dim
        gather_idx_exp = (
            gather_idx
            .unsqueeze(0).unsqueeze(0)          # (1, 1, patch_len, window_len)
            .unsqueeze(-1)                       # (1, 1, patch_len, window_len, 1)
            .expand(B, num_heads, -1, -1, head_dim)  # (B, num_heads, patch_len, window_len, head_dim)
        )

        k_padded_exp = k_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1)  # (B, num_heads, patch_len+2*pad, window_len, head_dim)
        v_padded_exp = v_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1)  # (B, num_heads, patch_len+2*pad, window_len, head_dim)

        k_windows = torch.gather(k_padded_exp, 2, gather_idx_exp)  # (B, num_heads, patch_len, window_len, head_dim)
        v_windows = torch.gather(v_padded_exp, 2, gather_idx_exp)

        k_context = torch.cat(
            [k_cls.unsqueeze(2).expand(-1, -1, patch_len, -1, -1), k_windows], dim=3
        )  # (B, num_heads, patch_len, context_len, head_dim)
        v_context = torch.cat(
            [v_cls.unsqueeze(2).expand(-1, -1, patch_len, -1, -1), v_windows], dim=3
        )

        # mask out-of-bounds window positions
        patch_idx = torch.arange(patch_len, device=q_patch.device).unsqueeze(1)      # (patch_len, 1)
        window_pos = (torch.arange(window_len, device=q_patch.device) - window_size) * dilation  # (window_len,)
        in_bounds = (patch_idx + window_pos >= 0) & (patch_idx + window_pos < patch_len)  # (patch_len, window_len)

        # mask: 0 for valid, -inf for out-of-bounds. CLS positions always valid.
        window_mask = torch.zeros(patch_len, window_len, device=q_patch.device, dtype=q_patch.dtype)
        window_mask[~in_bounds] = float("-inf")
        cls_mask = torch.zeros(patch_len, num_cls, device=q_patch.device, dtype=q_patch.dtype)
        attn_mask = torch.cat([cls_mask, window_mask], dim=1)  # (patch_len, context_len)

        # Block-diagonal mask: patch i attends only to columns [i*context_len : (i+1)*context_len].
        # This keeps patch_len as the query sequence dimension, enabling MMA (requires seq_len >= 8).
        col_idx = (
            torch.arange(patch_len, device=q_patch.device)[:, None] * context_len
            + torch.arange(context_len, device=q_patch.device)[None, :]
        )  # (patch_len, context_len)
        block_mask = q_patch.new_full((patch_len, patch_len * context_len), float("-inf"))
        block_mask.scatter_(1, col_idx, attn_mask)

        flat = B * num_heads
        out = F.scaled_dot_product_attention(
            q_patch.reshape(flat, patch_len, head_dim),
            k_context.reshape(flat, patch_len * context_len, head_dim),
            v_context.reshape(flat, patch_len * context_len, head_dim),
            attn_mask=block_mask,  # (patch_len, patch_len*context_len) broadcasts over flat
            dropout_p=dropout_p,
        )      # (flat, patch_len, head_dim)

        return out.reshape(B, num_heads, patch_len, head_dim)


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
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = StaticSparseAttention(
            embed_dim, num_heads, num_cls, window_size, dilation, attn_dropout
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

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Encode a bag of patch embeddings and return slide-level logits.

        Args:
            x: Patch embeddings of shape ``(B, patch_len, in_features)``.

        Returns:
            Dict with ``"logits"`` of shape ``(B, out_features)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B = x.shape[0]

        # Project patch embeddings into the transformer's working dimension
        x = self.input_proj(x)  # (B, patch_len, embed_dim)

        # Prepend CLS tokens
        cls = self.cls_tokens.expand(B, -1, -1)  # (B, num_cls, embed_dim)
        x = torch.cat([cls, x], dim=1)           # (B, num_cls + patch_len, embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_tokens = x[:, :self.num_cls]                           # (B, num_cls, embed_dim)
        weights = torch.softmax(self.cls_pool(cls_tokens), dim=1)  # (B, num_cls, 1)
        cls_out = (weights * cls_tokens).sum(dim=1)             # (B, embed_dim)

        logits = self.head(cls_out)  # (B, out_features)

        return {"logits": logits}

# Notes:
# embed_dim / num_heads should equal 64
