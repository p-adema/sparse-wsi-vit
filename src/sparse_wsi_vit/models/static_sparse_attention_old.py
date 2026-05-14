"""
Approach A: Static sparse attention for WSI slide encoding.

OLD VERSION — Longformer-style with F.scaled_dot_product_attention,
manual windowed chunking, and dilation. Kept for comparison with the
current FlexAttention-based implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class Rope2D(nn.Module):
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
        self.register_buffer("inv_freq", inv_freq)
        self.coord_high = coord_high

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        xy = coords.float() / self.coord_high
        freq_x = torch.einsum("bl, f -> blf", xy[..., 0], self.inv_freq)
        freq_y = torch.einsum("bl, f -> blf", xy[..., 1], self.inv_freq)
        freq_x = freq_x.repeat_interleave(2, dim=-1)
        freq_y = freq_y.repeat_interleave(2, dim=-1)
        freqs = torch.cat([freq_x, freq_y], dim=-1)
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        return x * cos + _rotate_half(x) * sin


class StaticSparseAttention(nn.Module):
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
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
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
        batch_size, seq_len, embed_dim = x.shape
        num_cls = self.num_cls
        patch_len = seq_len - num_cls

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if coords is not None:
            q = torch.cat([q[:, :, :num_cls], self.rope(q[:, :, num_cls:], coords)], dim=2)
            k = torch.cat([k[:, :, :num_cls], self.rope(k[:, :, num_cls:], coords)], dim=2)

        dropout_p = self.attn_dropout if self.training else 0.0

        q_cls, k_cls, v_cls = q[:, :, :num_cls], k[:, :, :num_cls], v[:, :, :num_cls]
        q_patch = q[:, :, num_cls:]

        cls_out = F.scaled_dot_product_attention(q_cls, k, v, dropout_p=dropout_p)

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
        q_patch: Tensor,
        k_cls: Tensor,
        v_cls: Tensor,
        k_patch: Tensor,
        v_patch: Tensor,
        patch_len: int,
        dropout_p: float,
    ) -> Tensor:
        batch_size, num_heads, _, head_dim = q_patch.shape
        num_cls = self.num_cls
        window_size = self.window_size
        dilation = self.dilation
        chunk_size = self.chunk_size
        window_len = 2 * window_size + 1
        context_len = num_cls + window_len
        pad = window_size * dilation

        k_padded = F.pad(k_patch, (0, 0, pad, pad))
        v_padded = F.pad(v_patch, (0, 0, pad, pad))

        window_pos = (torch.arange(window_len, device=q_patch.device) - window_size) * dilation

        use_unfold = (dilation == 1)
        if use_unfold:
            k_all_windows = k_padded.unfold(2, window_len, 1).permute(0, 1, 2, 4, 3)
            v_all_windows = v_padded.unfold(2, window_len, 1).permute(0, 1, 2, 4, 3)
        else:
            offsets = torch.arange(window_len, device=q_patch.device) * dilation

        k_cls_exp = k_cls.unsqueeze(2)
        v_cls_exp = v_cls.unsqueeze(2)

        chunks: list[Tensor] = []
        for start in range(0, patch_len, chunk_size):
            end = min(start + chunk_size, patch_len)
            chunk = end - start

            if use_unfold:
                k_windows = k_all_windows[:, :, start:end]
                v_windows = v_all_windows[:, :, start:end]
            else:
                chunk_positions = torch.arange(start, end, device=q_patch.device)
                gather_idx = chunk_positions.unsqueeze(1) + offsets.unsqueeze(0)
                gather_idx_exp = (
                    gather_idx
                    .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                    .expand(batch_size, num_heads, -1, -1, head_dim)
                )
                k_windows = torch.gather(
                    k_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1),
                    2, gather_idx_exp,
                )
                v_windows = torch.gather(
                    v_padded.unsqueeze(3).expand(-1, -1, -1, window_len, -1),
                    2, gather_idx_exp,
                )

            k_context = torch.cat(
                [k_cls_exp.expand(-1, -1, chunk, -1, -1), k_windows], dim=3
            )
            v_context = torch.cat(
                [v_cls_exp.expand(-1, -1, chunk, -1, -1), v_windows], dim=3
            )

            chunk_positions_for_mask = torch.arange(start, end, device=q_patch.device)
            in_bounds = (chunk_positions_for_mask.unsqueeze(1) + window_pos >= 0) & \
                        (chunk_positions_for_mask.unsqueeze(1) + window_pos < patch_len)
            window_mask = q_patch.new_zeros(chunk, window_len)
            window_mask[~in_bounds] = float("-inf")
            attn_mask = torch.cat(
                [q_patch.new_zeros(chunk, num_cls), window_mask], dim=1
            )

            num_queries_flat = batch_size * num_heads * chunk
            q_flat = q_patch[:, :, start:end].reshape(num_queries_flat, 1, head_dim)
            k_flat = k_context.reshape(num_queries_flat, context_len, head_dim)
            v_flat = v_context.reshape(num_queries_flat, context_len, head_dim)

            mask_flat = (
                attn_mask
                .unsqueeze(1)
                .unsqueeze(0)
                .expand(batch_size * num_heads, -1, -1, -1)
                .reshape(num_queries_flat, 1, context_len)
            )

            out_chunk = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                attn_mask=mask_flat,
                dropout_p=dropout_p,
            )

            chunks.append(out_chunk.reshape(batch_size, num_heads, chunk, head_dim))

        return torch.cat(chunks, dim=2)


class StaticSparseAttentionAdapter(nn.Module):
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


class StaticSparseViTBlock(nn.Module):
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
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.blocks = nn.ModuleList([
            StaticSparseViTBlock(
                embed_dim=embed_dim, num_heads=num_heads, num_cls=num_cls,
                window_size=window_size, dilation=dilation,
                expansion_factor=expansion_factor, attn_dropout=attn_dropout,
                proj_dropout=proj_dropout, chunk_size=chunk_size,
                rope_theta=rope_theta, rope_coord_high=rope_coord_high,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_pool = nn.Linear(embed_dim, 1, bias=False)
        self.head = nn.Linear(embed_dim, out_features)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> dict[str, Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        x = self.input_proj(x)
        cls = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x, coords)
        x = self.norm(x)
        cls_tokens = x[:, :self.num_cls]
        weights = torch.softmax(self.cls_pool(cls_tokens), dim=1)
        cls_out = (weights * cls_tokens).sum(dim=1)
        return {"logits": self.head(cls_out)}
