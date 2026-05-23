"""
Approach C: Native Sparse Attention (NSA) for WSI slide encoding.
"""

import functools
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import BlockMask, flex_attention, noop_mask

from flash_attn import flash_attn_func

from sparse_wsi_vit.models.nsa_kernels.flash_topk import FlashTopK
from sparse_wsi_vit.models.nsa_kernels.topk_to_block_mask import topk_to_block_mask
from sparse_wsi_vit.models.vit_5.models_vit5 import RMSNorm, vit_models
from sparse_wsi_vit.models.vit_5.rope import VisionRotaryEmbedding


class NativeSparseAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        flash=True,
        rope_size=0,
        rope_reg_size=0,
        rope_dynamic=False,
        rope_dynamic_high=100_000,
        num_registers=0,
        reg_theta=10000,
        qk_norm=False,
        *,  # NSA specific arguments:
        top_k: int,
        block_size: int = 128,
        kernel_block_m: int = 64,
        kernel_block_n: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        assert num_registers % block_size == 0
        self.num_registers = num_registers
        self.rope = (
            VisionRotaryEmbedding(
                head_dim // 2,
                rope_size,
                dynamic=rope_dynamic,
                coord_high=rope_dynamic_high,
            )
            if rope_size > 0
            else None
        )
        self.rope_reg = (
            VisionRotaryEmbedding(head_dim // 2, rope_reg_size, theta=reg_theta)
            if rope_reg_size > 0
            else None
        )
        self.rope_reg_size = rope_reg_size
        self.block_size = block_size

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.g_proj = nn.Linear(dim, num_heads * 2, bias=False)

        self.flash_topk = FlashTopK(
            head_dim=head_dim,
            num_heads=num_heads,
            top_k=top_k,
            is_causal=False,
            softmax_scale=1 / math.sqrt(head_dim),  # standard scale
            m_block_size=kernel_block_m,
            n_block_size=kernel_block_n,
        )
        self.top_k = top_k
        self.flex_attn = {}
        for requires_grad in (False, True):
            flex = torch.compile(flex_attention, dynamic=True)

            @torch.compiler.disable(reason="autotune crashes", recursive=True)
            def _flex(q, k, v, **kwargs):
                torch._dynamo.mark_dynamic(q, 2)
                torch._dynamo.mark_dynamic(k, 2)
                torch._dynamo.mark_dynamic(v, 2)
                return flex(q, k, v, **kwargs)

            self.flex_attn[requires_grad] = _flex

    def forward(self, x, coords=None):
        B, N, C = x.shape

        assert N % self.block_size == 0
        N_BLK = N // self.block_size

        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)

        g_cmp, g_sel = (
            self.g_proj(x).view(B, N, 2, self.num_heads).softmax(2).unbind(dim=2)
        )

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            q_cls, q_img = q[:, : self.num_registers], q[:, self.num_registers :]
            k_cls, k_img = k[:, : self.num_registers], k[:, self.num_registers :]
            q_img = self.rope(q_img, coords)
            k_img = self.rope(k_img, coords)

            if self.rope_reg is not None:
                reg_start = self.num_registers - self.rope_reg_size**2
                q_cls, q_reg = q[:, :reg_start], q[:, reg_start : self.num_registers]
                k_cls, k_reg = k[:, :reg_start], k[:, reg_start : self.num_registers]
                q_reg = self.rope_reg(q_reg, coords)
                k_reg = self.rope_reg(k_reg, coords)

                q = torch.cat((q_cls, q_reg, q_img), dim=1)
                k = torch.cat((k_cls, k_reg, k_img), dim=1)
            else:
                q = torch.cat((q_cls, q_img), dim=1)
                k = torch.cat((k_cls, k_img), dim=1)

        q_cmp = q.view(B, N_BLK, self.block_size, self.num_heads, head_dim).mean(2)
        k_cmp = k.view(B, N_BLK, self.block_size, self.num_heads, head_dim).mean(2)
        v_cmp = v.view(B, N_BLK, self.block_size, self.num_heads, head_dim).mean(2)
        _, idx_cmp, _, _ = self.flash_topk(q_cmp, k_cmp, v_cmp)
        o_cmp = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k_cmp.transpose(1, 2), v_cmp.transpose(1, 2)
        ).transpose(1, 2)

        # [B, H, T, D] -> [B, T, H, D] and back again
        o_sel = self.flex_attn[q.requires_grad](
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=topk_to_block_mask(idx_cmp, block_size=self.block_size),
        ).transpose(1, 2)  # ty:ignore[unresolved-attribute]

        x = (
            o_cmp.view(B, N_BLK, self.block_size, self.num_heads, head_dim)
            * g_cmp.view(B, N_BLK, self.block_size, self.num_heads, 1)
            + o_sel.view(B, N_BLK, self.block_size, self.num_heads, head_dim)
            * g_sel.view(B, N_BLK, self.block_size, self.num_heads, 1)
        ).view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _RegpoolVit5(vit_models):
    def __init__(self, block_size: int, embed_dim: int, num_classes: int, **kwargs):
        super().__init__(embed_dim=embed_dim, num_classes=0, **kwargs)
        self.block_size = block_size
        self.head = nn.Linear(2 * embed_dim, num_classes)
        self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward_features(self, x, coords=None):
        assert coords is not None
        assert self.reg_token is not None

        B, N, D = x.shape
        x = self.patch_embed(x)

        registers = self.reg_token.expand(B, self.block_size - 1, -1)
        cls = self.cls_token.expand(B, -1, -1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        # crop the last couple positions, not optimal, but it shouldn't matter much.
        x = x[:, : N // self.block_size * self.block_size]
        coords = coords[:, : N // self.block_size * self.block_size]

        x = torch.cat((cls, registers, x), dim=1)

        for i, blk in enumerate(self.blocks):
            # print(f"block {i}! {torch.cuda.memory_allocated()/1e9}GB alloc")
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, coords, use_reentrant=False
                )
            else:
                x = blk(x, coords)

        x = self.norm(x)
        return torch.cat((x[:, 0], x[:, 1 : 1 + self.reg_token.size(1)].mean(1)), dim=1)


class NSAViTSlideEncoder(nn.Module):
    """VitDensePreEmbedded but with Native Sparse Attention blocks"""

    def __init__(
        self,
        in_features: int = 1280,
        out_features: int = 1,
        downproj: int | None = None,
        top_k: int = 16,
        block_size: int = 128,
        num_registers: int | None = None,
        reg_theta=100,
        qk_norm=True,
        qkv_bias=False,
        rope_reg: bool | str = "partial",
        **vit5_kwargs,  # checkpoint_activations, depth, rope_dynamic_high, num_heads
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.vit5 = _RegpoolVit5(
            num_classes=out_features,
            drop_rate=0.0,
            drop_path_rate=0.1,
            pre_embedded_input=True,
            ape=False,  # it's learnable, which breaks with variable size inputs.
            patch_size=1,  # each token is a patch, effectively
            rope=True,
            rope_dynamic=True,
            rope_reg=rope_reg,
            embed_dim=in_features if downproj is None else downproj,
            Attention_block=functools.partial(
                NativeSparseAttention,
                top_k=top_k,
                block_size=block_size,
            ),
            num_registers=block_size if num_registers is None else num_registers,
            reg_theta=reg_theta,
            qk_norm=qk_norm,
            qkv_bias=qkv_bias,
            block_size=block_size,
            **vit5_kwargs,
        )
        self.down = nn.Linear(in_features, downproj) if downproj else None

        if self.down:
            nn.init.orthogonal_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> dict:
        """Run a forward pass over a feature bag.

        Args:
            x: Instance features of shape (B, N, D) or (N, D). A batch dimension
                is added automatically when the input is 2-D.
            coords: Coordinates for each feature embedding in the WSI, shape (B, N, 2)
                or (N, 2).

        Returns:
            A dictionary with:
                - ``"logits"`` (torch.Tensor): Bag-level predictions, shape (B, out_features).
                - ``"attention"`` (torch.Tensor, optional): Attention weights,
                  shape (B, num_branches, N). Only present when *return_attention* is True.

        Raises:
            ValueError: If *x* is not 2-D or 3-D, or if the feature dimension does
                not match ``in_features``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {x.shape=}"
            )
        if coords.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D (N, D) or 3-D (B, N, D) input, got {coords.shape=}"
            )

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.size(1) != x.size(1):
            raise ValueError(f"{x.shape=} and {coords.shape} must agree on N")

        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.in_features}, got {D}."
            )

        if self.down is not None:
            x = self.down(x)

        logits = self.vit5(x, coords)  # (B, out_features)

        out = {"logits": logits}
        return out
