"""
Block-sparse FlashAttention Triton kernels for static sparse attention.

Two kernel launches (forward + backward) replace the Python for-loop over chunks.
Compatible with torch.utils.checkpoint (no torch.compile dependency).

Attention pattern (handled outside these kernels):
- CLS tokens: F.scaled_dot_product_attention with full K, V
- Patch tokens: block-sparse Triton kernel (CLS blocks + window chunks)
"""

import math

import torch
from torch import Tensor

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Block mask construction
# ---------------------------------------------------------------------------

def build_block_mask(
    seq_len: int,
    num_cls: int,
    chunk_size: int,
    window_size: int,
    block_size: int,
    device: torch.device,
) -> Tensor:
    """Build KV-block index array for block-sparse patch attention.

    For each patch Q-block, lists the full-sequence KV-block indices it
    attends to (CLS blocks + windowed chunk blocks).  Padding entries are -1.

    Args:
        seq_len:     Full sequence length (num_cls + num_patches).
        num_cls:     Number of CLS tokens at the start.
        chunk_size:  Patches per logical chunk (Hilbert-ordered).
        window_size: Neighbouring chunks on each side.
        block_size:  Triton kernel tile size.
        device:      Target device.

    Returns:
        ``int32`` tensor of shape ``(num_patch_q_blocks, kv_per_patch)``.
    """
    assert chunk_size % block_size == 0, (
        f"chunk_size ({chunk_size}) must be a multiple of block_size ({block_size})"
    )

    num_patches = seq_len - num_cls
    num_patch_q_blocks = (num_patches + block_size - 1) // block_size
    num_kv_blocks = (seq_len + block_size - 1) // block_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    blocks_per_chunk = chunk_size // block_size
    num_cls_blocks = (num_cls + block_size - 1) // block_size if num_cls > 0 else 0

    # Worst-case width: CLS blocks (outside window) + full window
    max_window_blocks = (2 * window_size + 1) * blocks_per_chunk
    kv_per_patch = num_cls_blocks + max_window_blocks

    kv_indices = torch.full(
        (max(num_patch_q_blocks, 1), kv_per_patch), -1, dtype=torch.int32,
    )

    cls_set = set(range(num_cls_blocks))

    for qi in range(num_patch_q_blocks):
        full_seq_start = num_cls + qi * block_size
        chunk_id = full_seq_start // chunk_size
        lo_chunk = max(0, chunk_id - window_size)
        hi_chunk = min(num_chunks - 1, chunk_id + window_size)

        lo_block = lo_chunk * blocks_per_chunk
        hi_block = min((hi_chunk + 1) * blocks_per_chunk, num_kv_blocks)

        all_blocks = sorted(cls_set | set(range(lo_block, hi_block)))
        for j, blk in enumerate(all_blocks):
            kv_indices[qi, j] = blk

    return kv_indices.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Forward Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _block_sparse_attn_fwd(
    # Tensor pointers
    Q, K, V, Out, LSE,
    KV_IDX,
    # Q strides  (patch-only tensor, possibly non-contiguous)
    sq0, sq1, sq2, sq3,
    # K strides  (full-sequence tensor)
    sk0, sk1, sk2, sk3,
    # V strides
    sv0, sv1, sv2, sv3,
    # Out strides (contiguous, same shape as Q)
    so0, so1, so2, so3,
    # LSE strides (batch, heads, num_patches)
    sl0, sl1, sl2,
    # Scalar dims
    num_patches,          # Q length
    seq_len,              # KV length
    num_heads,
    scale,
    # Compile-time constants
    KV_PER_PATCH: tl.constexpr,
    BLOCK_SIZE:   tl.constexpr,
    HEAD_DIM:     tl.constexpr,
):
    pid_bh = tl.program_id(0)          # batch_idx * num_heads + head_idx
    pid_q  = tl.program_id(1)          # patch Q-block index

    batch_idx = pid_bh // num_heads
    head_idx  = pid_bh % num_heads

    # ---- Q tile --------------------------------------------------------
    q_start = pid_q * BLOCK_SIZE
    q_range = q_start + tl.arange(0, BLOCK_SIZE)
    q_mask  = q_range < num_patches
    d_range = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q
              + batch_idx * sq0
              + head_idx  * sq1
              + q_range[:, None] * sq2
              + d_range[None, :] * sq3)
    q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # ---- online softmax accumulators -----------------------------------
    row_max = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc     = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)

    # ---- iterate over KV blocks ----------------------------------------
    for j in tl.static_range(KV_PER_PATCH):
        kv_block = tl.load(KV_IDX + pid_q * KV_PER_PATCH + j)
        valid    = kv_block >= 0
        safe_blk = tl.maximum(kv_block, 0)

        kv_start = safe_blk * BLOCK_SIZE
        kv_range = kv_start + tl.arange(0, BLOCK_SIZE)
        kv_mask  = (kv_range < seq_len) & valid

        # Load K tile
        k_ptrs = (K
                  + batch_idx * sk0
                  + head_idx  * sk1
                  + kv_range[:, None] * sk2
                  + d_range[None, :] * sk3)
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # S = Q K^T * scale   →  (BLOCK_SIZE, BLOCK_SIZE)
        scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))

        # online softmax update
        block_max  = tl.max(scores, axis=1)
        new_max    = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        exp_scores = tl.exp(scores - new_max[:, None])
        exp_scores = tl.where(kv_mask[None, :], exp_scores, 0.0)

        row_sum = row_sum * correction + tl.sum(exp_scores, axis=1)
        acc     = acc * correction[:, None]

        # Load V tile and accumulate
        v_ptrs = (V
                  + batch_idx * sv0
                  + head_idx  * sv1
                  + kv_range[:, None] * sv2
                  + d_range[None, :] * sv3)
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(exp_scores.to(v_tile.dtype), v_tile).to(tl.float32)

        row_max = new_max

    # ---- normalise and store -------------------------------------------
    row_sum_safe = tl.maximum(row_sum, 1e-6)
    out_tile = acc / row_sum_safe[:, None]

    o_ptrs = (Out
              + batch_idx * so0
              + head_idx  * so1
              + q_range[:, None] * so2
              + d_range[None, :] * so3)
    tl.store(o_ptrs, out_tile.to(Q.dtype.element_ty), mask=q_mask[:, None])

    # store log-sum-exp for backward
    lse_val = row_max + tl.log(row_sum_safe)
    lse_ptrs = LSE + batch_idx * sl0 + head_idx * sl1 + q_range * sl2
    tl.store(lse_ptrs, lse_val, mask=q_mask)


# ---------------------------------------------------------------------------
# Backward Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _block_sparse_attn_bwd(
    # Input tensors
    Q, K, V, Out, dOut, LSE,
    # Gradient outputs
    dQ, dK, dV,
    # Block mask
    KV_IDX,
    # Q strides
    sq0, sq1, sq2, sq3,
    # K strides
    sk0, sk1, sk2, sk3,
    # V strides
    sv0, sv1, sv2, sv3,
    # Out strides  (also used for dOut — same layout)
    so0, so1, so2, so3,
    # LSE strides
    sl0, sl1, sl2,
    # dQ strides   (contiguous, same shape as Q)
    sdq0, sdq1, sdq2, sdq3,
    # dK strides   (fp32, contiguous, full-seq shape)
    sdk0, sdk1, sdk2, sdk3,
    # dV strides
    sdv0, sdv1, sdv2, sdv3,
    # Scalar dims
    num_patches,
    seq_len,
    num_heads,
    scale,
    # Compile-time constants
    KV_PER_PATCH: tl.constexpr,
    BLOCK_SIZE:   tl.constexpr,
    HEAD_DIM:     tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx  = pid_bh % num_heads

    q_start = pid_q * BLOCK_SIZE
    q_range = q_start + tl.arange(0, BLOCK_SIZE)
    q_mask  = q_range < num_patches
    d_range = tl.arange(0, HEAD_DIM)

    # ---- load tiles that persist across the KV loop --------------------
    q_ptrs = (Q + batch_idx * sq0 + head_idx * sq1
              + q_range[:, None] * sq2 + d_range[None, :] * sq3)
    q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    o_ptrs = (Out + batch_idx * so0 + head_idx * so1
              + q_range[:, None] * so2 + d_range[None, :] * so3)
    o_tile = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0)

    do_ptrs = (dOut + batch_idx * so0 + head_idx * so1
               + q_range[:, None] * so2 + d_range[None, :] * so3)
    do_tile = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0)

    lse_ptrs = LSE + batch_idx * sl0 + head_idx * sl1 + q_range * sl2
    lse = tl.load(lse_ptrs, mask=q_mask, other=0.0)

    # D_i = rowsum(dO * O)
    delta = tl.sum(do_tile * o_tile, axis=1)   # (BLOCK_SIZE,)

    # ---- dQ accumulator ------------------------------------------------
    dq_acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)

    # ---- iterate over KV blocks ----------------------------------------
    for j in tl.static_range(KV_PER_PATCH):
        kv_block = tl.load(KV_IDX + pid_q * KV_PER_PATCH + j)
        valid    = kv_block >= 0
        safe_blk = tl.maximum(kv_block, 0)

        kv_start = safe_blk * BLOCK_SIZE
        kv_range = kv_start + tl.arange(0, BLOCK_SIZE)
        kv_mask  = (kv_range < seq_len) & valid

        # Load K, V
        k_ptrs = (K + batch_idx * sk0 + head_idx * sk1
                  + kv_range[:, None] * sk2 + d_range[None, :] * sk3)
        v_ptrs = (V + batch_idx * sv0 + head_idx * sv1
                  + kv_range[:, None] * sv2 + d_range[None, :] * sv3)
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Recompute attention weights  P = exp(S - LSE)
        scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))
        attn_w = tl.exp(scores - lse[:, None])
        attn_w = tl.where(kv_mask[None, :], attn_w, 0.0)

        # dP = dO @ V^T
        d_attn = tl.dot(do_tile, tl.trans(v_tile))

        # dS = P * (dP - D_i)
        d_scores = attn_w * (d_attn - delta[:, None])
        d_scores = tl.where(kv_mask[None, :], d_scores, 0.0)

        # dQ += scale * dS @ K   (accumulated in registers, stored once)
        dq_acc += tl.dot(d_scores.to(k_tile.dtype), k_tile).to(tl.float32) * scale

        # dK += scale * dS^T @ Q   (atomic — multiple Q-blocks share KV)
        dk_tile = tl.dot(tl.trans(d_scores.to(q_tile.dtype)), q_tile).to(tl.float32) * scale
        dk_ptrs = (dK + batch_idx * sdk0 + head_idx * sdk1
                   + kv_range[:, None] * sdk2 + d_range[None, :] * sdk3)
        tl.atomic_add(dk_ptrs, dk_tile, mask=kv_mask[:, None])

        # dV += P^T @ dO   (atomic)
        dv_tile = tl.dot(tl.trans(attn_w.to(do_tile.dtype)), do_tile).to(tl.float32)
        dv_ptrs = (dV + batch_idx * sdv0 + head_idx * sdv1
                   + kv_range[:, None] * sdv2 + d_range[None, :] * sdv3)
        tl.atomic_add(dv_ptrs, dv_tile, mask=kv_mask[:, None])

    # ---- store dQ (no atomics — one program per Q-block) ---------------
    dq_ptrs = (dQ + batch_idx * sdq0 + head_idx * sdq1
               + q_range[:, None] * sdq2 + d_range[None, :] * sdq3)
    tl.store(dq_ptrs, dq_acc.to(Q.dtype.element_ty), mask=q_mask[:, None])


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------

class BlockSparseAttentionFunction(torch.autograd.Function):
    """Block-sparse FlashAttention via Triton."""

    @staticmethod
    def forward(ctx, q_patch, k, v, kv_indices, seq_len, block_size):
        """
        Args:
            q_patch:    (batch, heads, num_patches, head_dim)  bf16
            k:          (batch, heads, seq_len, head_dim)      bf16
            v:          (batch, heads, seq_len, head_dim)      bf16
            kv_indices: (num_patch_q_blocks, kv_per_patch)     int32
            seq_len:    int  — full sequence length (CLS + patches)
            block_size: int  — Triton tile size
        """
        batch_size, num_heads, num_patches, head_dim = q_patch.shape

        out = torch.empty(
            batch_size, num_heads, num_patches, head_dim,
            dtype=q_patch.dtype, device=q_patch.device,
        )
        lse = torch.empty(
            batch_size, num_heads, num_patches,
            dtype=torch.float32, device=q_patch.device,
        )

        kv_per_patch = kv_indices.shape[1]
        num_patch_q_blocks = (num_patches + block_size - 1) // block_size
        scale = 1.0 / math.sqrt(head_dim)

        grid = (batch_size * num_heads, num_patch_q_blocks)

        _block_sparse_attn_fwd[grid](
            q_patch, k, v, out, lse,
            kv_indices,
            # strides
            q_patch.stride(0), q_patch.stride(1), q_patch.stride(2), q_patch.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            # dims
            num_patches, seq_len, num_heads, scale,
            # constexpr
            KV_PER_PATCH=kv_per_patch,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
        )

        ctx.save_for_backward(q_patch, k, v, out, lse, kv_indices)
        ctx.seq_len = seq_len
        ctx.block_size = block_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        q_patch, k, v, out, lse, kv_indices = ctx.saved_tensors
        batch_size, num_heads, num_patches, head_dim = q_patch.shape
        seq_len = ctx.seq_len
        block_size = ctx.block_size
        scale = ctx.scale

        kv_per_patch = kv_indices.shape[1]
        num_patch_q_blocks = (num_patches + block_size - 1) // block_size

        dq = torch.zeros_like(q_patch)
        # fp32 accumulators for atomic safety
        dk = torch.zeros(
            batch_size, num_heads, seq_len, head_dim,
            dtype=torch.float32, device=q_patch.device,
        )
        dv = torch.zeros(
            batch_size, num_heads, seq_len, head_dim,
            dtype=torch.float32, device=q_patch.device,
        )

        d_out = d_out.contiguous()

        grid = (batch_size * num_heads, num_patch_q_blocks)

        _block_sparse_attn_bwd[grid](
            q_patch, k, v, out, d_out, lse,
            dq, dk, dv,
            kv_indices,
            # Q strides
            q_patch.stride(0), q_patch.stride(1), q_patch.stride(2), q_patch.stride(3),
            # K strides
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            # V strides
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            # Out / dOut strides
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            # LSE strides
            lse.stride(0), lse.stride(1), lse.stride(2),
            # dQ strides
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            # dK strides (fp32)
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            # dV strides (fp32)
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            # dims
            num_patches, seq_len, num_heads, scale,
            # constexpr
            KV_PER_PATCH=kv_per_patch,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
        )

        # cast gradients back to input dtype
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

        # return order: q_patch, k, v, kv_indices, seq_len, block_size
        return dq, dk, dv, None, None, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def block_sparse_attention(
    q_patch: Tensor,
    k: Tensor,
    v: Tensor,
    kv_indices: Tensor,
    seq_len: int,
    block_size: int = 128,
) -> Tensor:
    """Block-sparse attention for patch tokens via Triton.

    CLS tokens are handled separately by the caller with standard SDPA.

    Args:
        q_patch:    ``(batch, heads, num_patches, head_dim)`` — patch queries.
        k:          ``(batch, heads, seq_len, head_dim)``     — full-sequence keys.
        v:          ``(batch, heads, seq_len, head_dim)``     — full-sequence values.
        kv_indices: ``(num_patch_q_blocks, kv_per_patch)``    — block mask (int32).
        seq_len:    Full sequence length (CLS + patches).
        block_size: Triton tile size (default 128 for H100).

    Returns:
        ``(batch, heads, num_patches, head_dim)``
    """
    return BlockSparseAttentionFunction.apply(
        q_patch, k, v, kv_indices, seq_len, block_size,
    )
