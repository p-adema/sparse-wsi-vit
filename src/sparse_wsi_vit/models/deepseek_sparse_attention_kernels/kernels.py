"""
Approach B: DeepSeek Sparse Attention (DSA) for WSI slide encoding. 
This file contains the kernels that are used in the forward and backward
passes of both the Lightning Indexer and the Sparse Attention.
"""


import triton
import triton.language as tl
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Indexer forward kernel helpers — shared top-K merge logic
# ---------------------------------------------------------------------------

@triton.jit
def _topk_merge_and_store(
    block_scores, offs_k, mask_k, T,
    topk_scores, topk_indices,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, TOP_K: tl.constexpr,
):
    """Merge a (BLOCK_Q, BLOCK_K) score block into the running top-K buffer."""
    block_scores = tl.maximum(block_scores, 0.0)
    block_scores = tl.where(mask_k[None, :], block_scores, float("-inf"))

    for i in range(BLOCK_K):
        col_mask = tl.arange(0, BLOCK_K) == i

        candidate_score = tl.sum(
            tl.where(col_mask[None, :], block_scores, 0.0), axis=1
        )
        candidate_idx = tl.minimum(
            tl.sum(tl.where(col_mask, offs_k, 0)), T - 1
        )

        min_score     = tl.min(topk_scores, axis=1)
        raw_is_min    = topk_scores == min_score[:, None]
        cumsum        = tl.cumsum(raw_is_min.to(tl.int32), axis=1)
        slot_is_min   = raw_is_min & (cumsum == 1)
        should_insert = candidate_score > min_score

        topk_scores = tl.where(
            should_insert[:, None] & slot_is_min,
            candidate_score[:, None], topk_scores,
        )
        topk_indices = tl.where(
            should_insert[:, None] & slot_is_min,
            candidate_idx, topk_indices,
        )

    return topk_scores, topk_indices


# ---------------------------------------------------------------------------
# Indexer forward kernel — BF16 / FP32 path (original)
# Computes weighted dot-product scores, runs top-k selection.
# Writes IDX (int), VALID (float), SCORES (float).
# ---------------------------------------------------------------------------

@triton.jit
def _indexer_fwd(
    Q_ptr, K_ptr, IDX_ptr, VALID_ptr, SCORES_ptr, W_ptr,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    sib, sit, sik,
    svlb, svlt, svlk,
    ssb, sst, ssk,
    B, T, H, D,
    TOP_K:           tl.constexpr,
    BLOCK_Q:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_INDEXER_H: tl.constexpr,
    BLOCK_D:         tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_t  = tl.program_id(1)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_h = tl.arange(0, BLOCK_INDEXER_H)
    mask_t = offs_t < T

    W = tl.load(W_ptr + offs_h)  # (H,)

    topk_scores  = tl.full((BLOCK_Q, TOP_K), float("-inf"), dtype=tl.float32)
    topk_indices = tl.full((BLOCK_Q, TOP_K), 0,            dtype=tl.int32)

    for k_block in range(0, tl.cdiv(T, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < T

        block_scores = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            Q = tl.load(
                Q_ptr + pid_b*sqb + offs_t[:, None, None]*sqt
                      + offs_h[None, :, None]*sqh + offs_d[None, None, :]*sqd,
                mask=mask_t[:, None, None] & mask_d[None, None, :], other=0.0,
            )  # (BLOCK_Q, H, BLOCK_D)
            K = tl.load(
                K_ptr + pid_b*skb + offs_k[:, None, None]*skt
                      + offs_h[None, :, None]*skh + offs_d[None, None, :]*skd,
                mask=mask_k[:, None, None] & mask_d[None, None, :], other=0.0,
            )  # (BLOCK_K, H, BLOCK_D)

            Wq = tl.sum(Q * W[None, :, None], axis=1)  # (BLOCK_Q, BLOCK_D)
            Wk = tl.sum(K * W[None, :, None], axis=1)  # (BLOCK_K, BLOCK_D)
            block_scores += tl.dot(Wq, tl.trans(Wk))   # (BLOCK_Q, BLOCK_K)

        topk_scores, topk_indices = _topk_merge_and_store(
            block_scores, offs_k, mask_k, T,
            topk_scores, topk_indices,
            BLOCK_Q, BLOCK_K, TOP_K,
        )

    slot_offs = tl.arange(0, TOP_K)
    tl.store(
        IDX_ptr + pid_b*sib + offs_t[:, None]*sit + slot_offs[None, :]*sik,
        topk_indices, mask=mask_t[:, None],
    )
    tl.store(
        VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + slot_offs[None, :]*svlk,
        (topk_scores > float("-inf")).to(tl.float32), mask=mask_t[:, None],
    )
    tl.store(
        SCORES_ptr + pid_b*ssb + offs_t[:, None]*sst + slot_offs[None, :]*ssk,
        tl.where(topk_scores > float("-inf"), topk_scores, 0.0),
        mask=mask_t[:, None],
    )


# ---------------------------------------------------------------------------
# Indexer forward kernel — FP8 path
#
# Q and K are passed as FP8 (float8_e4m3fn).  We load them and immediately
# upcast to FP32 before any arithmetic — accumulation is always FP32.
# This halves memory bandwidth for Q/K loads vs BF16, ~4× vs FP32.
#
# On A100: no native FP8 tensor cores, but bandwidth savings still apply.
# On H100: FP8 tensor cores fire for tl.dot, giving additional speedup.
#
# The backward stays in BF16/FP32 (gradients must be full precision).
# We save the original BF16 q_proj/k_proj for the backward, not the FP8 copy.
# ---------------------------------------------------------------------------

@triton.jit
def _indexer_fwd_fp8(
    Q_ptr, K_ptr, IDX_ptr, VALID_ptr, SCORES_ptr, W_ptr,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    sib, sit, sik,
    svlb, svlt, svlk,
    ssb, sst, ssk,
    B, T, H, D,
    TOP_K:           tl.constexpr,
    BLOCK_Q:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_INDEXER_H: tl.constexpr,
    BLOCK_D:         tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_t  = tl.program_id(1)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_h = tl.arange(0, BLOCK_INDEXER_H)
    mask_t = offs_t < T

    # W stays FP32 — it's tiny (H scalars) and not bandwidth-bound
    W = tl.load(W_ptr + offs_h)  # (H,) fp32

    topk_scores  = tl.full((BLOCK_Q, TOP_K), float("-inf"), dtype=tl.float32)
    topk_indices = tl.full((BLOCK_Q, TOP_K), 0,            dtype=tl.int32)

    for k_block in range(0, tl.cdiv(T, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < T

        block_scores = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load FP8, upcast to FP32 immediately for safe arithmetic
            Q_fp8 = tl.load(
                Q_ptr + pid_b*sqb + offs_t[:, None, None]*sqt
                      + offs_h[None, :, None]*sqh + offs_d[None, None, :]*sqd,
                mask=mask_t[:, None, None] & mask_d[None, None, :], other=0.0,
            )  # (BLOCK_Q, H, BLOCK_D) fp8
            Q = Q_fp8.to(tl.float32)

            K_fp8 = tl.load(
                K_ptr + pid_b*skb + offs_k[:, None, None]*skt
                      + offs_h[None, :, None]*skh + offs_d[None, None, :]*skd,
                mask=mask_k[:, None, None] & mask_d[None, None, :], other=0.0,
            )  # (BLOCK_K, H, BLOCK_D) fp8
            K = K_fp8.to(tl.float32)

            Wq = tl.sum(Q * W[None, :, None], axis=1)  # (BLOCK_Q, BLOCK_D)
            Wk = tl.sum(K * W[None, :, None], axis=1)  # (BLOCK_K, BLOCK_D)
            block_scores += tl.dot(Wq, tl.trans(Wk))   # (BLOCK_Q, BLOCK_K) fp32

        topk_scores, topk_indices = _topk_merge_and_store(
            block_scores, offs_k, mask_k, T,
            topk_scores, topk_indices,
            BLOCK_Q, BLOCK_K, TOP_K,
        )

    slot_offs = tl.arange(0, TOP_K)
    tl.store(
        IDX_ptr + pid_b*sib + offs_t[:, None]*sit + slot_offs[None, :]*sik,
        topk_indices, mask=mask_t[:, None],
    )
    tl.store(
        VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + slot_offs[None, :]*svlk,
        (topk_scores > float("-inf")).to(tl.float32), mask=mask_t[:, None],
    )
    tl.store(
        SCORES_ptr + pid_b*ssb + offs_t[:, None]*sst + slot_offs[None, :]*ssk,
        tl.where(topk_scores > float("-inf"), topk_scores, 0.0),
        mask=mask_t[:, None],
    )


# ---------------------------------------------------------------------------
# Indexer backward kernels
#
# Forward:  score[b,q,k] = sum_h W[h] * sum_d Q[b,q,h,d] * K[b,k,h,d]
#
# Only the TOP_K selected (q,k) pairs receive non-zero dScore (via dRW from
# the attention backward flowing through sigmoid).  We store dScore into a
# sparse (B, T, TOP_K) buffer aligned with IDX, then scatter into dQ, dK.
#
# dQ[b,q,h,d] += sum_{slot} dScore[b,q,slot] * W[h] * K[b, idx[b,q,slot], h, d]
# dK[b,k,h,d] += sum_{q,slot: idx[b,q,slot]==k} dScore[b,q,slot] * W[h] * Q[b,q,h,d]
# dW[h]        = sum_{b,q,slot} dScore[b,q,slot] * sum_d Q[b,q,h,d]*K[b,idx[..],h,d]
# ---------------------------------------------------------------------------

@triton.jit
def _indexer_bwd_dQ(
    dSCORES_ptr, Q_ptr, K_ptr, W_ptr, IDX_ptr, VALID_ptr, dQ_ptr,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    sib, sit, sik,
    svlb, svlt, svlk,
    sdsb, sdst, sdsk,
    B, T, H, D, TOP_K,
    BLOCK_Q:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_D:         tl.constexpr,
    BLOCK_INDEXER_H: tl.constexpr,
):
    # One program per (batch, query-tile).
    # We loop over heads explicitly to stay within Triton's 3D tensor limit.
    # dQ[q, h, d] += sum_{slot} dS[q,slot] * W[h] * K[idx[q,slot], h, d]
    pid_b  = tl.program_id(0)
    pid_t  = tl.program_id(1)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_t = offs_t < T

    for h in range(BLOCK_INDEXER_H):
        W_h = tl.load(W_ptr + h)  # scalar

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # dQ_acc: (BLOCK_Q, BLOCK_D) — contribution from all slots for head h
            dQ_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

            for slot_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
                offs_k = slot_block * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < TOP_K

                idx = tl.load(
                    IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
                    mask=mask_t[:, None] & mask_k[None, :], other=0,
                )  # (BLOCK_Q, BLOCK_K)
                valid = tl.load(
                    VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
                    mask=mask_t[:, None] & mask_k[None, :], other=0.0,
                )
                dS = tl.load(
                    dSCORES_ptr + pid_b*sdsb + offs_t[:, None]*sdst + offs_k[None, :]*sdsk,
                    mask=mask_t[:, None] & mask_k[None, :], other=0.0,
                )  # (BLOCK_Q, BLOCK_K)

                # K for head h at selected key positions: (BLOCK_Q, BLOCK_K, BLOCK_D)
                K = tl.load(
                    K_ptr + pid_b*skb + idx[:, :, None]*skt
                          + h*skh + offs_d[None, None, :]*skd,
                    mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                    other=0.0,
                )

                # weighted_dS: (BLOCK_Q, BLOCK_K) — gate by validity and W_h
                weighted_dS = dS * valid * W_h  # scalar W_h broadcasts

                # sum over slot dim: (BLOCK_Q, BLOCK_D)
                dQ_acc += tl.sum(weighted_dS[:, :, None] * K, axis=1)

            # Accumulate into dQ (layout B, T, H, D)
            dQ_cur = tl.load(
                dQ_ptr + pid_b*sqb + offs_t[:, None]*sqt + h*sqh + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            tl.store(
                dQ_ptr + pid_b*sqb + offs_t[:, None]*sqt + h*sqh + offs_d[None, :]*sqd,
                dQ_cur + dQ_acc,
                mask=mask_t[:, None] & mask_d[None, :],
            )


@triton.jit
def _indexer_bwd_dK(
    dSCORES_ptr, Q_ptr, K_ptr, W_ptr, IDX_ptr, VALID_ptr, dK_ptr,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    sib, sit, sik,
    svlb, svlt, svlk,
    sdsb, sdst, sdsk,
    B, T, H, D, TOP_K,
    BLOCK_Q:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_D:         tl.constexpr,
    BLOCK_INDEXER_H: tl.constexpr,
):
    # One program per (batch, query-tile).
    # Scatters into dK via atomic_add (multiple queries may share key tokens).
    # dK[idx[q,slot], h, d] += dS[q,slot] * W[h] * Q[q, h, d]
    pid_b  = tl.program_id(0)
    pid_t  = tl.program_id(1)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_t = offs_t < T

    for h in range(BLOCK_INDEXER_H):
        W_h = tl.load(W_ptr + h)  # scalar

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Q for head h: (BLOCK_Q, BLOCK_D)
            Q_h = tl.load(
                Q_ptr + pid_b*sqb + offs_t[:, None]*sqt + h*sqh + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )

            for slot_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
                offs_k = slot_block * BLOCK_K + tl.arange(0, BLOCK_K)
                mask_k = offs_k < TOP_K

                idx = tl.load(
                    IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
                    mask=mask_t[:, None] & mask_k[None, :], other=0,
                )  # (BLOCK_Q, BLOCK_K)
                valid = tl.load(
                    VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
                    mask=mask_t[:, None] & mask_k[None, :], other=0.0,
                )
                dS = tl.load(
                    dSCORES_ptr + pid_b*sdsb + offs_t[:, None]*sdst + offs_k[None, :]*sdsk,
                    mask=mask_t[:, None] & mask_k[None, :], other=0.0,
                )  # (BLOCK_Q, BLOCK_K)

                # weighted_dS: (BLOCK_Q, BLOCK_K)
                weighted_dS = dS * valid * W_h

                # contrib to dK: (BLOCK_Q, BLOCK_K, BLOCK_D)
                contrib = weighted_dS[:, :, None] * Q_h[:, None, :]

                tl.atomic_add(
                    dK_ptr + pid_b*skb + idx[:, :, None]*skt
                           + h*skh + offs_d[None, None, :]*skd,
                    contrib,
                    mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                )


@triton.jit
def _indexer_bwd_dW(
    dSCORES_ptr, Q_ptr, K_ptr, W_ptr, IDX_ptr, VALID_ptr, dW_ptr,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    sib, sit, sik,
    svlb, svlt, svlk,
    sdsb, sdst, sdsk,
    B, T, H, D, TOP_K,
    BLOCK_Q:         tl.constexpr,
    BLOCK_K:         tl.constexpr,
    BLOCK_D:         tl.constexpr,
    BLOCK_INDEXER_H: tl.constexpr,
):
    # One program per (batch, query-tile).
    # dW[h] += sum_{q,slot} dS[q,slot] * sum_d Q[q,h,d] * K[idx[q,slot],h,d]
    pid_b  = tl.program_id(0)
    pid_t  = tl.program_id(1)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_t = offs_t < T

    # Accumulate dW per head — loop over heads to stay 2D/3D max
    for h in range(BLOCK_INDEXER_H):
        dW_h_acc = tl.zeros((BLOCK_Q,), dtype=tl.float32)  # per-query accumulator

        for slot_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
            offs_k = slot_block * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < TOP_K

            idx = tl.load(
                IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
                mask=mask_t[:, None] & mask_k[None, :], other=0,
            )  # (BLOCK_Q, BLOCK_K)
            valid = tl.load(
                VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            )
            dS = tl.load(
                dSCORES_ptr + pid_b*sdsb + offs_t[:, None]*sdst + offs_k[None, :]*sdsk,
                mask=mask_t[:, None] & mask_k[None, :], other=0.0,
            )  # (BLOCK_Q, BLOCK_K)

            masked_dS = dS * valid  # (BLOCK_Q, BLOCK_K)

            # QdotK_h[q, slot] = sum_d Q[q,h,d] * K[idx[q,slot],h,d]
            QdotK_h = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

            for d_block in range(0, tl.cdiv(D, BLOCK_D)):
                offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D

                Q_h = tl.load(
                    Q_ptr + pid_b*sqb + offs_t[:, None]*sqt + h*sqh + offs_d[None, :]*sqd,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0,
                )  # (BLOCK_Q, BLOCK_D)

                K_h = tl.load(
                    K_ptr + pid_b*skb + idx[:, :, None]*skt
                          + h*skh + offs_d[None, None, :]*skd,
                    mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                    other=0.0,
                )  # (BLOCK_Q, BLOCK_K, BLOCK_D)

                QdotK_h += tl.sum(Q_h[:, None, :] * K_h, axis=-1)  # (BLOCK_Q, BLOCK_K)

            # dW[h] += sum_{q,slot} dS[q,slot] * QdotK_h[q,slot]
            # → reduce over slot first, then over q
            dW_h_acc += tl.sum(masked_dS * QdotK_h, axis=1)  # (BLOCK_Q,)

        # sum over q rows then atomic-add into dW[h]
        tl.atomic_add(dW_ptr + h, tl.sum(dW_h_acc))


# ---------------------------------------------------------------------------
# LightningIndexerFunction — manual autograd.Function
# ---------------------------------------------------------------------------

class LightningIndexerFunction(torch.autograd.Function):

    # FP8 is supported on Ampere (A100) and later for bandwidth savings.
    # H100 additionally has native FP8 tensor cores for extra compute speedup.
    _FP8_DTYPE     = torch.float8_e4m3fn
    _FP8_AVAILABLE = hasattr(torch, "float8_e4m3fn")

    @staticmethod
    def forward(
        ctx,
        q_proj, k_proj, W,
        B, T, H, D, TOP_K,
        BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H,
    ):
        device = q_proj.device

        idx_ptr    = torch.zeros((B, T, TOP_K), device=device, dtype=torch.int32)
        valid_ptr  = torch.zeros((B, T, TOP_K), device=device, dtype=torch.float32)
        scores_ptr = torch.zeros((B, T, TOP_K), device=device, dtype=torch.float32)

        # Cast to FP8 for the indexer kernel — halves Q/K bandwidth vs BF16.
        # We keep the original BF16 tensors for the backward (gradients need
        # full precision).  The FP8 copies are throwaway.
        use_fp8 = LightningIndexerFunction._FP8_AVAILABLE
        if use_fp8:
            q_fp8 = q_proj.detach().to(LightningIndexerFunction._FP8_DTYPE)
            k_fp8 = k_proj.detach().to(LightningIndexerFunction._FP8_DTYPE)
            sqb, sqt, sqh, sqd = q_fp8.stride()
            skb, skt, skh, skd = k_fp8.stride()
        else:
            q_fp8, k_fp8 = q_proj, k_proj
            sqb, sqt, sqh, sqd = q_proj.stride()
            skb, skt, skh, skd = k_proj.stride()

        sib, sit, sik    = idx_ptr.stride()
        svlb, svlt, svlk = valid_ptr.stride()
        ssb, sst, ssk    = scores_ptr.stride()

        grid   = (B, math.ceil(T / BLOCK_Q))
        kernel = _indexer_fwd_fp8 if use_fp8 else _indexer_fwd

        kernel[grid](
            q_fp8, k_fp8, idx_ptr, valid_ptr, scores_ptr, W,
            sqb, sqt, sqh, sqd,
            skb, skt, skh, skd,
            sib, sit, sik,
            svlb, svlt, svlk,
            ssb, sst, ssk,
            B, T, H, D,
            TOP_K, BLOCK_Q, BLOCK_K, BLOCK_INDEXER_H, BLOCK_D,
        )

        # Save original precision tensors for backward — NOT the FP8 copies
        ctx.save_for_backward(q_proj, k_proj, W, idx_ptr, valid_ptr)
        ctx.dims        = (B, T, H, D, TOP_K)
        ctx.blocks      = (BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H)
        ctx.strides_q   = q_proj.stride()
        ctx.strides_k   = k_proj.stride()
        ctx.strides_idx = (sib, sit, sik)
        ctx.strides_vld = (svlb, svlt, svlk)

        return idx_ptr, valid_ptr, scores_ptr

    @staticmethod
    def backward(ctx, dIDX, dVALID, dSCORES):
        # dIDX and dVALID are zero — indices and binary validity have no gradient.
        # dSCORES arrives from sigmoid(scores).backward() via routing_weights.

        q_proj, k_proj, W, idx_ptr, valid_ptr = ctx.saved_tensors
        B, T, H, D, TOP_K      = ctx.dims
        BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H = ctx.blocks
        sqb, sqt, sqh, sqd     = ctx.strides_q
        skb, skt, skh, skd     = ctx.strides_k
        sib, sit, sik          = ctx.strides_idx
        svlb, svlt, svlk       = ctx.strides_vld

        dSCORES = dSCORES.contiguous()
        sdsb, sdst, sdsk = dSCORES.stride()

        dQ = torch.zeros_like(q_proj)
        dK = torch.zeros_like(k_proj)
        dW = torch.zeros_like(W)

        grid = (B, math.ceil(T / BLOCK_Q))

        _indexer_bwd_dQ[grid](
            dSCORES, q_proj, k_proj, W, idx_ptr, valid_ptr, dQ,
            sqb, sqt, sqh, sqd,
            skb, skt, skh, skd,
            sib, sit, sik,
            svlb, svlt, svlk,
            sdsb, sdst, sdsk,
            B, T, H, D, TOP_K,
            BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H,
        )

        _indexer_bwd_dK[grid](
            dSCORES, q_proj, k_proj, W, idx_ptr, valid_ptr, dK,
            sqb, sqt, sqh, sqd,
            skb, skt, skh, skd,
            sib, sit, sik,
            svlb, svlt, svlk,
            sdsb, sdst, sdsk,
            B, T, H, D, TOP_K,
            BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H,
        )

        _indexer_bwd_dW[grid](
            dSCORES, q_proj, k_proj, W, idx_ptr, valid_ptr, dW,
            sqb, sqt, sqh, sqd,
            skb, skt, skh, skd,
            sib, sit, sik,
            svlb, svlt, svlk,
            sdsb, sdst, sdsk,
            B, T, H, D, TOP_K,
            BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H,
        )

        # Return grads aligned with forward args:
        # q_proj, k_proj, W,  B, T, H, D, TOP_K,  BLOCK_Q, BLOCK_K, BLOCK_D, BLOCK_INDEXER_H
        return dQ, dK, dW, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Attention forward kernel — MQA version
# Q: (B, H, T, D) — per-head queries
# K: (B, T, D)    — shared across all heads (MQA)
# V: (B, T, D)    — shared across all heads (MQA)
# K/V loads drop the pid_h * skh/svh offset — same K/V for every head program.
# ---------------------------------------------------------------------------

@triton.jit
def _dsa_attention_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, IDX_ptr, VALID_ptr, RW_ptr,
    sqb, sqh, sqt, sqd,
    skb, skt, skd,           # K has no head stride — (B, T, D)
    svb, svt, svd,           # V has no head stride — (B, T, D)
    sob, soh, sot, sod,
    sib, sit, sik,
    svlb, svlt, svlk,
    srb, srt, srk,
    B, H, T, D,
    TOP_K:             tl.constexpr,
    BLOCK_Q:           tl.constexpr,
    BLOCK_K:           tl.constexpr,
    BLOCK_ATTENTION_H: tl.constexpr,
    BLOCK_D:           tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_t  = tl.program_id(2)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_t = offs_t < T
    scale  = 1.0 / tl.sqrt(D.to(tl.float32))

    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)

    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        tl.store(
            O_ptr + pid_b*sob + pid_h*soh
                  + offs_t[:, None]*sot + offs_d[None, :]*sod,
            tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32),
            mask=mask_t[:, None] & mask_d[None, :],
        )

    for k_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOP_K

        idx = tl.load(
            IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
            mask=mask_t[:, None] & mask_k[None, :], other=0,
        )
        valid = tl.load(
            VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )
        rw = tl.load(
            RW_ptr + pid_b*srb + offs_t[:, None]*srt + offs_k[None, :]*srk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )

        QK = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            # Q is per-head — include pid_h * sqh
            Q = tl.load(
                Q_ptr + pid_b*sqb + pid_h*sqh
                      + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            # K is shared — NO pid_h offset
            K = tl.load(
                K_ptr + pid_b*skb
                      + idx[:, :, None]*skt + offs_d[None, None, :]*skd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            QK += tl.sum(Q[:, None, :] * K, axis=-1)

        QK         = tl.where(valid > 0.0, QK * scale, float("-inf"))
        block_max  = tl.max(QK, axis=1)
        m_new      = tl.maximum(m, block_max)
        exp_logits = tl.where(valid > 0.0, tl.exp(QK - m_new[:, None]) * rw, 0.0)
        alpha      = tl.exp(m - m_new)
        l          = l * alpha + tl.sum(exp_logits, axis=1)
        m          = m_new

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            # V is shared — NO pid_h offset
            V = tl.load(
                V_ptr + pid_b*svb
                      + idx[:, :, None]*svt + offs_d[None, None, :]*svd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            O_old = tl.load(
                O_ptr + pid_b*sob + pid_h*soh
                      + offs_t[:, None]*sot + offs_d[None, :]*sod,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            tl.store(
                O_ptr + pid_b*sob + pid_h*soh
                      + offs_t[:, None]*sot + offs_d[None, :]*sod,
                O_old * alpha[:, None] + tl.sum(exp_logits[:, :, None] * V, axis=1),
                mask=mask_t[:, None] & mask_d[None, :],
            )

    l_safe = tl.maximum(l, 1e-6)
    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        O = tl.load(
            O_ptr + pid_b*sob + pid_h*soh
                  + offs_t[:, None]*sot + offs_d[None, :]*sod,
            mask=mask_t[:, None] & mask_d[None, :], other=0.0,
        )
        tl.store(
            O_ptr + pid_b*sob + pid_h*soh
                  + offs_t[:, None]*sot + offs_d[None, :]*sod,
            O / l_safe[:, None],
            mask=mask_t[:, None] & mask_d[None, :],
        )


# ---------------------------------------------------------------------------
# Attention backward kernel — MQA version
# K, V, dK, dV have no head dimension.
# dK and dV use atomic_add — all H head programs scatter into the same tensor.
# ---------------------------------------------------------------------------

@triton.jit
def _dsa_attention_bwd(
    dO_ptr, Q_ptr, K_ptr, V_ptr, O_ptr, IDX_ptr, VALID_ptr, RW_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dRW_ptr,
    sqb, sqh, sqt, sqd,
    skb, skt, skd,           # K: (B, T, D) — no head dim
    svb, svt, svd,           # V: (B, T, D) — no head dim
    sob, soh, sot, sod,
    sib, sit, sik,
    svlb, svlt, svlk,
    srb, srt, srk,
    B, H, T, D,
    TOP_K:             tl.constexpr,
    BLOCK_Q:           tl.constexpr,
    BLOCK_K:           tl.constexpr,
    BLOCK_ATTENTION_H: tl.constexpr,
    BLOCK_D:           tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_t  = tl.program_id(2)
    offs_t = pid_t * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_t = offs_t < T
    scale  = 1.0 / tl.sqrt(D.to(tl.float32))

    # Pass 1: recompute m and l
    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)

    for k_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOP_K
        idx = tl.load(
            IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
            mask=mask_t[:, None] & mask_k[None, :], other=0,
        )
        valid = tl.load(
            VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )
        rw = tl.load(
            RW_ptr + pid_b*srb + offs_t[:, None]*srt + offs_k[None, :]*srk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )
        QK = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            Q = tl.load(
                Q_ptr + pid_b*sqb + pid_h*sqh
                      + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            K = tl.load(
                K_ptr + pid_b*skb
                      + idx[:, :, None]*skt + offs_d[None, None, :]*skd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            QK += tl.sum(Q[:, None, :] * K, axis=-1)

        QK    = tl.where(valid > 0.0, QK * scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(QK, axis=1))
        alpha = tl.exp(m - m_new)
        l     = l * alpha + tl.sum(
            tl.where(valid > 0.0, tl.exp(QK - m_new[:, None]) * rw, 0.0), axis=1
        )
        m = m_new

    l_safe = tl.maximum(l, 1e-6)

    # Pass 2: Di = rowsum(dO * O)
    Di = tl.zeros((BLOCK_Q,), tl.float32)
    for d_block in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        O = tl.load(
            O_ptr + pid_b*sob + pid_h*soh
                  + offs_t[:, None]*sot + offs_d[None, :]*sod,
            mask=mask_t[:, None] & mask_d[None, :], other=0.0,
        )
        dO = tl.load(
            dO_ptr + pid_b*sob + pid_h*soh
                   + offs_t[:, None]*sot + offs_d[None, :]*sod,
            mask=mask_t[:, None] & mask_d[None, :], other=0.0,
        )
        Di += tl.sum(dO * O, axis=1)

    # Pass 3: dQ, dK, dV, dRW
    for k_block in range(0, tl.cdiv(TOP_K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOP_K

        idx = tl.load(
            IDX_ptr + pid_b*sib + offs_t[:, None]*sit + offs_k[None, :]*sik,
            mask=mask_t[:, None] & mask_k[None, :], other=0,
        )
        valid = tl.load(
            VALID_ptr + pid_b*svlb + offs_t[:, None]*svlt + offs_k[None, :]*svlk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )
        rw = tl.load(
            RW_ptr + pid_b*srb + offs_t[:, None]*srt + offs_k[None, :]*srk,
            mask=mask_t[:, None] & mask_k[None, :], other=0.0,
        )
        valid_bool = valid > 0.0

        QK = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            Q = tl.load(
                Q_ptr + pid_b*sqb + pid_h*sqh
                      + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            K = tl.load(
                K_ptr + pid_b*skb
                      + idx[:, :, None]*skt + offs_d[None, None, :]*skd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            QK += tl.sum(Q[:, None, :] * K, axis=-1)

        QK      = tl.where(valid_bool, QK * scale, float("-inf"))
        softmax = tl.where(valid_bool, tl.exp(QK - m[:, None]) / l_safe[:, None], 0.0)
        P       = softmax * rw

        dP = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            dO = tl.load(
                dO_ptr + pid_b*sob + pid_h*soh
                       + offs_t[:, None]*sot + offs_d[None, :]*sod,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            # V is shared — no pid_h offset
            V = tl.load(
                V_ptr + pid_b*svb
                      + idx[:, :, None]*svt + offs_d[None, None, :]*svd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            dP += tl.sum(dO[:, None, :] * V, axis=-1)
            # dV: scatter into shared (B, T, D) tensor — atomic because H programs race
            tl.atomic_add(
                dV_ptr + pid_b*svb
                       + idx[:, :, None]*svt + offs_d[None, None, :]*svd,
                (P * valid.to(tl.float32))[:, :, None] * dO[:, None, :],
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
            )

        # dRW — indexed into (B, T, TOP_K), shared across heads, atomic
        dRW_block = tl.where(valid_bool, softmax * (dP - Di[:, None]), 0.0)
        tl.atomic_add(
            dRW_ptr + pid_b*srb + offs_t[:, None]*srt + offs_k[None, :]*srk,
            dRW_block,
            mask=mask_t[:, None] & mask_k[None, :],
        )

        dScore = tl.where(valid_bool, rw * softmax * (dP - Di[:, None]) * scale, 0.0)

        for d_block in range(0, tl.cdiv(D, BLOCK_D)):
            offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            Q = tl.load(
                Q_ptr + pid_b*sqb + pid_h*sqh
                      + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            K = tl.load(
                K_ptr + pid_b*skb
                      + idx[:, :, None]*skt + offs_d[None, None, :]*skd,
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
                other=0.0,
            )
            # dQ is per-head — no race, direct store
            dQ_tile = tl.load(
                dQ_ptr + pid_b*sqb + pid_h*sqh
                       + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                mask=mask_t[:, None] & mask_d[None, :], other=0.0,
            )
            dQ_tile += tl.sum(dScore[:, :, None] * K, axis=1)
            tl.store(
                dQ_ptr + pid_b*sqb + pid_h*sqh
                       + offs_t[:, None]*sqt + offs_d[None, :]*sqd,
                dQ_tile,
                mask=mask_t[:, None] & mask_d[None, :],
            )
            # dK: scatter into shared (B, T, D) tensor — atomic because H programs race
            tl.atomic_add(
                dK_ptr + pid_b*skb
                       + idx[:, :, None]*skt + offs_d[None, None, :]*skd,
                dScore[:, :, None] * Q[:, None, :],
                mask=mask_t[:, None, None] & mask_k[None, :, None] & mask_d[None, None, :],
            )


# ---------------------------------------------------------------------------
# DSAAttentionFunction — manual autograd.Function
# ---------------------------------------------------------------------------

class DSAAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q, k, v, idx, valid, routing_weights, o,
        B, H, T, D, TOP_K,
        BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D,
    ):
        sqb, sqh, sqt, sqd  = q.stride()
        skb, skt, skd  = k.stride()
        svb, svt, svd  = v.stride()
        sob, soh, sot, sod  = o.stride()
        sib, sit, sik       = idx.stride()
        svlb, svlt, svlk    = valid.stride()
        srb, srt, srk       = routing_weights.stride()

        grid = (B, H, math.ceil(T / BLOCK_Q))

        _dsa_attention_fwd[grid](
            q, k, v, o, idx, valid, routing_weights,
            sqb, sqh, sqt, sqd,
            skb, skt, skd,
            svb, svt, svd,
            sob, soh, sot, sod,
            sib, sit, sik,
            svlb, svlt, svlk,
            srb, srt, srk,
            B, H, T, D,
            TOP_K, BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D,
        )

        ctx.save_for_backward(q, k, v, idx, valid, routing_weights, o)
        ctx.strides = (
            sqb, sqh, sqt, sqd,
            skb, skt, skd,
            svb, svt, svd,
            sob, soh, sot, sod,
            sib, sit, sik,
            svlb, svlt, svlk,
            srb, srt, srk,
        )
        ctx.dims = (B, H, T, D, TOP_K, BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D)
        return o

    @staticmethod
    def backward(ctx, dO):
        q, k, v, idx, valid, routing_weights, o = ctx.saved_tensors
        (sqb, sqh, sqt, sqd,
         skb, skt, skd,
         svb, svt, svd,
         sob, soh, sot, sod,
         sib, sit, sik,
         svlb, svlt, svlk,
         srb, srt, srk) = ctx.strides
        B, H, T, D, TOP_K, BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D = ctx.dims

        dQ  = torch.zeros_like(q)
        dK  = torch.zeros_like(k)
        dV  = torch.zeros_like(v)
        dRW = torch.zeros_like(routing_weights)

        grid = (B, H, math.ceil(T / BLOCK_Q))

        _dsa_attention_bwd[grid](
            dO.contiguous(), q, k, v, o, idx, valid, routing_weights,
            dQ, dK, dV, dRW,
            sqb, sqh, sqt, sqd,
            skb, skt, skd,
            svb, svt, svd,
            sob, soh, sot, sod,
            sib, sit, sik,
            svlb, svlt, svlk,
            srb, srt, srk,
            B, H, T, D,
            TOP_K, BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D,
        )

        # forward args: q, k, v, idx, valid, routing_weights, o,
        #               B, H, T, D, TOP_K,
        #               BLOCK_Q, BLOCK_K, BLOCK_ATTENTION_H, BLOCK_D
        # non-tensor scalars: 9
        return (dQ, dK, dV,
                None, None,  # idx, valid
                dRW,         # routing_weights — flows back to sigmoid → scores → indexer
                None,        # o
                *([None] * 9))
    