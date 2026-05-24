# instead of compression.py we just select the top k blocks here
import torch

def get_wsi_block_indices(q_cmp: torch.Tensor, k_cmp: torch.Tensor, block_counts: int) -> torch.Tensor:
    """
    Pure PyTorch Router for WSI Top-K Block Selection.
    Replaces the legacy Triton compression kernels.
    """
    B, Num_Blocks, HQ, D = q_cmp.shape
    HKV = k_cmp.shape[2]

    # Broadcast Keys if using GQA
    if HQ != HKV:
        k_cmp = k_cmp.repeat_interleave(HQ // HKV, dim=2)

    # Transpose for matmul: (B, HQ, Num_Blocks, D)
    q_cmp_t = q_cmp.transpose(1, 2)
    k_cmp_t = k_cmp.transpose(1, 2)

    # 1. Compute attention scores between all macro-patches
    scores = torch.matmul(q_cmp_t, k_cmp_t.transpose(-1, -2)) / (D ** 0.5)

    # 2. Diagonal Mask (Self-Exclusion)
    # Prevents the router from selecting the local block, as that is handled by the Local Branch
    diagonal_mask = torch.eye(Num_Blocks, device=scores.device, dtype=torch.bool)
    diagonal_mask = diagonal_mask.view(1, 1, Num_Blocks, Num_Blocks)
    scores = scores.masked_fill(diagonal_mask, float('-inf'))

    # 3. Return pure Distant Top-K indices
    # Shape: (B, HQ, Num_Blocks, block_counts)
    return scores.topk(block_counts, dim=-1).indices.contiguous()