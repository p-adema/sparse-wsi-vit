import cuda.bindings.driver as cuda  # ty:ignore[unresolved-import]
import cutlass
import torch
from cutlass import cute
from torch.nn.attention.flex_attention import BlockMask, noop_mask


@cute.kernel
def _nonzero_kernel(
    dense,  # [B, H, L, L]
    idx_col,  # [B, H, L, L]
    len_col,  # [B, H, L]
):
    num_b_h_c = (
        cute.size(dense, mode=[0])
        * cute.size(dense, mode=[1])
        * cute.size(dense, mode=[2])
    )
    tpg = cute.arch.grid_dim()[0] * cute.arch.block_dim()[0]
    tid = (
        cute.arch.thread_idx()[0] + cute.arch.block_dim()[0] * cute.arch.block_idx()[0]
    )
    for b_h in range(tid, num_b_h_c, tpg):
        head_batch = b_h // cute.size(dense, mode=[2])
        col = b_h % cute.size(dense, mode=[2])
        head = head_batch % cute.size(dense, mode=[1])
        batch = head_batch // cute.size(dense, mode=[1])

        seen = cutlass.Int32(0)
        for row in range(cute.size(dense, mode=[3])):
            if dense[batch, head, col, row]:
                idx_col[batch, head, col, seen] = row
                seen += 1

        len_col[batch, head, col] = seen


@cute.jit
def _nonzero_call(
    dense,  # [B, H, L, L]
    idx_col,  # [B, H, L, L]
    len_col,  # [B, H, L]
    stream,
    sm_count: cutlass.Constexpr = cutlass.utils.HardwareInfo().get_device_multiprocessor_count(),  # ty:ignore[invalid-parameter-default]
):
    _nonzero_kernel(dense, idx_col, len_col).launch(
        grid=[
            cutlass.Int32(sm_count),  # ty:ignore[invalid-argument-type]
            1,
            1,
        ],  # number of blocks
        block=[32, 1, 1],  # number of threads
        stream=stream,
    )


def _nonzero_compile(dense_leading: int):
    dense = torch.randint(0, 1, (1, 4, 715, 715), dtype=torch.int8, device="cuda")
    if dense_leading != 3:
        dense.transpose_(dense_leading, -1)
    idx_col = dense.new_empty(dense.shape, dtype=torch.int32)
    len_col = dense.new_empty(dense.shape[:-1], dtype=torch.int32)
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    return cute.compile(
        _nonzero_call,
        cute.runtime.from_dlpack(dense, assumed_align=4)
        .mark_layout_dynamic(dense_leading)
        .mark_compact_shape_dynamic(dense_leading, dense.dim_order()),
        cute.runtime.from_dlpack(idx_col, assumed_align=4)
        .mark_layout_dynamic(3)
        .mark_compact_shape_dynamic(3, idx_col.dim_order()),
        cute.runtime.from_dlpack(len_col, assumed_align=4)
        .mark_layout_dynamic(2)
        .mark_compact_shape_dynamic(2, len_col.dim_order()),
        stream,
    )


_nonzero_comp = {
    2: _nonzero_compile(2),
    3: _nonzero_compile(3),
}


@torch.library.custom_op(
    "wsivit::nonzero_column",
    mutates_args={},
    device_types="cuda",
)
def nonzero_column(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx_col = dense.new_empty(dense.shape, dtype=torch.int32)
    len_col = dense.new_empty(dense.shape[:-1], dtype=torch.int32)
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    dense_leading = 3 if dense.stride(3) == 1 else 2

    _nonzero_comp[dense_leading](
        cute.runtime.from_dlpack(dense, assumed_align=4)
        .mark_layout_dynamic(dense_leading)
        .mark_compact_shape_dynamic(dense_leading, dense.dim_order()),
        cute.runtime.from_dlpack(idx_col, assumed_align=4)
        .mark_layout_dynamic(3)
        .mark_compact_shape_dynamic(3, idx_col.dim_order()),
        cute.runtime.from_dlpack(len_col, assumed_align=4)
        .mark_layout_dynamic(2)
        .mark_compact_shape_dynamic(2, len_col.dim_order()),
        stream,
    )
    return idx_col, len_col


@nonzero_column.register_fake
def _(dense: torch.Tensor):
    idx_col = dense.new_empty(dense.shape, dtype=torch.int32)
    len_col = dense.new_empty(dense.shape[:-1], dtype=torch.int32)
    return idx_col, len_col


def topk_to_block_mask(
    idx_cmp: torch.Tensor,
    self: bool = True,
    local: int = 4,
    cls: bool = True,
    block_size: int = 128,
):
    """
    Takes a Top-K asssignment from Flash-Top-K (shaped [B, N_BLK, H, K])
    and produces a flex-attention compatible BlockMask.

    self: Add self-attention
    local: Add sliding window attention
    cls: Add global attention on the first block
    """
    block_indices = idx_cmp.transpose(1, 2)
    B, H, N_BLK, K = block_indices.shape

    full_dense = block_indices.new_zeros(B, H, N_BLK, N_BLK, dtype=torch.int8)
    full_dense.scatter_(3, block_indices, 1)
    if self:
        torch.diagonal(full_dense, dim1=2, dim2=3).fill_(1)
    for offset in range(1, local + 1):
        torch.diagonal(full_dense, offset, dim1=2, dim2=3).fill_(1)
        torch.diagonal(full_dense, -offset, dim1=2, dim2=3).fill_(1)
    if cls:
        full_dense[..., :1, :] = 1
        full_dense[..., :, :1] = 1

    full_kv_indices, full_kv_num_blocks = nonzero_column(full_dense)
    full_dense = full_dense.transpose_(-1, -2).contiguous()
    full_q_indices, full_q_num_blocks = nonzero_column(full_dense)

    # Trivial empty, all blocks are full blocks
    kv_num_blocks = q_num_blocks = block_indices.new_zeros(B, H, N_BLK)
    kv_indices = q_indices = full_q_indices

    seq_lengths = (N_BLK * block_size, N_BLK * block_size)
    BLOCK_SIZE = (block_size, block_size)
    mask_mod = noop_mask

    return BlockMask(
        seq_lengths,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        BLOCK_SIZE,
        mask_mod,
    )
