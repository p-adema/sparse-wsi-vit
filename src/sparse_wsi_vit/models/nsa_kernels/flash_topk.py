# Requires: nvidia-cutlass-dsl
import functools
import math
import time
from collections.abc import Callable
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch.library
from cutlass import Tensor
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from flash_attn.flash_attn_interface import _flash_attn_backward

"""
A flash attention v2 forward pass example for NVIDIA Ampere SM80 architecture using CUTE DSL.
** modified to also return the indices and raw scores of the top-K highest scoring keys **

- Matrix Q is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension
- Matrix K is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix V is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix O is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension

This kernel supports the following features:
    - Utilizes CpAsync for efficient memory operations
    - Utilizes Ampere's tensor core for matrix multiply-accumulate (MMA) operations
    - Utilizes register pipeline to overlap shared memory-to-register transfers with computations.
    - Leverages DSL to implement an integrated online softmax fusion pattern.

This kernel works as follows:
1. Load Q and K matrices from global memory (GMEM) to shared memory (SMEM) using CpAsync operations.
2. Perform matrix multiply-accumulate (MMA) operations using tensor core instructions to compute intermediate result S.
3. Apply padding mask or causal mask to S during initial iterations.
4. Apply online softmax to S and rescale O using results from previous iteration.
5. Load V matrices and perform matrix multiply-accumulate (MMA) operations to compute final result O.
6. Normalize O after all iterations complete and store result back to global memory (GMEM).

Now, it also does:
1. Use bitonic sorting to sort the

To run this example:

.. code-block:: bash

    python examples/ampere/flash_attention_v2.py                                            \
      --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                  \
      --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                      \
      --num_head 16 --softmax_scale 1.0 --is_causal

The above command configures the model to use float16 for inputs and outputs. The problem dimensions
are set to: batch size of 1, query sequence length of 1280, key sequence length of 1536, head dimension
of 128, and 16 attention heads. The softmax scale is set to 1.0 and causal masking is enabled. The computation
uses tiles of size 128x128 for m and n dimensions, and utilizes 128 parallel threads.

To collect the performance with NCU profiler:

.. code-block:: bash

    ncu python examples/ampere/flash_attention_v2.py                                        \
        --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                \
        --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                    \
        --num_head 16 --softmax_scale 1.0 --is_causal --skip_ref_check

There are some constraints for this example:
* Only fp16 and bf16 data types are supported.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The log-sum-exp(for training) is not computed in the kernel.
* The values of `m_block_size`, `n_block_size`, and `head_dim` must be selected to stay within shared memory capacity limits.
* `m_block_size * 2` must be divisible by `num_threads`, otherwise the kernel will not be able to get the correct result.
"""


class FlashAttnTopKKernel:
    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
        top_k: int = 8,
        log_smem: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        assert n_block_size // 4 >= top_k, "It's a pain otherwise"
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._top_k = top_k
        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._log_smem = log_smem

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        # Check if data type is fp16 or bf16
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False

        # Check if head dimension is a multiple of 8
        if head_dim % 8 != 0:
            return False

        # Check if number of threads is a multiple of 32
        if num_threads % 32 != 0:
            return False

        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mT: cute.Tensor,
        mA: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention v2 kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, 1)

        Prepares the shared memory layout, tiled copy atoms, tiled mma and shared memory storage.
        Then launches the kernel function with the prepared parameters.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale: softmax scale
        :type softmax_scale: cutlass.Float32
        """
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout
        sVal_layout = cute.make_layout((self._m_block_size, self._top_k, 4))

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sVal: cute.struct.Align[
                cute.struct.MemRange[cute.Float32, cute.cosize(sVal_layout)], 1024
            ]
            sIdx: cute.struct.Align[
                cute.struct.MemRange[cute.Int32, cute.cosize(sVal_layout)], 1024
            ]

        if self._log_smem:
            print("Debug info about FlashTopK SharedStorage:")
            print("    sQ", cute.cosize(sQ_layout) * 2 / 1024, "KiB")
            print("    sK", cute.cosize(sKV_layout) * 2 / 1024, "KiB")
            print("    sV", cute.cosize(sQ_layout) * 2 / 1024, "KiB")
            print("    sVal", cute.cosize(sVal_layout) * 4 / 1024, "KiB")
            print("    sIdx", cute.cosize(sVal_layout) * 4 / 1024, "KiB")
            numel = (
                cute.cosize(sQ_layout)
                + cute.cosize(sQ_layout) * 2
                + cute.cosize(sVal_layout) * 4
            )
            print("    =", numel * 2 / 1024, "KiB")

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid_dim: (m_block, batch_size, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mT,
            mA,
            mLSE,
            softmax_scale,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mT: cute.Tensor,
        mA: cute.Tensor,
        mLSE,
        softmax_scale: cutlass.Constexpr,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Kernel function for flash attention v2.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale_log2: softmax scale log2
        :type softmax_scale_log2: cutlass.Float32
        :param sQ_layout: query layout
        :type sQ_layout: cute.ComposedLayout
        :param sKV_layout: key/value layout
        :type sKV_layout: cute.ComposedLayout
        :param sO_layout: output layout
        :type sO_layout: cute.ComposedLayout
        :param gmem_tiled_copy_QKV: tiled copy for QKV load
        :type gmem_tiled_copy_QKV: cute.TiledCopy
        :param gmem_tiled_copy_O: tiled copy for O store
        :type gmem_tiled_copy_O: cute.TiledCopy
        :param tiled_mma: tiled mma
        :type tiled_mma: cute.TiledMma
        :param SharedStorage: shared storage
        :type SharedStorage: cutlass.Constexpr
        """
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_idx, head_idx = cute.arch.block_idx()
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div(
                    (m_block + 1) * self._m_block_size,
                    self._n_block_size,
                ),
                n_block_max,
            )
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        # (m_block_size, head_dim)
        gQ = cute.local_tile(
            mQ[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(
            mV[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()

        smem_size = max(
            SharedStorage.__sizeof__(),
            cute.size_in_bytes(
                cutlass.Float32, cute.make_layout((self._m_block_size, self._top_k, 2))
            ),
        )
        smem_ptr = smem.allocate(smem_size, byte_alignment=SharedStorage.__alignof__())
        storage = SharedStorage(smem_ptr)

        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transpose view of V to tensor with layout (head_dim, n_block_size) for tiled mma
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for Q and KV
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )

        # Repeat the partitioning with identity layouts
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and do special process for mn.
        # This is to reduce register pressure and gets 2-3% performance gain compared with allocating the whole tile.
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        # Set predicates for head_dim bounds, seqlen_q/k bounds is processed at the first tile.
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block],
                    tKsK[None, n, None],
                    pred=tKVpKV[None, n, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # Softmax intermediate result: row_max and row_sum
        # ///////////////////////////////////////////////////////////////////////////////
        # shape: (atom_v_m * rest_m)
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        # shape: (atom_v_m * rest_m)
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        topk_shape = (self._top_k, (2, acc_O.shape[1]))
        rTopVal = cute.make_rmem_tensor(topk_shape, cutlass.Float32)
        rTopIdx = cute.make_rmem_tensor(topk_shape, cutlass.Int32)

        # group parameters for compute_one_n_block
        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            batch_size=batch_idx,
            num_head=head_idx,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )
        topk_params = SimpleNamespace(
            rV=rTopVal,
            rI=rTopIdx,
        )

        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last ceil_div(m_block_size, n_block_size) blocks.
        # We will have at least 1 "masking" iteration.
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        topk_params,
                        is_first_n_block=(n_tile == 0),
                        in_mask_steps=True,
                    )
            else:
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    topk_params,
                    is_first_n_block=True,
                    in_mask_steps=True,
                )

        # Start async loads of rest k-tiles in reverse order, no k-residue handling needed
        for n_tile in range(mask_steps, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                topk_params,
                is_first_n_block=False,
                in_mask_steps=False,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # normalize acc_O by row_sum and calculate the lse
        self.normalize_softmax(acc_O, row_sum)
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        # smem copy atom for O
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        # tiled copy atom for O
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        gO = cute.local_tile(
            mO[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        # sync before all smem stores are done.
        self.cta_sync_barrier.arrive_and_wait()
        # load acc O from smem to rmem for wider vectorization
        cute.copy(
            gmem_tiled_copy_O,
            tOsO,
            tOrO,
        )
        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

        sVal_layout = cute.make_layout((self._m_block_size, self._top_k, 4))

        sVal = storage.sVal.get_tensor(sVal_layout)
        sIdx = storage.sIdx.get_tensor(sVal_layout)

        for m in cutlass.range_constexpr(acc_O.shape[1]):
            for m_atom in cutlass.range_constexpr(2):
                m_pos = (tidx % 32) // 4 + m_atom * 8 + tidx // 32 * 16 + m * 64
                cute.autovec_copy(
                    topk_params.rV[None, (m_atom, m)], sVal[m_pos, None, tidx % 4]
                )
                cute.autovec_copy(
                    topk_params.rI[None, (m_atom, m)], sIdx[m_pos, None, tidx % 4]
                )
        cute.arch.sync_threads()
        for m in range(tidx, self._m_block_size, self._num_threads):
            # set index 1 and 3 to ascending
            for i in cutlass.range_constexpr(self._top_k // 2):
                i_flip = self._top_k - i - 1
                flip1, flip3 = sVal[m, i, 1], sVal[m, i, 3]
                sVal[m, i, 1] = sVal[m, i_flip, 1]
                sVal[m, i, 3] = sVal[m, i_flip, 3]
                sVal[m, i_flip, 1] = flip1
                sVal[m, i_flip, 3] = flip3
                flip1, flip3 = sIdx[m, i, 1], sIdx[m, i, 3]
                sIdx[m, i, 1] = sIdx[m, i_flip, 1]
                sIdx[m, i, 3] = sIdx[m, i_flip, 3]
                sIdx[m, i_flip, 1] = flip1
                sIdx[m, i_flip, 3] = flip3

            # 0 & 1 merged descending
            self._bitonic_merge_in_left(
                sVal[m, None, 0],
                sIdx[m, None, 0],
                sVal[m, None, 1],
                sIdx[m, None, 1],
                left_direction=0,
            )
            # 2 & 3 merged *ascending*
            self._bitonic_merge_in_left(
                sVal[m, None, 2],
                sIdx[m, None, 2],
                sVal[m, None, 3],
                sIdx[m, None, 3],
                left_direction=1,
            )
            # 0 contains answer!
            self._bitonic_merge_in_left(
                sVal[m, None, 0],
                sIdx[m, None, 0],
                sVal[m, None, 2],
                sIdx[m, None, 2],
                left_direction=0,
            )
            gT = cute.local_tile(
                mT[batch_idx, m_block, head_idx, m, None],
                (self._top_k,),
                (0,),
            )
            gA = cute.local_tile(
                mA[batch_idx, m_block, head_idx, m, None],
                (self._top_k,),
                (0,),
            )
            cute.autovec_copy(sIdx[m, None, 0], gT)
            cute.autovec_copy(sVal[m, None, 0], gA)

        gLSE = cute.local_tile(
            mLSE[batch_idx, head_idx, None], (self._m_block_size,), (m_block,)
        )
        if tidx % 4 == 0:
            # too lazy to get all threads to do this copy, it's only 4 values anyway
            for m in cutlass.range_constexpr(acc_O.shape[1]):
                for m_atom in cutlass.range_constexpr(acc_O.shape[0][0]):
                    m_pos = (tidx % 32) // 4 + m_atom * 8 + tidx // 32 * 16 + m * 64
                    gLSE[m_pos] = (
                        cute.math.log(softmax_params.row_sum[m_atom + m * 2])
                        + softmax_params.row_max[m_atom + m * 2] * softmax_scale
                    )

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        topk_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param gmem_copy_params: gmem copy parameters
        :type gmem_copy_params: SimpleNamespace
        :param smem_copy_params: smem copy parameters
        :type smem_copy_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param topk_params: top-k parameters
        :type topk_params: SimpleNamespace
        :param is_first_n_block: is first n block
        :type is_first_n_block: cutlass.Constexpr
        """
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # wait for smem tile QK before mma calculation for S
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # load smem tile V for O, special process for the first tile to avoid loading nan.
        # The `if` here is a constexpr, won't be generated in the IR.
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.mK.layout.shape[1],
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # S gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # load first QK k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        # mma for S
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            # load next QK k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # wait for smem tile V for O
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > 0:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////////
        # top-k calculation
        # ///////////////////////////////////////////////////////////////////////////////
        for m in cutlass.range_constexpr(cute.size(acc_S, mode=[1])):
            for m_atom in cutlass.range_constexpr(2):
                slice_S = acc_S[(None, m_atom), m, None]
                rVals = cute.make_rmem_tensor(cute.size(slice_S), cutlass.Float32)
                rVals.store(slice_S.load())

                rIdx = cute.make_rmem_tensor(cute.size(slice_S), cutlass.Int32)
                for n in cutlass.range_constexpr(cute.size(acc_S, mode=[2])):
                    block_offset = basic_params.n_block * self._n_block_size
                    thread_offset = (cute.arch.thread_idx()[0] % 4) * 2
                    rIdx[2 * n] = block_offset + thread_offset + n * 8
                    rIdx[2 * n + 1] = block_offset + thread_offset + n * 8 + 1

                if is_first_n_block:
                    # Sort descending and store
                    self._bitonic_sort(rVals, rIdx)
                    topV = cute.local_tile(rVals, (self._top_k,), (0,)).load()
                    topI = cute.local_tile(rIdx, (self._top_k,), (0,)).load()
                    topk_params.rV[None, (m_atom, m)].store(topV)
                    topk_params.rI[None, (m_atom, m)].store(topI)
                else:
                    # Sort *ascending* and merge in
                    self._bitonic_sort(rVals, rIdx, direction=1)
                    topV = cute.local_tile(
                        rVals, (self._top_k,), (cute.size(rVals) // self._top_k - 1,)
                    )
                    topI = cute.local_tile(
                        rIdx, (self._top_k,), (cute.size(rVals) // self._top_k - 1,)
                    )
                    self._bitonic_merge_in_left(
                        topk_params.rV[None, (m_atom, m)],
                        topk_params.rI[None, (m_atom, m)],
                        topV,
                        topI,
                    )

        # ///////////////////////////////////////////////////////////////////////////////
        # online softmax
        # ///////////////////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////////////////
        # O gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # Convert layout of acc_S to gemm O accept layout.
        # Due to the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # load first V k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        # mma for O
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            # load next V k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax and rescale acc_O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        :param in_mask_steps: in mask steps
        :type in_mask_steps: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        # if it is the first tile, create a mask for residual of S to -inf for softmax.
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # Each iteration processes one row of acc_S
        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # mask residual of S with -inf
            if cutlass.const_expr(in_mask_steps):
                if cutlass.const_expr(not self._is_causal):
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        if cute.elem_less(
                            basic_params.mK.shape[1], tScS_mn[0, c][3] + 1
                        ):
                            acc_S_mn[r, c] = -cutlass.Float32.inf
                else:
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    col_idx_limit = cutlass.min(
                        tScS_mn[r, 0][1] + 1, basic_params.mK.shape[1]
                    )
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        # only consider the column index, so the row index sets to 0.
                        if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            # (n_block_size)
            acc_S_row = acc_S_mn[r, None].load()
            # row_max_cur_row => f32
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            # quad reduction for row_max
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            # compute exp(x - max) using exp2(x * log_2(e) - max * log_2(e))
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            # acc_S_row_sum => f32
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            # if it is not the first tile, load the row r of previous row_max and minus row_max_cur_row to update row_sum.
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            # update row_max, row_sum and acc_S
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Normalize acc_O by row_sum.

        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # do quad reduction for row_sum.
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """make acc tensor as mn layout view

        :param acc: input tensor
        :type acc: cute.Tensor
        :return: acc tensor mn layout view
        :rtype: cute.Tensor
        """
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),  # MMA_N
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),  # MMA_N
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """thread quad reduction

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: x + y)

    @cute.jit
    def _bitonic_merge(
        self,
        a,
        idx,
        low: cutlass.Constexpr[int] = 0,
        count: cutlass.Constexpr[int] = None,
        direction: cutlass.Constexpr[int] = 0,
    ):
        if cutlass.const_expr(count is None):
            count = cute.size(a)
        if cutlass.const_expr(count > 1):
            k = count // 2
            for i in cutlass.range_constexpr(low, low + k):
                fst, snd = a[i], a[i + k]
                if (direction == 1 and fst > snd) or (direction == 0 and fst < snd):
                    a[i], a[i + k] = snd, fst
                    idx[i], idx[i + k] = idx[i + k], idx[i]

            self._bitonic_merge(a, idx, low, k, direction)
            self._bitonic_merge(a, idx, low + k, k, direction)

    @cute.jit
    def _bitonic_sort(
        self,
        a,
        idx,
        low: cutlass.Constexpr[int] = 0,
        count: cutlass.Constexpr[int] = None,
        direction: cutlass.Constexpr[int] = 0,
    ):
        if cutlass.const_expr(count is None):
            count = cute.size(a)
        if cutlass.const_expr(count > 1):
            k = count // 2
            self._bitonic_sort(a, idx, low, k, direction)
            self._bitonic_sort(a, idx, low + k, k, not direction)
            self._bitonic_merge(a, idx, low, count, direction)

    @cute.jit
    def _bitonic_merge_in_left(
        self, a1, idx1, a2, idx2, left_direction: cutlass.Constexpr = 0
    ):
        k = cute.size(a1)
        for i in cutlass.range_constexpr(k):
            fst, snd = a1[i], a2[i]
            if fst < snd:
                a1[i] = snd
                idx1[i] = idx2[i]

        self._bitonic_merge(a1, idx1, 0, k, left_direction)


class FlashTopK(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
        top_k: int = 8,  # 16 also works
        softmax_scale: float = 1.0,
        log_smem: bool = False,
    ):
        super().__init__()
        flash_topk = self._compile(
            head_dim,
            num_head,
            is_causal,
            m_block_size,
            n_block_size,
            num_threads,
            top_k,
            softmax_scale,
            log_smem,
        )

        self.flash_topk = flash_topk

    def forward(self, q, k, v):
        return self.flash_topk(q, k, v)

    @staticmethod
    def _convert(tns: torch.Tensor, dim: int, width: int) -> cute.Tensor:
        return (
            from_dlpack(tns.detach(), assumed_align=16)
            .mark_layout_dynamic(leading_dim=dim)
            .mark_compact_shape_dynamic(
                mode=dim,
                stride_order=tns.dim_order(),
                divisibility=(128 // width),
            )
        )

    @staticmethod
    @functools.cache
    def _compile(
        head_dim: int,
        num_head: int,
        is_causal: bool,
        m_block_size: int,
        n_block_size: int,
        num_threads: int,
        top_k: int,
        softmax_scale: float,
        log_smem: bool,
    ):
        kernel = FlashAttnTopKKernel(
            head_dim,
            m_block_size,
            n_block_size,
            num_threads,
            is_causal,
            top_k,
            log_smem,
        )

        def create_tensor() -> cute.Tensor:
            torch_tensor = (
                torch.empty((1, 4096, num_head, head_dim), dtype=torch.int32)
                .random_(-2, 2)
                .to(torch.bfloat16)
                .cuda()
            )
            return FlashTopK._convert(torch_tensor, 3, 2), torch_tensor

        def create_indices(dtype):
            shape = (
                1,
                math.ceil(4096 / m_block_size),
                num_head,
                top_k,
                m_block_size,
            )
            torch_tensor = torch.full(shape, -1, dtype=dtype).movedim(4, 3).cuda()
            cute_tensor = FlashTopK._convert(torch_tensor, 3, 4)
            return cute_tensor, torch_tensor

        def create_lse():
            torch_tensor = torch.full(
                (1, head_dim, 4096), -1, dtype=torch.float32
            ).cuda()
            cute_tensor = FlashTopK._convert(torch_tensor, 2, 128)  # ragged last
            return cute_tensor, torch_tensor

        torch.cuda.manual_seed(0)

        q, _q_torch = create_tensor()
        k, _k_torch = create_tensor()
        v, _v_torch = create_tensor()
        o, _o_torch = create_tensor()
        i, _i_torch = create_indices(torch.int32)
        t, _t_torch = create_indices(torch.float32)
        lse, _lse_torch = create_lse()

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        func = cute.compile(kernel, q, k, v, o, i, t, lse, softmax_scale, stream)

        @torch.library.custom_op(
            f"wsivit::flash_topk_{head_dim}_{num_head}_"
            f"{m_block_size}_{n_block_size}_{num_threads}_{is_causal}_{top_k}",
            mutates_args={},
            device_types="cuda",
        )
        def flash_topk(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            out = query.new_empty(
                query.size(0),
                query.size(1),
                query.size(2),
                value.size(3),
                dtype=torch.bfloat16,
                device="cuda",
            )
            idx = torch.empty(
                (
                    query.size(0),
                    math.ceil(query.size(1) / m_block_size),
                    query.size(2),
                    top_k,
                    m_block_size,
                ),
                dtype=torch.int32,
                device="cuda",
            ).movedim(4, 3)
            top_val = torch.empty_like(idx, dtype=torch.float32)
            lse = torch.empty(
                (query.size(0), query.size(2), query.size(1)),
                dtype=torch.float32,
                device="cuda",
            )
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)
            func(
                FlashTopK._convert(query, 3, 2),
                FlashTopK._convert(key, 3, 2),
                FlashTopK._convert(value, 3, 2),
                FlashTopK._convert(out, 3, 2),
                FlashTopK._convert(idx, 3, 4),
                FlashTopK._convert(top_val, 3, 4),
                FlashTopK._convert(lse, 2, 128),
                # softmax_scale (constexpr elided),
                stream,
            )
            idx = idx.movedim(3, 2).reshape(
                query.size(0),
                math.ceil(query.size(1) / m_block_size) * m_block_size,
                query.size(2),
                top_k,
            )[:, : query.size(1)]
            top_val = top_val.movedim(3, 2).reshape(
                query.size(0),
                math.ceil(query.size(1) / m_block_size) * m_block_size,
                query.size(2),
                top_k,
            )[:, : query.size(1)]
            return out, idx, top_val, lse

        @flash_topk.register_fake
        def _(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
            out = query.new_empty(
                query.size(0),
                query.size(1),
                query.size(2),
                value.size(3),
                dtype=torch.bfloat16,
                device="cuda",
            )
            topk = query.new_empty(
                (
                    query.size(0),
                    query.size(1),
                    query.size(2),
                    top_k,
                ),
                dtype=torch.int32,
                device="cuda",
            )
            lse = query.new_empty(
                (query.size(0), query.size(2), query.size(1)),
                dtype=torch.float32,
                device="cuda",
            )
            return out, topk, torch.empty_like(topk, dtype=torch.float32), lse

        def backward(ctx, d_out, d_idx, d_top_val, d_lse):
            q, k, v = ctx.qkv
            o, lse = ctx.ol
            d_q = torch.full_like(q, 0)
            d_k = torch.full_like(k, 0)
            d_v = torch.full_like(v, 0)
            _flash_attn_backward(
                d_out,
                q,
                k,
                v,
                o,
                lse,
                d_q,
                d_k,
                d_v,
                0,  # dropout
                softmax_scale,
                False,  # causal
                -1,  # window
                -1,  # window
                0,  # softcap
                None,  # alibi
                False,  # deterministic
            )
            return d_q, d_k, d_v

        def setup_context(ctx, inputs, output):
            o, _, _, lse = output
            ctx.qkv = inputs
            ctx.ol = o, lse

        flash_topk.register_autograd(backward, setup_context=setup_context)
        return flash_topk


def example():
    print("Compiling kernel...")
    start = time.perf_counter()
    num_head = 4
    num_tokens = 4096
    scale = 1 / 8
    # NSA configuratie, top 8 (or 16) blocks
    ftk = FlashTopK(
        head_dim=64,
        num_head=num_head,
        top_k=8,
        log_smem=True,
        softmax_scale=scale,
    )

    print(f"took {time.perf_counter() - start:.1f}s.")
    q = torch.randn(1, num_tokens, num_head, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, num_tokens, num_head, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, num_tokens, num_head, 64, dtype=torch.bfloat16, device="cuda")
    print("Running tests...")
    if num_tokens <= 4096:
        torch.library.opcheck(ftk.flash_topk, (q, k, v), raise_exception=False)
        print("opcheck OK")
        q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True)
        o_ftk, _, _, lse_ftk = ftk(q, k, v)
        lse_torch = (
            torch.einsum("qd,kd->qk", q[0, :16, 0, :].float(), k[0, :, 0, :].float())
            .mul(scale)
            .logsumexp(-1)
        )
        torch.testing.assert_close(lse_ftk[0, 0, :16], lse_torch)
        print("lse probably OK")
        o_ftk.sum().backward()
        qg_ftk, kg_ftk, vg_ftk = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None
        o_nn = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
        ).transpose(1, 2)
        o_nn.sum().backward()
        # ATOL is a bit on the high side (0.0005), but there's no real way around it
        # given BF16 and large reductions. RTOL is default for BF16.
        torch.testing.assert_close(o_ftk, o_nn, atol=5e-4, rtol=1.6e-2)
        print("output OK")
        torch.testing.assert_close(q.grad, qg_ftk, atol=5e-4, rtol=1.6e-2)
        torch.testing.assert_close(k.grad, kg_ftk, atol=5e-4, rtol=1.6e-2)
        torch.testing.assert_close(v.grad, vg_ftk, atol=5e-4, rtol=1.6e-2)
        print("grad OK")
    else:
        print("skipped checks")
    torch.cuda.synchronize()
    start = time.perf_counter()
    o, i, t, lse = ftk(q, k, v)
    torch.cuda.synchronize()
    print(f"Forward pass in {(time.perf_counter() - start) * 1000:.1f}ms")
    print(f"{o.shape=} {i.shape=} {t.shape=} {lse.shape=}")
    print(f"lse values: {lse.flatten()[:10]}")
    print("Predicted top-K: (first 8)")
    print(f"values: {t[0, :8, 0]}")
    print(f"indices: {i[0, :8, 0]}")
    print("Actual top-K (can differ slightly due to ties):")
    scores = torch.einsum("qd,kd->qk", q[0, :8, 0].float(), k[0, :, 0].float())
    print(scores.topk(8, dim=1))

    # Example shapes:
    #       o       [1, 4096, 4, 64]
    #       i       [1, 4096, 4, 8]
    #       t       [1, 4096, 4, 8]
    #       lse     [1, 4, 4096]


if __name__ == "__main__":
    example()
