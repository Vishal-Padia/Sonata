import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils


_BLOCK = 256
VEC = 8

@cute.kernel
def _skinny_gemv_kernel(
    y: cute.Tensor,   # (M, N) bf16 output
    x: cute.Tensor,   # (M, K) bf16 activation
    W: cute.Tensor,   # (N, K) bf16 weight
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = bidx * bdim + tidx


    if n < N:
        for m in cutlass.range(M):
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(K):
                xv = x[m, k].to(cutlass.Float32)
                wv = W[n, k].to(cutlass.Float32)
                acc = acc + xv * wv
            # Downcast fp32 -> bf16 only in the epilogue (the store)
            y[m, n] = acc.to(cutlass.BFloat16)


@cute.jit
def skinny_gemv(
    y: cute.Tensor,
    x: cute.Tensor,
    W: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    stream,
):
    # One thread per output column; enough CTAs to cover all N columns
    grid = (N + _BLOCK - 1) // _BLOCK
    _skinny_gemv_kernel(y, x, W, M, N, K).launch(
        grid=[grid, 1, 1],
        block=[_BLOCK, 1, 1],
        stream=stream,
    )


@cute.kernel
def _skinny_gemv_coalesced_kernel(y, x, W, M, N, K):
    lane, _, _ = cute.arch.thread_idx()
    n, _, _ = cute.arch.block_idx()

    if n < N:
        for m in cutlass.range(M):
            partial = cutlass.Float32(0.0)
            for k in cutlass.range(lane, K, 32):
                partial = partial + x[m, k].to(cutlass.Float32) * W[n, k].to(cutlass.Float32)
            # 32 lanes each hold a partial, sum them into one number
            total = cute.arch.warp_reduction_sum(partial)
            if lane == 0:
                y[m, n] = total.to(cutlass.BFloat16)

# Split-K is specialized per M (1 or 2) in Python rather than with a Constexpr
# annotation, because `cutlass.Constexpr` as a param type segfaults this DSL build.
# M is only ever 1 (streaming) or 2 (cfg-doubled), so two kernels cover every case.
# Both split each output column across `warps` (= S) warps of a block, reduce each
# warp with a shuffle, then combine the S warp-totals through a 32-slot smem scratch.

@cute.kernel
def _skinny_gemv_splitk_m1_kernel(y, x, W, M, N, K):
    tid, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    n, _, _ = cute.arch.block_idx()
    warp_id = tid // 32
    lane = tid % 32
    warps = bdim // 32

    smem = utils.SmemAllocator()
    s = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=4)
    if n < N:
        acc0 = cutlass.Float32(0.0)
        for k in cutlass.range(tid, K, bdim):
            acc0 = acc0 + x[0, k].to(cutlass.Float32) * W[n, k].to(cutlass.Float32)
        w0 = cute.arch.warp_reduction_sum(acc0)
        if lane == 0:
            s[warp_id] = w0
        cute.arch.sync_threads()
        if tid == 0:
            a = cutlass.Float32(0.0)
            for i in cutlass.range(warps):
                a = a + s[i]
            y[0, n] = a.to(cutlass.BFloat16)


@cute.kernel
def _skinny_gemv_splitk_m2_kernel(y, x, W, M, N, K):
    tid, _, _ = cute.arch.thread_idx()
    bdim, _, _ = cute.arch.block_dim()
    n, _, _ = cute.arch.block_idx()
    warp_id = tid // 32
    lane = tid % 32
    warps = bdim // 32

    smem = utils.SmemAllocator()
    s = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=4)
    if n < N:
        # W[n, k] is loaded ONCE and fed into both rows' accumulators, so M=2 no
        # longer re-reads the weight (the fix for the cfg-decode regime).
        # acc0 = cutlass.Float32(0.0)
        # acc1 = cutlass.Float32(0.0)
        # for k in cutlass.range(tid, K, bdim):
        #     wv = W[n, k].to(cutlass.Float32)
        #     acc0 = acc0 + x[0, k].to(cutlass.Float32) * wv
        #     acc1 = acc1 + x[1, k].to(cutlass.Float32) * wv

        wt = cute.zipped_divide(W[n, None], (VEC, ))
        x0t = cute.zipped_divide(x[0, None], (VEC, ))
        x1t = cute.zipped_divide(x[1, None], (VEC, ))

        wfrag = cute.make_rmem_tensor(cute.make_layout(VEC), cutlass.BFloat16)
        x0frag = cute.make_rmem_tensor(cute.make_layout(VEC), cutlass.BFloat16)
        x1frag = cute.make_rmem_tensor(cute.make_layout(VEC), cutlass.BFloat16)

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)
        for kv in cutlass.range(tid, K // VEC, bdim):
            cute.autovec_copy(wt[None, kv], wfrag)
            cute.autovec_copy(x0t[None, kv], x0frag)
            cute.autovec_copy(x1t[None, kv], x1frag)

            for j in cutlass.range_constexpr(VEC):
                wv = wfrag[j].to(cutlass.Float32)
                acc0 = acc0 + x0frag[j].to(cutlass.Float32) * wv
                acc1 = acc1 + x1frag[j].to(cutlass.Float32) * wv

        # Reduce row 0, then reuse the same smem scratch for row 1 (bracketed by
        # barriers so a warp cannot overwrite a slot tid 0 is still reading).
        w0 = cute.arch.warp_reduction_sum(acc0)
        if lane == 0:
            s[warp_id] = w0
        cute.arch.sync_threads()
        if tid == 0:
            a = cutlass.Float32(0.0)
            for i in cutlass.range(warps):
                a = a + s[i]
            y[0, n] = a.to(cutlass.BFloat16)
        cute.arch.sync_threads()

        w1 = cute.arch.warp_reduction_sum(acc1)
        if lane == 0:
            s[warp_id] = w1
        cute.arch.sync_threads()
        if tid == 0:
            a = cutlass.Float32(0.0)
            for i in cutlass.range(warps):
                a = a + s[i]
            y[1, n] = a.to(cutlass.BFloat16)


@cute.jit
def skinny_gemv_coalesced(y, x, W, M, N, K, stream):
    _skinny_gemv_coalesced_kernel(y, x, W, M, N, K).launch(
        grid=[N, 1, 1],       # one block per output column
        block=[32, 1, 1],     # one warp per block
        stream=stream,
    )

_SPLIT = 4 # split factor S: warps cooperating per output column

@cute.jit
def skinny_gemv_splitk_m1(y, x, W, M, N, K, stream):
    _skinny_gemv_splitk_m1_kernel(y, x, W, M, N, K).launch(
        grid=[N, 1, 1],
        block=[_SPLIT * 32, 1, 1],
        stream=stream,
    )

@cute.jit
def skinny_gemv_splitk_m2(y, x, W, M, N, K, stream):
    _skinny_gemv_splitk_m2_kernel(y, x, W, M, N, K).launch(
        grid=[N, 1, 1],
        block=[_SPLIT * 32, 1, 1],
        stream=stream,
    )

# Compiled artifacts are cached per (M, N, K). Recompiling on every call would
# make the benchmark measure the JIT compiler instead of the kernel
_compiled_cache = {}

def _run(jit_fn, x, W):
    M, K = x.shape
    N = W.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    key = (jit_fn, M, N, K)          # kernel identity is part of the key
    stream = cutlass_torch.current_stream()
    if key not in _compiled_cache:
        _compiled_cache[key] = cute.compile(
            jit_fn, from_dlpack(y), from_dlpack(x), from_dlpack(W), M, N, K, stream,
        )
    _compiled_cache[key](
        from_dlpack(y), from_dlpack(x), from_dlpack(W), M, N, K, stream,
    )
    return y


def candidate_cute(x, W):            return _run(skinny_gemv, x, W)
def candidate_cute_coalesced(x, W):  return _run(skinny_gemv_coalesced, x, W)
def candidate_cute_splitk(x, W):
    # M is only ever 1 (streaming) or 2 (cfg); pick the matching specialization.
    M = x.shape[0]
    return _run(skinny_gemv_splitk_m2 if M > 1 else skinny_gemv_splitk_m1, x, W)