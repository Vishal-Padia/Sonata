import math
import torch
import cutlass
import cutlass.cute as cute
import torch.nn.functional as F
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

_BLK_M = 64
_BLK_N = 64
_BLK_K = 128

@cute.kernel
def _bmm_kernel(
    C: cute.Tensor,
    B: cute.Tensor,
    BATCH: cutlass.Int32,
    SEQLEN: cutlass.Int32,
    NGROUPS: cutlass.Int32,
    DSTATE: cutlass.Int32,
    CHUNK: cutlass.Int32,
    NCHUNKS: cutlass.Int32,
    CB_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    # Decode bidz -> (b, c, g). Flattening: bidz = b*(NCHUNKS*NGROUPS) + c*NGROUPS + g,
    # so b is outermost, g is innermost. Reverse with two divmods.
    b   = bidz // (NCHUNKS * NGROUPS)
    rem = bidz %  (NCHUNKS * NGROUPS)
    c   = rem // NGROUPS
    g   = rem %  NGROUPS

    # === Stride expressions ===
    # Toy shape: BATCH=1, SEQLEN=512, NGROUPS=2, DSTATE=128, CHUNK=256, NCHUNKS=2.
    #
    # C and B layout: (BATCH, SEQLEN, NGROUPS, DSTATE), contiguous bf16.
    #   strides = (SEQLEN*NGROUPS*DSTATE, NGROUPS*DSTATE, DSTATE, 1) = (131072, 256, 128, 1).
    #   C.stride()[1] = 256 = NGROUPS*DSTATE  -- confirms the seqlen (chunk) stride.
    #
    # C base for this GEMM, element (b, c*CHUNK + row_i, g, k):
    #   c_base = b*(SEQLEN*NGROUPS*DSTATE) + (c*CHUNK + row_i)*(NGROUPS*DSTATE) + g*DSTATE + k
    #         = b*131072 + (c*256 + row_i)*256 + g*128 + k
    # B base for this GEMM, element (b, c*CHUNK + row_j, g, k):
    #   b_base = b*131072 + (c*256 + row_j)*256 + g*128 + k     (same, row_j instead of row_i)
    #
    # CB_out layout: (BATCH, NCHUNKS, NGROUPS, CHUNK, CHUNK), contiguous fp32.
    #   strides = (NCHUNKS*NGROUPS*CHUNK*CHUNK, NGROUPS*CHUNK*CHUNK, CHUNK*CHUNK, CHUNK, 1)
    #          = (262144, 131072, 65536, 256, 1).
    # CB_out base for this GEMM, element (b, c, g, row_i, row_j):
    #   cb_base = ((b*NCHUNKS + c)*NGROUPS + g)*(CHUNK*CHUNK) + row_i*CHUNK + row_j
    #          = ((b*2 + c)*2 + g)*65536 + row_i*256 + row_j
    #
    # === Concrete check for (b,c,g) = (0,1,0)  [chunk 1, group 0 -- the chunk-stride tripwire] ===
    #   c_base(row_i=0, k=0)      = 0 + (1*256 + 0)*256 + 0 + 0 = 65536   -> C[0, 256, 0, 0]
    #   b_base(row_j=0, k=0)      = 65536                            -> B[0, 256, 0, 0]
    #   cb_base(row_i=0, row_j=0) = ((0*2+1)*2+0)*65536 + 0 + 0 = 131072 -> CB_out[0, 1, 0, 0, 0]
    #   Verified: torch CB_out.stride()[1] = 131072, so CB_out[0,1,0,0,0] flat offset = 131072. ✓
    #
    # === Thread -> element mapping choice ===
    # We do NOT hand-map threads to output elements. With atom_layout_mnk = (2,2,1) and
    # MmaF16BF16Op (16x8x16), the TiledMMA partitions the 64x64 output tile across 4 warps
    # (128 threads). thr_mma = tiled_mma.get_slice(tidx); tCgC = thr_mma.partition_C(gC)
    # returns each thread's owned slice of the tile. The K-loop walks DSTATE=128 in BK=16
    # steps (8 iterations), calling cute.gemm(tiled_mma, acc, rA, rB, acc) per step. The
    # tile's (row_i, row_j) -> thread assignment is owned entirely by the TiledMMA.

    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)))

    C_bg = cute.slice_(C, (b, None, g, None))   # (SEQLEN, DSTATE)
    B_bg = cute.slice_(B, (b, None, g, None))   # (SEQLEN, DSTATE)
    CB_bcg = cute.slice_(CB_out, (b, c, g, None, None))  # (CHUNK, CHUNK)

    # Tile the 2D slices. coord is (m_tile, k_tile); k_tile=0 since DSTATE=128 fills one K-tile.
    gA = cute.local_tile(C_bg, tiler=(_BLK_M, _BLK_K), coord=(c * (CHUNK // _BLK_M) + bidx, 0))
    gB = cute.local_tile(B_bg, tiler=(_BLK_N, _BLK_K), coord=(c * (CHUNK // _BLK_N) + bidy, 0))
    gC = cute.local_tile(CB_bcg, tiler=(_BLK_M, _BLK_N), coord=(bidx, bidy))

    thr_mma = tiled_mma.get_slice(tidx)
    tCrA = thr_mma.partition_A(gA)
    tCrB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    acc  = thr_mma.make_fragment_C(tCgC.shape)
    acc.fill(0.0)

    rA = thr_mma.make_fragment_A(tCrA.shape)
    rB = thr_mma.make_fragment_B(tCrB.shape)
    cute.autovec_copy(tCrA, rA)
    cute.autovec_copy(tCrB, rB)
    cute.gemm(tiled_mma, acc, rA, rB, acc)
    cute.autovec_copy(acc, tCgC)


@cute.jit
def launch_bmm(
    C: cute.Tensor,
    B: cute.Tensor,
    BATCH: cutlass.Int32,
    SEQLEN: cutlass.Int32,
    NGROUPS: cutlass.Int32,
    DSTATE: cutlass.Int32,
    CHUNK: cutlass.Int32,
    NCHUNKS: cutlass.Int32,
    CB_out: cute.Tensor,
    stream,
):
    m_tiles = (CHUNK + _BLK_M - 1) // _BLK_M
    n_tiles = (CHUNK + _BLK_N - 1) // _BLK_N
    grid = [m_tiles, n_tiles,  BATCH * NCHUNKS * NGROUPS]
    block = (128, 1, 1)
    _bmm_kernel(C, B, BATCH, SEQLEN, NGROUPS, DSTATE, CHUNK, NCHUNKS, CB_out).launch(
        grid=grid,
        block=block,
        stream=stream,
    )

def test_bmm():
    # batch=1, seqlen=512, ngroups=1, dstate=128, chunk_size=256 -> nchunks=2.
    BATCH, SEQLEN, NGROUPS, DSTATE, CHUNK = 1, 512, 2, 128, 256
    NCHUNKS = SEQLEN // CHUNK  # 2

    # Seeded generator so a failure is reproducible bit-for-bit across runs.
    g = torch.Generator(device="cuda").manual_seed(0)

    C = torch.randn(BATCH, SEQLEN, NGROUPS, DSTATE, device="cuda",
                   dtype=torch.bfloat16, generator=g)
    B = torch.randn(BATCH, SEQLEN, NGROUPS, DSTATE, device="cuda",
                   dtype=torch.bfloat16, generator=g)
    CB_out = torch.empty(BATCH, NCHUNKS, NGROUPS, CHUNK, CHUNK, dtype=torch.float32, device='cuda')
    stream = cutlass_torch.current_stream()

    C_chunked = C.reshape(BATCH, NCHUNKS, CHUNK, NGROUPS, DSTATE)
    B_chunked = B.reshape(BATCH, NCHUNKS, CHUNK, NGROUPS, DSTATE)
    CB_ref = torch.einsum("bcigd,bcjgd->bcgij", C_chunked.float(), B_chunked.float())

    assert CB_ref.shape == (BATCH, NCHUNKS, NGROUPS, CHUNK, CHUNK), (
        f"ref shape wrong: {tuple(CB_ref.shape)} "
        f"expected {(BATCH, NCHUNKS, NGROUPS, CHUNK, CHUNK)}"
    )

    hand_diag = (C[0, 0, 0, :].float() * B[0, 0, 0, :].float()).sum().item()
    assert abs(hand_diag - CB_ref[0, 0, 0, 0, 0].item()) < 1e-4, (
        f"ref diag self-check failed: einsum={CB_ref[0,0,0,0,0].item():.6f} "
        f"hand={hand_diag:.6f}"
    )
    hand_off = (C_chunked[0, 0, 3, 0, :].float() * B_chunked[0, 0, 7, 0, :].float()).sum().item()
    assert abs(hand_off - CB_ref[0, 0, 0, 3, 7].item()) < 1e-4, (
        f"ref off-diag self-check failed: einsum={CB_ref[0,0,0,3,7].item():.6f} "
        f"hand={hand_off:.6f}"
    )

    print(f"[ref] CB_ref shape={tuple(CB_ref.shape)}  "
          f"min={CB_ref.min().item():.4f} max={CB_ref.max().item():.4f}  "
          f"abs_mean={CB_ref.abs().mean().item():.4f}")
    print(f"[ref] CB_ref[0,0,0,0,0]={CB_ref[0,0,0,0,0].item():.6f} "
          f"(hand={hand_diag:.6f})")
    print(f"[ref] CB_ref[0,0,0,3,7]={CB_ref[0,0,0,3,7].item():.6f} "
          f"(hand={hand_off:.6f})")
    assert CB_ref[0,0,0,3,7] != CB_ref[0,0,0,7,3]

    compiled = cute.compile(launch_bmm, from_dlpack(C, assumed_align=16), from_dlpack(B, assumed_align=16), BATCH, SEQLEN, NGROUPS, DSTATE, CHUNK, NCHUNKS, from_dlpack(CB_out, assumed_align=16), stream=stream)
    compiled(from_dlpack(C, assumed_align=16), from_dlpack(B, assumed_align=16), BATCH, SEQLEN, NGROUPS, DSTATE, CHUNK, NCHUNKS, from_dlpack(CB_out, assumed_align=16), stream=stream)
    torch.cuda.synchronize()
    assert CB_out.shape == CB_ref.shape
    max_abs = (CB_out - CB_ref).abs().max().item()
    assert torch.allclose(CB_out, CB_ref, atol=1e-3, rtol=1e-2), (
        f"kernel mismatch: max_abs={max_abs}"
    )
    print(f"[kernel] max_abs={max_abs:.6f}  (tol: atol=1e-3 rtol=1e-2)")

def test_bmm_ragged():
    """Partial-last-chunk tripwire -- the real Zonos regime (prefill 60-125 tokens =
    one partial chunk of 256). The kernel currently has NO edge masking, so for a
    chunk whose valid rows < CHUNK it reads the junk rows past seqlen and corrupts
    the output.

    This is a pure correctness test, not an OOB test: the buffer is padded to
    NCHUNKS*CHUNK (so every indexed row is in-bounds -- no crash), but the tail rows
    (>= SEQLEN) are poisoned with garbage. The reference encodes the *masked
    semantics* we want -- CB[...,i,j] must be 0 wherever i or j is an out-of-range
    (padded) position, NOT the dot of garbage. RED until masked edge loads land in
    the kernel; GREEN after.
    """
    BATCH, NGROUPS, DSTATE, CHUNK = 1, 2, 128, 256
    SEQLEN = 300                          # not a multiple of CHUNK -> last chunk is partial
    NCHUNKS = math.ceil(SEQLEN / CHUNK)   # 2
    PADDED = NCHUNKS * CHUNK              # 512; kernel walks full chunks, so allocate padded
    valid_last = SEQLEN - (NCHUNKS - 1) * CHUNK  # 44 real rows in the last chunk

    g = torch.Generator(device="cuda").manual_seed(0)
    C = torch.randn(BATCH, PADDED, NGROUPS, DSTATE, device="cuda",
                   dtype=torch.bfloat16, generator=g)
    B = torch.randn(BATCH, PADDED, NGROUPS, DSTATE, device="cuda",
                   dtype=torch.bfloat16, generator=g)
    # Poison the tail (rows >= SEQLEN). A correctly-masked kernel ignores these;
    # the current unmasked kernel reads them and corrupts the partial chunk.
    C[:, SEQLEN:] = 7.0
    B[:, SEQLEN:] = 7.0

    CB_out = torch.empty(BATCH, NCHUNKS, NGROUPS, CHUNK, CHUNK, dtype=torch.float32, device="cuda")
    stream = cutlass_torch.current_stream()

    # Reference with masked semantics: zero the tail rows, THEN einsum. This yields
    # CB[...,i,j] = 0 wherever i or j is an out-of-range (padded) position.
    Cm, Bm = C.clone(), B.clone()
    Cm[:, SEQLEN:] = 0
    Bm[:, SEQLEN:] = 0
    Cc = Cm.reshape(BATCH, NCHUNKS, CHUNK, NGROUPS, DSTATE)
    Bc = Bm.reshape(BATCH, NCHUNKS, CHUNK, NGROUPS, DSTATE)
    CB_ref = torch.einsum("bcigd,bcjgd->bcgij", Cc.float(), Bc.float())

    compiled = cute.compile(launch_bmm, from_dlpack(C, assumed_align=16), from_dlpack(B, assumed_align=16),
                            BATCH, PADDED, NGROUPS, DSTATE, CHUNK, NCHUNKS,
                            from_dlpack(CB_out, assumed_align=16), stream=stream)
    compiled(from_dlpack(C, assumed_align=16), from_dlpack(B, assumed_align=16),
             BATCH, PADDED, NGROUPS, DSTATE, CHUNK, NCHUNKS,
             from_dlpack(CB_out, assumed_align=16), stream=stream)
    torch.cuda.synchronize()

    # Diagnostics: the full chunk should already be correct; the partial chunk is the
    # tripwire. Within the partial chunk, the valid 44x44 block reads only real rows
    # so it should match even now -- the corruption is entirely in the out-of-range
    # region (rows/cols >= 44) where the kernel has garbage but the ref has 0.
    full_err  = (CB_out[:, 0] - CB_ref[:, 0]).abs().max().item()
    part_err  = (CB_out[:, 1] - CB_ref[:, 1]).abs().max().item()
    valid_err = (CB_out[:, 1, :, :valid_last, :valid_last]
                 - CB_ref[:, 1, :, :valid_last, :valid_last]).abs().max().item()
    print(f"[ragged] SEQLEN={SEQLEN} CHUNK={CHUNK} nchunks={NCHUNKS} valid_last={valid_last}")
    print(f"[ragged] full-chunk    max_abs = {full_err:.6e}")
    print(f"[ragged] valid-block   max_abs = {valid_err:.6e}  (the real {valid_last}x{valid_last} dots -- should be ~0 already)")
    print(f"[ragged] partial-chunk max_abs = {part_err:.6e}  (out-of-range must go to 0 after masking)")

    max_abs = (CB_out - CB_ref).abs().max().item()
    assert torch.allclose(CB_out, CB_ref, atol=1e-3, rtol=1e-2), (
        f"ragged partial-chunk mismatch: max_abs={max_abs}  "
        f"(EXPECTED RED until masked edge loads are added to the kernel)"
    )
    print(f"[ragged] max_abs={max_abs:.6f}  PASS")

if __name__ == "__main__":
    test_bmm()
    test_bmm_ragged()