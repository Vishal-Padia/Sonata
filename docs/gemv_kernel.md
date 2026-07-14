# bf16 skinny-GEMV for the projections (kernel work)

Reproduce: `python kernels/bench_gemv_graph.py` (CUDA-graph timed, the fair number), `python kernels/bench_gemv.py` (plain wall time), NCU via `kernels/ncu_target.py`. CuTeDSL kernels in `kernels/gemv.py`. A10G sm86, bf16 I/O, fp32 accumulate.

> **Headline:** a hand-written CuTeDSL memory-bound GEMV **beats cuBLAS on all three
> projection shapes at M=2 (cfg-decode): 1.36× out_proj, 1.20× fc1, 1.07× in_proj**, all
> at/above the measured memory roofline. This is the quality-free per-step lever the op-level
> profiling pointed at (projections are 85% of a Mamba2 layer, memory-bound on weight reads).

## Why a custom GEMV at all

`docs/baseline_op_level.md` found the two projection matmuls are ~85% of a Mamba2 decode layer and are bound by reading the weight bytes, not math. cuBLAS is slow at M=1-2 because it runs a GEMM kernel that pads M to a tensor-core tile and wastes the MMA on 1-2 rows. A plain FMA reduction that streams the weight once (no tensor cores) is the right shape for a memory-bound load, so it can beat cuBLAS here.

## The roofline is 482 GB/s, not 600

`bench_gemv.py` measures the achievable HBM ceiling with a copy loop: **482 GB/s (80% of the 600 spec)**. That reframes the headroom. Against 482, cuBLAS is already at ~97% on in_proj and fc1 at M=1, so there is almost nothing to take on the large-N shapes at M=1. The real headroom is small-N (out_proj, where cuBLAS sits at ~72-80%) and the M=2 path. Quoting the 600 spec overstated the opportunity; the measured roof is the honest target.

## The kernel

Split-K, one output column per CTA, warps stride K so adjacent threads read adjacent k (coalesced), fp32 accumulation, bf16 store in the epilogue. Two specializations because `M` is only ever 1 (streaming) or 2 (cfg): the **m2** kernel loads `W[n,k]` once and feeds both rows' accumulators, so cfg does not re-read the weight. Loads are **128-bit vectorized** (8 bf16 per thread via `autovec_copy`), which is what actually moved the number.

## NCU: it is memory-bound, and vectorization was the lever

NCU on out_proj M=2 (`ncu_target.py`), before vs after vectorizing the loads:

| metric | scalar loads | 128-bit loads |
|---|---|---|
| Duration | 56.96 µs | **41.66 µs** |
| DRAM throughput | 65.8% | **84.7%** (507 GB/s) |
| L1/TEX throughput | 36% | 89.7% |
| Achieved occupancy | 82% | 86% |
| Compute (SM) | 31% | 45% |

It was memory-bound by design (memory >> compute), just not saturating DRAM. NCU ruled out occupancy (82-86%, fine) and compute (idle). Scalar 2-byte loads under-drove the memory system; 128-bit loads took DRAM 66% -> 85% and cut the kernel 27%. NCU also flags a tail effect on out_proj (2.13 waves, a partial wave worth up to ~33%), which is the small remaining gap.

## The graph-capture gotcha (also the integration prerequisite)

CuTeDSL launches on **stream 0 and then synchronizes** by default (`dsl.py:2771`), so under a `torch.cuda.graph` capture the kernel lands on the wrong stream and is never recorded. The graph replays empty and the bench reports a physically impossible 2.5 µs (6000+ GB/s). The fix is to thread the launch stream through the `@cute.jit` launcher (`.launch(..., stream=stream)`) and pass `cutlass_torch.current_stream()`, which is the capture stream inside the graph. This is not just a benchmarking fix: it is exactly what integration into Zonos's CUDA-graphed decode loop requires, since that decode step is captured once and replayed.

## Results (CUDA-graph timed, kernel-vs-kernel)

M=2 (cfg-decode, the production path). Both paths bf16, numerically PASS (rel err < 5e-2, at the bf16 output-rounding floor):

| shape | cuBLAS | cute_splitk | speedup | % of roof |
|---|---|---|---|---|
| out_proj (2048x4096) | 47.7 µs | **35.1 µs** | **1.36×** | 99% |
| in_proj (8512x2048) | 73.5 µs | **68.7 µs** | **1.07×** | 106% |
| fc1 (16384x2048) | 154.8 µs | **129.2 µs** | **1.20×** | 108% |

(>100% of roof because a read-mostly GEMV can exceed the read+write copy ceiling; these are at the practical read-bandwidth limit.)

M=1 (pure streaming, no cfg) is mixed: `coalesced` wins out_proj (1.08×) and fc1 (1.03×) but the split-K m1 kernel is still scalar and lags, and neither beats cuBLAS on in_proj (already ~98% roof there). Production runs cfg, so M=2 is the path that matters; vectorizing m1 is a loose end for a batch-1-no-cfg regime.

## Integrated result (under Zonos's CUDA graph)

`kernels/patch.py` monkeypatches decode-shaped, bias-free projection Linears to the GEMV
(prefill M and unsupported shapes fall back to cuBLAS; attention Linears take 3D `(M,1,K)` input,
Mamba's `step()` takes 2D `(M,K)`, both handled). `kernels/bench_decode.py`, stock vs patched, on
the stock CUDA-graph decode path:

| patch scope | GPU p50 | wall/step | RTF | vs stock |
|---|---|---|---|---|
| stock (cuBLAS) | 9.17 ms | 11.30 ms | 1.03× | — |
| Mamba2 projections (68) | 8.57 ms | 10.74 ms | 1.08× | **−6.9%** |
| + attention qkv/out/MLP (116) | **7.99 ms** | **10.22 ms** | **1.14×** | **−12.9%** |

**−12.9% decode-step GPU latency, RTF 1.03→1.14×**, and the patch is captured/replayed correctly
inside Zonos's CUDA graph (290 unique codes, not empty-graph garbage). Adding the attention layers
roughly doubled the Mamba-only win, consistent with attention being ~41% of true compute. A
quality-preserving speedup on the memory-bound step Phase 0 called mostly irreducible.

## Quality: the kernel is fine, the model is brittle (correcting the prediction above)

The prediction of "near-zero flips" was **wrong**, and why is the interesting part. The teacher-forced
probe (`kernels/gemv_probe.py`) shows **4.6% cb0 token flips** vs the cuBLAS tape, as much as int8 quant.
But a direct per-op check says the kernel is **>= cuBLAS accuracy** vs fp32 (out_proj 1.66e-3 vs 2.41e-3;
in_proj/fc1 tied; cute-vs-cuBLAS diff 0.01-0.28%). So the flips are **not degradation** -- they are the
34-layer AR stack amplifying a sub-0.3% rounding difference into different token choices, the same
brittleness the quant study found. A flip from a *more-accurate* kernel is a different-but-valid rollout,
not a worse one (a driver update or a different GPU would flip tokens too).

Consequence: the flip-rate gate (built for quant, a degradation) **mislabels a correctness-preserving
swap as RED**, and you cannot claim bit-identical-to-cuBLAS output. The honest quality gate is
perceptual, not token-match: does `cute_stream.wav` (from `kernels/gen_cute_audio.py`) sound clean vs
`baseline.wav`. **Confirmed by ear: clean, no artifacts, just a different (valid) rollout** — for both
the Mamba-only and full (Mamba+attention) patches.

## Streaming, end to end: a throughput win with a TTFA cost

`python scripts/run_streaming.py --patch bench`, stock vs full patch, on the streaming sweep:

| chunk | per-frame p90 (stock → patch) | RTF (stock → patch) | TTFA p50 (stock → patch) |
|---|---|---|---|
| 14 | 12.16 → **10.92** | 0.84 → 0.90 | 565 → 607 |
| 16 | 11.22 → **9.99** | 0.89 → 0.96 | 584 → 623 |
| 18 | 10.84 → **9.64** | 0.92 → **1.00** | 602 → 641 |
| 20 | 10.69 → **9.39** | 0.93 → **1.01** | 625 → 658 |

**Steady-state per-frame drops ~13%**, so sustained streaming crosses into **>= real time** (chunk 18-20
hit RTF 1.00-1.01 vs stock's 0.93 ceiling) and the knee moves from chunk 16 to **14** (a smaller chunk now
sustains). **But TTFA rises a consistent ~40 ms** at every chunk. That is a one-time **graph-capture** cost:
the patched capture records 116 cute launches vs stock's 46 cuBLAS launches, and Zonos captures the graph
once per `generate()`, so TTFA (first audio within the call) pays it.

Attempted fix (this is the informative part): pre-bind a fixed input buffer + copy-in to eliminate the
per-call `from_dlpack`. It **did not** recover TTFA (unchanged at 623 ms) and slightly *hurt* steady-state
(added 116 copy kernels to the graph). So the ~40 ms is **not** dlpack host overhead — it is the cute
launches' own capture-time marshaling. Reverted. The real fix is to stop re-capturing the graph every
request (a persistent server captures once and replays across utterances, which amortizes it away) or to
cut the launch count by fusing; both are bigger than a wrapper tweak.

Net: the kernel is a **throughput/sustained-streaming win** (long-form, RTF >= 1) that currently trades
~40 ms of cold single-utterance TTFA. Honest tradeoff, not a free lunch.

## Next

- **Amortize the graph** (capture once, reuse across requests) to erase the TTFA cost — the change that
  turns this into a clean win on both metrics.
- Add the cute-vs-cuBLAS per-op check to `validation_harness.py` as a regression (tight tolerance, not 5e-2).
- Loose end: vectorize the M=1 (no-cfg) path; production cfg (M=2) is already covered.
