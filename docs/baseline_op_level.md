# Op-level breakdown inside one Mamba2 layer

Reproduce: `python scripts/op_level_profile.py` (chrome trace → `docs/op_level_trace.json`). One real mamba2 layer (idx 2), `Block.forward` + `mixer.step()` driven directly, 200 iters each under `torch.profiler`, fully eager, batch=2 (cfg-doubled), A10G sm86. Separate, shorter run from the CUDA-event pass (the profiler perturbs timing).

> **Headline:** Inside a Mamba2 decode step, **~85% of the compute is the two projection GEMVs
> (memory-bound weight reads), and only ~7% is the SSM-specific work (selective-scan + conv1d).**
> The selective-scan kernel is *already* at the state-memory roofline (6.99 µs to read+write the
> ~4 MB state). The big wall-time gap (compute ≈146 µs vs device-timeline ≈812 µs/step) is launch
> bubbles — already removed by the shipped CUDA graph. **This changes what Phase 1 can claim.**

## Per-step op breakdown (one mamba2 layer, self-CUDA time)

| op | kernel | µs/step | % of layer compute | bound by |
|---|---|---|---|---|
| `in_proj` GEMV (2048→8512) | cutlass_80_wmma_bf16 | 73.6 | 50% | weight read (~35 MB) |
| `out_proj` GEMV (4096→2048) | ampere_bf16_s16816gemm | 51.0 | 35% | weight read (~17 MB) |
| `selective_state_update` | `_selective_scan_update_kernel` | 6.99 | 4.8% | **state R/W (~4 MB) — at roofline** |
| `causal_conv1d_update` | `causal_conv1d_update_kernel` | 3.37 | 2.3% | conv state + weight |
| gated RMSNorm (+ entry norm) | `_layer_norm_fwd_1pass` | 2.82 | 1.9% | elementwise |
| `A = -exp(A_log)` | `aten::exp` + `aten::neg` | 2.94 | 2.0% | **recomputed every step (constant!)** |
| `copy_` / `Memset` (state bookkeeping) | elementwise | 2.67 | 1.8% | - |
| **sum (compute)** | | **≈146 µs** | 100% | |
| device-timeline span (eager, w/ bubbles) | | ≈812 µs | - | launch-bound |

Projections = **124.6 µs = 85%**. SSM-specific (scan + conv) = **10.4 µs = 7%**. Everything else ~8%.

## This refines the true compute split (corrects the eager 69/27)

The per-block doc flagged that eager per-layer times are launch-bubble-inflated. Now I can quantify it. A mamba2 layer's *compute* is ~146 µs. ×34 layers = **4.97 ms**. The CUDA-graph step (bubble-free) is **9.00 ms** GPU. So the remaining ~4.0 ms is the 12 attention layers + embed/heads/cfg/sampling -> ~310 µs per attention layer (their big 2048↔8192 MLP GEMVs are compute-heavy, few bubbles).

| | eager (bubble-inflated) | **true compute (graph regime, est.)** |
|---|---|---|
| mamba2 (34) | 68.8% | **~55%** (4.97 ms) |
| attention (12) | 27.2% | **~41%** (~3.7 ms) |
| embed+heads+cfg+sampling | ~4% | ~4% |

That 146 µs × 34 + ~3.7 ms ≈ 9 ms reconciliation is independent confirmation the decomposition is trustworthy. **Under the production (graphed) path, attention is nearly as expensive as Mamba2** - far more than the eager 27% suggested. The decode-attention kernel is more valuable than we thought.

## What this means for Phase 1 (honest reframing)

1. **Fusing selective-scan + conv + gated-norm — the headline "fused Mamba2 decode step" — touches only ~7% of a layer's compute (~10 µs of 146 µs).** A *perfect* fusion of that path cannot speed the layer by more than ~7% in compute. And `selective_state_update` is *already* at the state-memory roofline (6.99 µs for ~4 MB R/W), so there is essentially nothing to reclaim there.
2. **The layer is dominated by projection weight reads (~52 MB/layer, 85% of compute)** - fundamentally irreducible at batch 1–2 (you must read the weights once per step). No fusion avoids that floor. So a **2–3× per-step speedup from scan fusion alone is not achievable on this hardware**; the plan's success threshold needs to be re-derived against this.
3. **The historical wall-time killer (launch bubbles, ~5–6×) is already gone** via the shipped CUDA graph.
Phase 1 cannot re-win it.
4. **Real, smaller wins that do exist:**
   - **Free:** `A = -exp(A_log)` is recomputed every step but is constant - precompute once, save ~2.9 µs/layer × 34 ≈ **100 µs/step** (~1% of the 9 ms step). Trivial, do it.
   - Fuse the gated-norm + activation into the `out_proj` epilogue and collapse conv+scan+norm into one kernel: removes ~4 kernel launches/layer and the small `zxbcdt`/`xBC`/`y` intermediate round-trips. Helps graph size and a few µs; not a headline.
   - A better **memory-bound GEMV** for the projections at M=1/2 (tensor cores are underutilized here - cutlass/ampere GEMM kernels at M=2 run above the pure-bandwidth floor: 73.6 µs vs ~58 µs ideal for 35 MB). Closing that gap on both projections is the *largest* realistic per-layer win, and it's a GEMV-kernel problem, not an SSM problem.
5. **Re-weight toward attention.** At ~41% true compute, the decode-attention layers (dominated by the 8192-wide MLP GEMVs) are a co-equal target. The Dr-GRPO decode-attention kernel is closer to headline than the per-block eager numbers implied.

Caveat on scope: this is **batch-1/2 steady-state decode on A10G**. At server batch > 1 the projections become compute-bound GEMMs and the balance shifts; the Phase 3 chunked-scan *prefill* is a different (tensor-core) regime where the scan genuinely matters. The finding is specific to the streaming TTFA regime Phase 1 targets - and there, the SSM scan is not where the time is.

## Method notes

- `LayerNormFn` shows 600 calls = entry add-norm (200, block only) + internal gated RMSNorm (400, block+mixer); both map to `_layer_norm_fwd_1pass_kernel`. `aten::mm` = 800 = 2 projections × 400.
- Self-CUDA kernel time = GPU-busy; the `record_function` region's CUDA-total (812 µs/step) spans inter-kernel idle and is *not* compute. The ~146 µs compute is the sum of self-CUDA kernels.
- Batch=2 matches cfg; at true batch=1 streaming the projections dominate even more (state halves to ~2 MB, weights unchanged).
