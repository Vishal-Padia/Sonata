# Per-block decode-loop attribution

> **Steady-state per-step decode latency on NVIDIA A10G (sm86):** p50 = **9.00 ms** GPU, **10.07 ms** wall (full production step, CUDA-graph path, incl. sampling). Per-frame budget at 1× RTF = **11.61 ms** -> current **RTF = 11.61 / 10.07 = 1.15×** (backbone decode only, excludes vocode). This replaces the 5.8 ms placeholder - that was a 4090 estimate; the A10G dev box is the real baseline Phase 1 is measured against.

Reproduce: `python scripts/per_block_breakdown.py`. Fixed: seed 421, the prompt in the script, cfg_scale=2.0, bf16, 50-step warmup, 540-step measure, first 20 decode steps dropped. Two passes - **eager** (CUDA graph + torch.compile OFF, so per-layer forward hooks fire) for the attribution, **CUDA-graph** (stock path) for the headline. GPU time = `torch.cuda.Event` pairs read after one `synchronize()`; wall time = `perf_counter` around a synced generate.

## The questions, answered

**Q1 - Mamba2 vs attention vs everything else?** Mamba2 is the dominant bucket.

| bucket | layers | sum p50 | mean/layer | % of eager GPU step |
|---|---|---|---|---|
| **mamba2** | 34 | 26.93 ms | 792 µs | **68.8%** |
| attention | 12 | 10.67 ms | 889 µs | 27.2% |
| backbone overhead (bb - Σlayer) | - | 0.75 ms | - | 1.9% |
| embed + heads + cfg (step - bb) | - | 0.82 ms | - | 2.1% |
| **eager decode GPU step** | 46 | **39.17 ms** | - | 100% (p90 40.61) |

Mamba2 is well above the 40% threshold, so **Phase 1's framing holds: fuse the Mamba2 decode step first.** Note per-layer attention (889 µs) is actually slightly *more* expensive than per-layer mamba2 (792 µs) - mamba2 wins the bucket purely on **count (34 vs 12)**, which is the robust part of this conclusion regardless of bubble effects (see caveat below). Attention is 27% - secondary on this hardware, not the headline; the Dr-GRPO decode-attention kernel is a supporting, not leading, Phase 1 win.

**Q2 - host/launch overhead?** This is the surprise, and it's already handled.

- Eager **GPU**/step = 39.17 ms, but CUDA-graph **GPU**/step = **9.00 ms** for the *same kernels*.
- A graph can't make kernels faster - it removes launch latency. So ~**30 ms (77%)** of the eager GPU time was **inter-kernel idle bubbles**: at batch 1 the decode fires hundreds of tiny memory-boundkernels and the GPU sits idle waiting on Python to launch the next one. Classic launch-bound decode.
- **But Zonos already ships CUDA-graph capture** (`model.py:140-179`), and it already collapses 39 -> 9 ms.So the plan's "if overhead > 30%, do CUDA-graph capture first" is **moot - it's done in the reference.** Phase 1's graph item is *"keep the new fused kernels graph-capturable,"* not *"introduce graphs."*
- Residual host overhead on the production path: graph wall 10.07 − graph GPU 9.00 = **~1.07 ms/step** (sampling + Python loop). Modest, ~10%; a minor Phase-1 target (fuse sampling, trim the loop).
- **The true per-step compute is memory-bound at ~9 ms** - exactly the regime the fused-kernel thesis targets (reduce HBM traffic on state + weights within that 9 ms).

**Q3 - per-layer uniform?** Yes, strikingly. All 12 attention layers ≈ 887–897 µs; all 34 mamba2 layers fall in two tight clusters: the first mamba2 after each attention layer (idx 1,5,9,…,45) ≈ 716 µs, the rest (idx 2,3,6,7,…) ≈ 833 µs - a consistent ~15% step tied to position-after-attention (likely a pipelining/measurement artifact at the attention->mamba boundary, not a compute difference). There is **no special layer-0 cost and no drift across depth**, so **one universal Mamba2 decode kernel handles all 34 layers - no per-position specialization needed.**

**Q4 - steady-state per-step latency?** See the headline. 9.00 ms GPU / 10.07 ms wall, RTF 1.15× on A10G.

## Phase 1 decisions this locks in

1. **Fuse the Mamba2 decode step first** (68.8% bucket, dominant by 34-vs-12 layer count). ✓ plan as written.
2. **CUDA-graph capture is not a Phase 1 win to "introduce"** - it's already in the baseline and already removed the 77% launch overhead. The work is to keep the fused kernel graph-compatible (stable state/cache pointers across replays).
3. **Decode-attention kernel is secondary** (27%), not the headline, on this hardware.
4. **One universal Mamba2 kernel** suffices (per-layer cost is uniform).
5. Target: cut into the **~9 ms memory-bound compute**; the fused kernel must reach the A10G memory roofline. NCU confirmation of memory-bound-near-roofline is the next-after-next step.

### Caveat on the eager per-layer numbers

The per-layer table is **eager GPU time, which includes launch bubbles** (and a little overhead from the timing hooks themselves). True per-layer *compute* can't be measured directly - under the graph the hooks don't fire. So treat the per-layer **absolutes** as upper bounds and the **split** as indicative. Bubbles likely inflate mamba2 (many tiny kernels) more than attention (few large GEMMs), so the true compute split is probably *less* mamba-heavy than 68.8% - but the 34-vs-12 count keeps mamba2 dominant either way. The headline (Q4) uses the near-uninstrumented CUDA-graph path and is not affected.

## Per-layer table (eager GPU time)

% column is share of the 39.17 ms eager GPU step. `d_int` = attn MLP intermediate (8192 for attention layers, 0 for mamba2).

| idx | type | d_int | p50 (µs) | p90 (µs) | % step |
|---|---|---|---|---|---|
| 0 | attention | 8192 | 897.0 | 919.6 | 2.3% |
| 1 | mamba2 | 0 | 721.9 | 757.8 | 1.8% |
| 2 | mamba2 | 0 | 835.6 | 859.1 | 2.1% |
| 3 | mamba2 | 0 | 833.5 | 855.0 | 2.1% |
| 4 | attention | 8192 | 890.9 | 913.4 | 2.3% |
| 5 | mamba2 | 0 | 717.8 | 747.5 | 1.8% |
| 6 | mamba2 | 0 | 833.5 | 866.3 | 2.1% |
| 7 | mamba2 | 0 | 831.5 | 855.1 | 2.1% |
| 8 | attention | 8192 | 887.8 | 908.3 | 2.3% |
| 9 | mamba2 | 0 | 717.8 | 745.7 | 1.8% |
| 10 | mamba2 | 0 | 832.5 | 855.0 | 2.1% |
| 11 | mamba2 | 0 | 831.0 | 854.0 | 2.1% |
| 12 | attention | 8192 | 888.8 | 908.3 | 2.3% |
| 13 | mamba2 | 0 | 715.8 | 743.4 | 1.8% |
| 14 | mamba2 | 0 | 833.5 | 856.1 | 2.1% |
| 15 | mamba2 | 0 | 832.5 | 857.1 | 2.1% |
| 16 | attention | 8192 | 886.8 | 908.3 | 2.3% |
| 17 | mamba2 | 0 | 716.8 | 744.6 | 1.8% |
| 18 | mamba2 | 0 | 833.5 | 856.2 | 2.1% |
| 19 | mamba2 | 0 | 832.5 | 854.1 | 2.1% |
| 20 | attention | 8192 | 887.8 | 909.3 | 2.3% |
| 21 | mamba2 | 0 | 717.8 | 746.6 | 1.8% |
| 22 | mamba2 | 0 | 832.5 | 854.1 | 2.1% |
| 23 | mamba2 | 0 | 831.5 | 859.1 | 2.1% |
| 24 | attention | 8192 | 887.8 | 907.3 | 2.3% |
| 25 | mamba2 | 0 | 716.8 | 743.4 | 1.8% |
| 26 | mamba2 | 0 | 833.5 | 857.1 | 2.1% |
| 27 | mamba2 | 0 | 831.5 | 854.0 | 2.1% |
| 28 | attention | 8192 | 888.8 | 908.4 | 2.3% |
| 29 | mamba2 | 0 | 714.8 | 744.4 | 1.8% |
| 30 | mamba2 | 0 | 834.6 | 857.1 | 2.1% |
| 31 | mamba2 | 0 | 832.5 | 853.1 | 2.1% |
| 32 | attention | 8192 | 887.8 | 907.4 | 2.3% |
| 33 | mamba2 | 0 | 717.8 | 744.4 | 1.8% |
| 34 | mamba2 | 0 | 832.5 | 856.1 | 2.1% |
| 35 | mamba2 | 0 | 831.5 | 853.0 | 2.1% |
| 36 | attention | 8192 | 887.8 | 906.2 | 2.3% |
| 37 | mamba2 | 0 | 716.8 | 744.4 | 1.8% |
| 38 | mamba2 | 0 | 832.5 | 855.0 | 2.1% |
| 39 | mamba2 | 0 | 832.5 | 857.2 | 2.1% |
| 40 | attention | 8192 | 888.8 | 907.3 | 2.3% |
| 41 | mamba2 | 0 | 717.8 | 752.6 | 1.8% |
| 42 | mamba2 | 0 | 833.5 | 855.1 | 2.1% |
| 43 | mamba2 | 0 | 832.5 | 855.0 | 2.1% |
| 44 | attention | 8192 | 886.8 | 906.2 | 2.3% |
| 45 | mamba2 | 0 | 717.8 | 745.5 | 1.8% |

## Method notes (the gotchas that bit, for reproducibility)

- **Hooks don't fire under CUDA-graph replay** - only during capture. Per-layer attribution *requires* fully-eager decode: `can_use_cudagraphs -> False` **and** `disable_torch_compile=True` (compile also hides Python hooks). This is why there are two passes.
- **Eager GPU time ≠ compute time.** CUDA events span GPU stalls, so a launch-bound eager step reads ~4× its true compute (39 vs 9 ms). The eager-vs-graph GPU gap is the launch-overhead measurement; don't read `wall − eager_GPU` as the overhead (that gives a misleadingly tiny 1.4 ms - the bubbles are *inside* the eager GPU number).
- Decode-only filtering by `inference_params.seqlen_offset > 0` (prefill events carry offset 0); first 20 decode steps dropped (KV cache short, conv state filling).
- Not done here (by design): op-level profiling inside a layer (next step, scoped to the mamba2 bucket), NCU, and timing `selective_state_update` separately.
