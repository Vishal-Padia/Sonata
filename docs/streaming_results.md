# Streaming results (Phase 1 / M3)

Reproduce: `python scripts/run_streaming.py bench` (A10G sm86, seed 421, text "The kernels run on time, frame by frame.", `MAX_NEW_TOKENS=256`, cfg 2.0). Warmup once (prime cuBLAS/cuDNN + graph), then per chunk: **100 early-stop runs** for TTFA (p50/p90) and **6 full runs** for steady-state inter-chunk per-frame latency + RTF.

> **Headline:** streaming Zonos now produces a **measured TTFA of 584 ms (p50) / 588 ms (p90)**
> at the knee (chunk=16), where none existed before. Steady-state **sustains real time**
> (per-frame 11.2 ms < the 11.61 ms budget) from chunk 16 up. Streaming is numerically
> **exact** vs one-shot decode (`check-vocode` GREEN, 67.5 dB).

## Delay-pattern floor (pinned)

`NQ = 9` codebooks; codebook `k` is delayed by `k+1` (`codebook_pattern.py`). True frame 0 needs delayed position 9, of which position 1 is filled by prefill → **first complete frame after 8 decode steps ≈ 93 ms**. This is a hard floor no kernel can remove. Note the *chunking* floor is higher: `tail_frames = ceil(5254/512) = 11`, and `stream_and_vocode` requires `chunk > tail_frames`, so the **minimum feasible chunk is 12**, not the 8-step delay floor.

## TTFA vs chunk-size sweep

| chunk | TTFA p50 (ms) | TTFA p90 (ms) | steady per-frame p50 (ms) | per-frame p90 (ms) | RTF | sustains (p90 < 11.61)? |
|---|---|---|---|---|---|---|
| 12 | 545.5 | 547.3 | 16.57 | 16.67 | 0.64 | no |
| 14 | 565.1 | 568.0 | 12.04 | 12.16 | 0.84 | no |
| **16** | **584.1** | **588.1** | **11.17** | **11.22** | 0.89 | **yes** |
| 18 | 602.5 | 606.8 | 10.81 | 10.84 | 0.92 | yes |
| 20 | 624.7 | 628.1 | 10.62 | 10.69 | 0.93 | yes |

TTFA variance is negligible (p50 ≈ p90), so these are solid.

## What the curve says (the real M3 finding)

The two columns move in opposite directions, and that tradeoff *is* the story:

- **Smaller chunk → lower TTFA** (first audio sooner) but **higher per-frame latency**. At chunk=12 only `12 − tail_frames(11) = 1` frame is committed per chunk, yet each chunk still re-decodes `CONTEXT=10` warm-up frames + the window — so the overlap-save recompute overhead, amortized over ~1 committed frame, balloons per-frame latency to 16.6 ms (**over budget → can't keep up**).
- **Larger chunk → sustains real time** (more committed frames absorb the fixed per-chunk overhead: chunk=20 commits 9 frames/chunk → per-frame drops to generate-bound ~10.6 ms) but **higher TTFA**.
- **Knee = chunk 16**: the lowest TTFA (584 ms) that still sustains (per-frame p90 11.22 < 11.61).

This *refines* the earlier "vocode is 12× realtime, overlap tax negligible" e2e finding: the overlap tax is negligible **only when `committed_frames/chunk` is large**. Near the chunk floor it dominates.

## Steady-state vs end-to-end RTF (why "sustains" but RTF < 1)

`per-frame` is steady-state inter-chunk latency (excludes startup); `RTF` is audio/wall over the whole 2.6 s clip **including** the ~584 ms TTFA. So the knee sustains ongoing playback (steady per-frame 11.17 ms < 11.61 ms → steady RTF ≈ 11.61/11.17 ≈ **1.04×**), while the short-clip RTF (0.89×) is dragged down by the one-time TTFA. For long-form audio, end-to-end RTF → steady-state. The margin is thin (~4%), consistent with the backbone being ~1.02× and memory-bound (Phase 0) — on A10G this is *right at* the real-time edge.

## Attribution (from `e2e` split, warm)

generate-only ≈ **1.02×**, vocode-only ≈ **12.4×**. The pipeline is **backbone-bound**, not vocoder-bound; the 0.9× end-to-end is serial interleaving (generate + vocode back-to-back on one stream) plus TTFA startup.

## Exactness gate (banked)

`run_streaming.py check-vocode` compares `chunked_decode` **and** `stream_and_vocode` against one-shot
`autoencoder.decode()` on the *same* codes → both **GREEN, 67.5 dB**. This is the matched-reference gate: it isolates chunking from token-sequence divergence, so it's the correct M2/M3 fidelity check. (Comparing the streamed wav to `baseline.wav` is meaningless — different stochastic rollouts diverge; that yields RED by construction, not from any chunking error.)

## Cheap TTFA/throughput levers this exposes (future, not Phase 1)

- **Decouple first-chunk from steady chunk**: emit a small first chunk (12, ~545 ms TTFA) then switch to a larger steady chunk (20, sustains) — gets both low TTFA and sustained throughput.
- **Amortize CUDA-graph capture**: TTFA currently includes a per-run graph recapture (Zonos resets the graph each generate); a persistent-graph server path would shave it further.
- **Overlap vocode with generate** (async/side stream): vocode is 12× realtime and currently serializes behind generate; hiding it lifts end-to-end RTF toward the generate-bound 1.02× ceiling.
