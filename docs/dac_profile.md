# DAC decoder per-op profile (Phase 2)

Reproduce: `python scripts/dac_profile.py`. A10G sm86, one-shot `autoencoder.decode()` of a 228-frame (~2.6 s) clip, 50 iters, torch.profiler with per-block `record_function` labels. Architecture map (conv stack, causal analysis, receptive field) is in `docs/dac_architecture.md`; this is the measured other half.

> **Headline:** the **Snake activation is ~half the decode compute** (`sin`/`pow`/`mul`/`add`
> elementwise sweeps), and the cost concentrates in the **last two upsampling blocks (~70%)**,
> which run at the highest sample rates. Convolutions are only ~31%. So the compute lever is
> **fusing the Snake elementwise sweeps** — not the convs. But once we measured (see
> [Measured: what actually wins](#measured-what-actually-wins)), the hand-written fused kernel
> lost to cuDNN and `torch.compile` already fuses Snake for a free 2.5x, and none of it moves the
> streaming headline because the vocoder is ~3% of the streaming wall. Phase 2 has **no
> custom-kernel work that pays off on this hardware.**

## Where the time goes (per region)

Read the shares, not the absolutes: the region sum (60.6 ms) is inflated by `record_function` device-time spanning idle/overlap; the real number is the **35.09 ms/call** wall (RTF **75.4x**).

| region | ms (device) | share | why |
|---|---|---|---|
| conv1 (stem) | 0.40 | ~1% | latent rate, 1 timestep/frame |
| block.0 | 3.32 | ~5% | 768 ch but only 8x upsampled |
| block.1 | 11.13 | ~18% | 384 ch, 64x |
| **block.2** | **21.82** | **~36%** | 192 ch, 256x — high sample rate |
| **block.3** | **20.75** | **~34%** | 96 ch, 512x — full 44.1 kHz |
| head (snake1 + conv2) | 3.18 | ~5% | waveform rate, cheap |

The late blocks dominate because sample count grows faster than channel count shrinks: block.3 has the fewest channels (96) but the most timesteps (full rate). **Target the late blocks.**

## Where the time goes (per op)

| op | self-CUDA share | what it is |
|---|---|---|
| `mul` + `add` + `sin` + `pow` | **~53%** | **Snake** (`x + (1/α)·sin²(αx)`) + residual adds |
| `cudnn_convolution` | ~26% | the depthwise/full conv1d |
| `copy_` | ~10% | intermediate buffers |
| `cudnn_convolution_transpose` | ~6% | the upsampling |

Snake is not a minor activation to fold — it is the **single biggest cost**. `sin` and `pow` are transcendental elementwise passes over the full activation maps, 29 of them (per `docs/dac_architecture.md`), concentrated in the high-sample-rate late blocks. This is the clean argument for the fused snake+conv kernel: it removes ~half the decode's elementwise traffic.

## The streaming angle, quantified

One-shot decode of the whole clip is **75x realtime**. Our streaming overlap-save vocoder (`kernels`/`run_streaming`) measured **12x** — the 6x gap is the redundant re-decode of the CONTEXT frames on every chunk. So the cached-conv streaming DAC's payoff is concrete: recover that 6x (stream at ~75x instead of 12x) **and** make it sample-accurate vs one-shot, instead of overlap-save's approximation. Still not a latency need (12x already clears realtime), but a real efficiency + correctness win.

## Measured: what actually wins

The profile said "fuse Snake into the conv." We then built it and measured it, and it did not survive contact with the numbers. Three measurements, in order:

**1. Hand-written fused snake+conv loses to cuDNN by 15-25x.** `kernels/fused_snake_conv.py` is a naive per-output kernel (one thread per output element, recomputes Snake `out_channels*K` times per input). Benched at the real late-block shapes (`kernels/bench_snake_conv.py`):

| shape | fused | separate (snake -> cuDNN) | speedup | max_err |
|---|---|---|---|---|
| block.2 res (C=192, L=58368, K7 d1) | 62.03 ms | 2.747 ms | **0.04x** | 7.0e-2 |
| block.2 res (C=192, L=58368, K7 d9) | 67.30 ms | 2.805 ms | **0.04x** | 6.8e-2 |
| block.3 res (C=96, L=116736, K7 d1) | 31.10 ms | 2.110 ms | **0.07x** | 4.8e-2 |
| block.3 res (C=96, L=116736, K7 d9) | 32.38 ms | 2.120 ms | **0.07x** | 4.7e-2 |

Fused is **14-25x slower** and less accurate (`cute.math.sin` loses precision at large `αx`). The Snake recompute costs far more than the one HBM pass it saves, and a tiled snake-once rewrite would still have to beat cuDNN's tuned conv — a bad trade for a non-bottleneck.

**2. `torch.compile` fuses Snake for a free 2.5x** (`scripts/dac_compile.py`, one-shot decode of a 228-frame clip):

| | ms/call | RTF | speedup | max_err |
|---|---|---|---|---|
| eager | 35.10 | 75x | — | — |
| compiled (default) | 14.18 | 248x | **2.48x** | 9.8e-4 |
| compiled (max-autotune) | 13.98 | 251x | 2.51x | — |

Inductor does exactly the pointwise fusion the hand kernel was trying to do, correctly (max_err ~1e-3) and for zero effort. This is the whole Snake-fusion win, already available.

**3. But it does nothing for streaming, and hurts TTFA.** Wiring `torch.compile` into the streaming vocode path (`scripts/run_streaming.py --compile-vocoder e2e`):

| metric | stock | +compiled vocoder |
|---|---|---|
| vocode-only | 12.4x | **26.2x** (2.1x faster) |
| generate-only | 1.02x | 0.98x (unchanged) |
| streaming RTF | 0.92x | **0.91x (unmoved)** |
| TTFA | ~618 ms | **729 ms (worse)** |

The compile win on vocode is real (2.1x), but in the streaming pipeline vocode is ~0.1 s of a ~2.95 s wall (~3%), so halving it is invisible in RTF, and the compile/dynamic-shape overhead adds ~110 ms to TTFA. Same pattern as the GEMV graph-capture cost: a one-time overhead landing on the headline latency metric.

## What Phase 2 resolves to

No custom vocoder kernel pays off on this hardware. Concretely:

1. **Skip the fused snake+conv kernel.** It loses to cuDNN 15-25x, and the fusion it targets is already free via `torch.compile`.
2. **Use `torch.compile` only for batch/offline decode** (decode the whole sequence at once), where vocode dominates and the 2.5x is worth it. **Do not** compile the vocoder in the streaming path: no RTF gain, worse TTFA. `--compile-vocoder` stays as a batch-mode option.
3. **Cached-conv streaming DAC** is still a correctness/efficiency item if we want it (per-conv left-history cache + the ~5254-sample TAIL lookahead, `docs/dac_architecture.md`, replacing overlap-save's re-decode), but it is not a latency need: streaming already clears realtime at 12x vocode and the backbone dominates the wall.

Scope note that held up under measurement: the vocoder is ~3% of end-to-end wall in the streaming pipeline (decode backbone dominates), so nothing done to the vocoder moves the streaming TTFA/RTF headline. The real remaining kernel work is Phase 3 (the SSD chunked-scan prefill), which is compute-bound and on the TTFA critical path — a regime where a hand kernel can actually win.
