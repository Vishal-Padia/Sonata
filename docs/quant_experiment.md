# Weight-quantization quality experiment (Phase 1 go/no-go)

**Verdict: sub-16-bit weight-only quantization of the Zonos projections is a quality no-go.** This closes the largest Phase 1 lever and re-scopes the phase (see end).

## Why we tested this

Op-level profiling (`docs/baseline_op_level.md`) found that ~85% of a Mamba2 decode step is the two projection GEMVs, and they are bound by **reading the weight bytes**, not by math. The SSM scan is ~7% and already at the memory roofline; launch overhead is already removed by the shipped CUDA graph. So the only large remaining lever was **shrink the weight bytes** → quantize the projections. Discipline: measure the quality ceiling with fake-quant (quantize→dequantize, matmul still bf16, **zero** performance work) before writing any fast kernel.

## Methodology (and the dead end we corrected)

A first attempt compared full generated waveforms (frame-aligned mel) across configs. It was **uninterpretable**: the no-quant control scored RED and int8 scored *better* than clean — the metric was dominated by autoregressive rollout divergence (stochastic sampling + ULP-level non-determinism flip a token, and the rest of the sequence diverges into different-but-valid audio), not by quantization error.

Fix (`scripts/quant_logit_probe.py`): **greedy decode** (`run_reference.py --greedy`, temperature→argmax) to remove sampling noise, then **teacher forcing** — greedy-decode once to fix a token tape `S`, then replay `S` through the clean model and each quantized model, intercepting only `sample_from_logits` so both passes use identical `generate()` machinery and differ *only* in weights. The sequence never diverges, so the comparison is pure numerical effect.

Note: a bit-exact greedy *golden* isn't achievable on this box (two greedy runs gave 115712 vs 116736 samples — argmax exposes the forward's ULP non-determinism, near-tied tokens flip, rollout diverges). It doesn't matter: the in-process **clean-vs-clean self-check is exactly 0.000 flips / 0 MSE / 0 KL**, a pristine noise floor. Every number below is pure quantization effect.

Pass bar: **cb0 token-flip ≤ ~1–2%** (the coarse codebook drives the frame); confirmed by ear.

## Results

| config | cb0 flip% | all-cb flip% | frame-any% | logit MSE | KL(c‖q) |
|---|---|---|---|---|---|
| clean (self-check) | 0.000 | 0.000 | 0.000 | 0 | 0 |
| int8 per-ch, both | 5.06 | 7.22 | 46.8 | 1.6e-2 | 0.004 |
| int8 per-ch, in_proj | 6.75 | 6.56 | 46.4 | 1.4e-2 | 0.004 |
| int8 per-ch, out_proj | 7.17 | 6.66 | 46.0 | 1.3e-2 | 0.003 |
| int8 group=64, both | 7.60 | 6.94 | 48.5 | 1.3e-2 | 0.003 |
| int4 per-ch, both | 16.88 | 28.46 | 94.9 | 4.4e-1 | 0.088 |
| int4 group=64, both | 13.08 | 20.11 | 85.7 | 1.9e-1 | 0.041 |

## Reading of the result

- **int8 fails by a wide margin.** cb0 flip ~5–7% vs a ~1–2% bar; ~47% of frames have ≥1 codebook token flipped (~7%/codebook compounded over 9 books). Audibly "non-human" — the model's decisions change on nearly half the frames.
- **Group-wise doesn't rescue it** (7.6% ≈ per-channel). Kills the outlier hypothesis — the sensitivity is intrinsic, consistent with the measured ~1.1% weight error, not driven by a few outliers.
- **Not a localizable hot-spot.** `out_proj` alone — a plain readout, 1.15% weight error, no SSM parameters — already flips ~7% of cb0. So protecting the dt/B/C rows of `in_proj` won't help; a deep 34-layer × 9-codebook AR stack amplifies ~1% weight error too much.
- **int4 is catastrophic** (13–17% cb0, 85–95% frame-any). Definitively out.

## Why no other sub-16-bit format reopens this

int8 per-channel is the **best 1-byte format** for these (roughly Gaussian per-channel) weights: ~1% relative error. fp8 e4m3 has 3 mantissa bits → ~6% relative error per element; its only edge is dynamic range, which per-channel int8 already covers via per-channel scales. So fp8 would flip *more*, not fewer — it can't beat the int8 that already failed. There is nothing between 1 byte (int8, fails) and 2 bytes (bf16) that helps. **The "shrink the weights" lever is closed for this model.**

## Phase 1 re-scope (consequence)

The "2–3× decode speedup via kernels" headline is not reachable on this model/hardware: scan fusion is ~7%, launch overhead is already graphed, and weight quant kills quality. Quality-preserving levers that remain:

1. **bf16 roofline skinny-GEMV** — projections run above the bandwidth floor (73.6 µs vs ~58 µs ideal); a well-scheduled memory-bound GEMV closes that gap, quality-free. Apply to Mamba2 projections **and** attention MLP GEMVs (~41% of true compute). Realistic ceiling ~10–15% of the step.
2. **The streaming engine as the headline** — RTF is already ~1.15× > realtime at batch 1 on the A10G, so the contribution is incremental frame-by-frame decode + streaming DAC vocoder → a measured **TTFA** where none was reported (the long-open feature request). Does not depend on a big per-step speedup.

The decision on which to build first is deferred. This negative result + its methodology is itself a primary evidence artifact: profile → find the real bottleneck → rigorously test the obvious fix → prove it doesn't work → pivot.

## Reproduce

- `scripts/fake_quant.py` — `apply_fake_quant` (per-channel / group-wise int8/int4), importable without the generation pipeline.
- `scripts/run_reference.py --greedy` — deterministic (argmax) generation via shared `apply_fake_quant`.
- `scripts/quant_logit_probe.py` — teacher-forced logit/flip probe; clean-vs-clean self-check must read 0.000/0/0 before trusting any row.
