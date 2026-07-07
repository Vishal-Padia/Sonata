# The kernel that wasn't there: profiling my way to real-time streaming Zonos

*Or: what 7% taught me before I wrote a single kernel.*

I set out to write a fused Mamba2 decode kernel for Zonos. The plan, in writing, said 2 to 3x on the per-frame decode step. The first thing profiling told me was that the kernel I was about to write would buy me about 7%, and that the part of it everyone treats as the bottleneck was already sitting at the memory roofline. So I didn't write it. I shipped streaming instead, and that turned out to be the better story.

This is a post about a project where the kernels were the point and the honest answer was structure. If that sounds like a climbdown, it isn't. Finding out that the obvious kernel doesn't help, with numbers, before spending three weeks on it, is the job.

## The setup

[Zonos](https://github.com/Zyphra/Zonos) is Zyphra's open-weight TTS model. The hybrid checkpoint I used has a Mamba2 + attention backbone that autoregressively predicts DAC codec tokens, and a DAC decoder that turns those tokens into a 44.1 kHz waveform. It's the only open TTS model with a genuinely SSM-based backbone, and it's the closest open thing to the architecture class Cartesia's Sonic sits in: SSM backbone, neural codec, time-to-first-audio as the headline metric. It generates in batch. Run the model N times, collect all the tokens, decode the whole sequence once at the end, so the first sound doesn't exist until the last token does. There's a streaming request that's been open on the repo since launch.

My actual goal is kernels. Zonos was the substrate: a real model with a real latency-critical decode loop, a place to write a fused Mamba2 decode step and a streaming vocoder. I wrote a detailed plan up front, and the useful thing about it is how thoroughly measurement contradicted it. The profiling docs in the repo are the honest record. Where they disagree with what I set out to do, treat it as chronology: what I believed going in versus what the profiler made me believe after.

Every number below is on an A10G (sm86). Not a 4090. That matters later.

## What I thought I was optimizing

Phase 1's headline was a fused Mamba2 decode step. Take the causal-conv1d update, the selective state update, the gated norm, fuse them into one kernel, keep the recurrent state resident, touch HBM once per step instead of once per sub-op. Standard memory-bound-decode reasoning. The state tensor is the thing you're told you're traffic-bound on at batch 1.

Before writing it I built a per-block profiling harness ([`scripts/per_block_breakdown.py`](https://github.com/Vishal-Padia/Sonata/blob/main/scripts/per_block_breakdown.py)) and a per-op one ([`scripts/op_level_profile.py`](https://github.com/Vishal-Padia/Sonata/blob/main/scripts/op_level_profile.py)), because the plan also said, in bold, resolve the real cost centers before touching a kernel. That instinct paid for the whole project.

## Where the time actually goes

One decode step on the A10G is 9.00 ms of GPU time, 10.07 ms wall, against an 11.61 ms per-frame real-time budget. So the stock model already runs at about 1.15x real time at batch 1. Zyphra reports around 2x on a 4090; the A10G is just slower, and this is the real baseline I have to work against.

Two findings reorganized the plan.

First, the launch overhead was already gone. Eager, the decode step is 39 ms of GPU time. Under CUDA graph it's 9 ms, for the exact same kernels. That 30 ms was inter-kernel launch bubbles, the classic batch-1 decode problem, and Zonos already ships CUDA-graph capture that removes it. The plan had "add CUDA graphs first if overhead is high" as a headline lever. It was already pulled.

Second, and this is the one that killed the kernel: inside a Mamba2 layer the selective scan is about 7% of the compute, and it's already at the roofline. I drove one real layer under the profiler ([`docs/baseline_op_level.md`](https://github.com/Vishal-Padia/Sonata/blob/main/docs/baseline_op_level.md)) and the split is:

| op | us/step | share |
|---|---|---|
| in_proj GEMV | 73.6 | 50% |
| out_proj GEMV | 51.0 | 35% |
| selective_state_update | 6.99 | ~5% |
| causal_conv1d_update | 3.37 | ~2% |
| norm + misc | ~8 | ~8% |

The two projection matmuls are 85% of the layer, and they're slow for a boring reason. At batch 1 they're memory-bound on reading the weights, about 52 MB per layer. The selective_state_update kernel reads and writes the roughly 4 MB of recurrent state in 6.99 us, which is basically the A10G's bandwidth. There's nothing to fuse out of it. The scan I was going to spend three weeks on is 10 us of a 146 us layer, and it's already optimal.

Fusing it perfectly caps out at a 7% layer win. That's the whole thesis of Phase 1, retired by its own harness.

## The obvious fix, and why it's dead too

If 85% of the layer is reading weight bytes, the obvious move is to read fewer bytes: quantize the projection weights. int8 halves the traffic. So before writing any fast int8 kernel, I checked whether int8 even preserves the audio, because a fast kernel that sounds bad is worthless.

The first way I measured this was wrong, in a way worth showing. I quantized, regenerated audio, compared it to the baseline waveform, and everything came back RED, including the un-quantized control. The model samples, so two runs produce different but equally valid token sequences, and comparing two different rollouts frame-by-frame measures divergence, not quality. The same trap comes back for the streaming gate later, so it's worth internalizing once.

The fix is to stop comparing rollouts. I froze one greedy token sequence, teacher-forced it through the clean model and each quantized model, and counted how often the quantized model's argmax disagreed with clean ([`scripts/quant_logit_probe.py`](https://github.com/Vishal-Padia/Sonata/blob/main/scripts/quant_logit_probe.py)). In-process, clean against clean is exactly zero flips, so every number is pure quantization effect.

int8 per-channel flips 5 to 7% of the coarse-codebook tokens, and about 47% of frames end up with at least one token changed. It sounds non-human. Group-wise doesn't rescue it, which tells you the sensitivity is intrinsic and not outlier-driven, consistent with the flat ~1.1% weight error. int4 is 13 to 17%, gone. And nothing between int8 and bf16 helps: fp8 e4m3 has 3 mantissa bits, roughly 6% relative error, worse than int8's ~1% on these weights. The full table is in [`docs/quant_experiment.md`](https://github.com/Vishal-Padia/Sonata/blob/main/docs/quant_experiment.md).

So the one lever with real headroom, shrink the weight bytes, is closed on quality. That isn't a detour, it's a result: the cost that dominates this model at batch 1 is mostly irreducible.

## So, streaming

Here's the reframe that saved the phase. The model is already at 1.15x real time. It just isn't structured to stream. You don't need a faster model to fix that, you need to emit audio as it's produced. That's the open request, it's the metric the target domain actually reports, and it doesn't depend on the kernel win I couldn't get.

Streaming is two halves: hand out decoded frames the moment they're ready, and vocode them in chunks. The backbone already carries its own KV cache and SSM state across steps, so the decode loop needs no new state machinery, which is the part I was most worried about and turned out to be free. The only genuinely fiddly bit is the delay pattern.

## The delay pattern

Zonos emits 9 codebooks per frame under a delay: codebook k for a given frame lands k+1 steps late. A complete frame isn't assembled until 9 delayed positions exist, and since the first comes from prefill, the first true frame is ready 8 decode steps in, about 93 ms. That's a hard floor, no kernel removes it, and it's worth pinning to the exact integer before quoting any TTFA number.

[`scripts/streaming.py`](https://github.com/Vishal-Padia/Sonata/blob/main/scripts/streaming.py) is a near-verbatim copy of Zonos's generate loop that yields each frame the instant its last codebook is written. I checked it the deterministic way: within a single run, the frames I hand out have to be byte-identical to reverting the delay pattern on the full tensor. The `torch.stack` in the yield copies, so a handed-out frame is frozen, which means the check also proves I never emit a frame before all 9 of its codebooks are final. It passes at 228 frames.

## The vocoder, in chunks

The DAC decoder is a stack of transposed convolutions, so it has a receptive field. Decode a chunk in isolation and the edge samples are wrong, because the convs see zero-padding where the neighboring frames should be, and concatenating those gives you a click at every boundary.

I didn't guess the receptive field, I measured it. Perturb one frame, decode, see how far the change reaches: 10 frames of left context, and a right tail of 5254 samples that lacked future context when decoded. So the chunked decoder prepends the context, drops the warm-up samples, and holds back the tail to re-decode on the next window once it finally has right context. The one thing you have to get right is that holding back the tail means re-decoding those frames later, not skipping them, or you silently drop audio at every boundary.

The gate is the important part, and it's the same lesson as the quant probe: don't compare to the baseline waveform. Compare chunked decoding against one-shot decoding of the same codes, so only the chunking differs. Both the batch and the incremental path come back GREEN at 67.5 dB against one-shot ([`scripts/run_streaming.py`](https://github.com/Vishal-Padia/Sonata/blob/main/scripts/run_streaming.py), `check-vocode`). Comparing the streamed wav to the baseline gives RED, and that RED means nothing: different rollouts, same trap as before.

## Numbers

TTFA trades against throughput through the chunk size, so I swept it ([`docs/streaming_results.md`](https://github.com/Vishal-Padia/Sonata/blob/main/docs/streaming_results.md)), 100 runs each for the percentiles:

| chunk | TTFA p50 | steady per-frame p90 | keeps up? |
|---|---|---|---|
| 12 | 545 ms | 16.67 ms | no |
| 14 | 565 ms | 12.16 ms | no |
| 16 | 584 ms | 11.22 ms | yes |
| 20 | 625 ms | 10.69 ms | yes |

The two columns pull against each other, and that's the finding. A smaller chunk gives you first audio sooner but worse throughput, because with the tail at 11 frames a size-12 chunk commits only 1 new frame while still re-decoding 10 context frames, so the per-committed-frame overhead blows past the budget. A bigger chunk amortizes that overhead and settles at the generate-bound rate. The knee is chunk 16: the lowest TTFA (584 ms p50, 588 ms p90) that still holds steady-state per-frame latency under the 11.61 ms budget.

Splitting the clock: generate runs at 1.02x, vocode at 12.4x. It's backbone-bound, and the sustaining margin at the knee is about 4%. On the A10G this streams, but right at the edge, which is exactly what you'd expect from a memory-bound decode you weren't allowed to quantize.

## What this was actually about

If you came in expecting "and then the fused kernel made it 3x faster," that kernel isn't here, and the reason it isn't is the point. The scan was 7% and already at the roofline. The launch overhead was already handled. The one lever with real headroom fails the quality bar. The honest output of a kernel-focused profiling pass on this model, this round, was structure: exact, real-time streaming with a 584 ms TTFA where there was none.

The judgment is the deliverable. Profile first, find that the assumed bottleneck is 7%, prove the obvious fix doesn't work before building it, then ship the thing that actually moved the number.

The same profiling also tells me where kernels do pay off here, which is the next round. Prefill, which is the tensor-core-bound chunked scan and a genuinely different regime from decode. A better memory-bound GEMV for the projections, which is a GEMV problem and not an SSM one, and is the largest quality-free per-step win left on the table. And batch greater than 1, where the projections turn back into compute-bound GEMMs. Kernels are still the throughline. This is just where the profiling honestly pointed this time.

Frame by frame. On time, mostly.

The code, the plan, and every profiling doc referenced above live in the repo: [github.com/Vishal-Padia/Sonata](https://github.com/Vishal-Padia/Sonata).

As always, happy to chat if anything here is unclear or wrong. Just ping me on [Twitter](https://x.com/KyrieBlunders).