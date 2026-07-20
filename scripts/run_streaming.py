import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling scripts

# torchaudio>=2.7 precomputes MelSpectrogram filterbanks on the ambient torch.device;
# under a CUDA device context that trips a device mismatch. Build on CPU, move back.
_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *args, **kwargs):
        outer_device = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if outer_device.type != "cpu":
            self.to(outer_device)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

from zonos.codebook_pattern import revert_delay_pattern
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

from streaming import chunked_decode, measure_receptive_field, stream_and_vocode, stream_frames
from audio_fidelity import Verdict, classify, mel_l1_db, waveform_metrics

SEED = 421
TEXT = "The kernels run on time, frame by frame."
MAX_NEW_TOKENS = 256
CHUNK_FRAMES = 20  # ~232 ms of new audio/chunk; must clear TAIL (~10.3 frames) with margin

# bench: sweep chunk sizes. Min feasible chunk = tail_frames+1 = 12 (stream_and_vocode
# asserts chunk > tail_frames, and tail=5254 -> ceil/512 = 11), NOT the 8-step delay floor.
SWEEP_CHUNKS = [12, 14, 16, 18, 20]
TTFA_RUNS = 100   # early-stop runs per chunk (stop at first chunk) -> TTFA p50/p90
FULL_RUNS = 6     # full runs per chunk -> inter-chunk latency + RTF
FRAME_MS = 1000.0 / (44100.0 / 512.0)  # 11.61 ms per-frame realtime budget


_PATCH_GEMV = False    # set by --patch: swap projections to the CuTe GEMV after load
_COMPILE_VOC = False   # set by --compile-vocoder: torch.compile the DAC decoder (2.5x eager, free)


def load_model():
    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    if _PATCH_GEMV:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
        from patch import patch_zonos_projections
        n = patch_zonos_projections(model, include_attn=True)
        print(f"[--patch] swapped {n} projections to CuTe GEMV", flush=True)
    if _COMPILE_VOC:
        # dynamic=True: streaming vocode windows vary in length; avoid per-shape recompiles.
        model.autoencoder.dac.decoder = torch.compile(model.autoencoder.dac.decoder, dynamic=True)
        print("[--compile-vocoder] torch.compile'd the DAC decoder", flush=True)
    return model


def standard_conditioning(model):
    wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text=TEXT, speaker=speaker, language="en-us")
    return model.prepare_conditioning(cond)


def _fixed_codes(model, session=None):
    """Drive stream_frames to completion and return the run's cleaned codes
    [1, 9, N] -- a fixed tensor for the deterministic vocoder gates."""
    session = {} if session is None else session
    conditioning = standard_conditioning(model)
    for _ in stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS, session=session):
        pass
    codes = revert_delay_pattern(session["delayed_codes"])[..., : session["n_frames_yielded"]]
    return codes.masked_fill(codes >= 1024, 0)  # the vocoder-time cleanup generate() does at the end


def _gate(reference, candidate, model, label):
    ref = reference.reshape(-1).float().cpu()
    cand = candidate.reshape(-1).float().cpu()
    n = min(ref.numel(), cand.numel())
    if ref.numel() != cand.numel():
        print(f"  [WARN] {label}: length ref={ref.numel()} cand={cand.numel()}, truncating to {n}")
    ref, cand = ref[:n], cand[:n]
    peak, snr = waveform_metrics(ref, cand)
    mel = mel_l1_db(ref, cand, model.autoencoder.sampling_rate)
    verdict = classify(snr, mel)
    print(f"  {label:<26} SNR={snr:7.2f} dB  mel-L1={mel:.4f} dB  peak={peak:.2e}  -> {verdict.value}")
    return verdict


@torch.inference_mode()
def cmd_probe(args):
    model = load_model()
    hop = model.autoencoder.dac.config.hop_length
    context, tail = measure_receptive_field(model)
    print(f"hop_length : {hop} samples/frame")
    print(f"CONTEXT    : {context} frames ({context * hop} samples of left context)")
    print(f"TAIL       : {tail} samples ({tail / hop:.2f} frames worth)")


@torch.inference_mode()
def cmd_check_decode(args):
    model = load_model()
    conditioning = standard_conditioning(model)
    session = {}
    frames = list(stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS, session=session))
    streamed = torch.stack(frames, dim=-1)  # [batch, NQ, N]
    N = session["n_frames_yielded"]
    assert N == len(frames), f"session reports {N} frames but generator yielded {len(frames)}"
    batch = revert_delay_pattern(session["delayed_codes"])  # raw: skip >=1024 cleanup (not part of the un-stagger check)
    assert torch.equal(streamed, batch[..., :N]), "un-stagger mismatch: streaming and batch paths disagree"
    assert (streamed != -1).all(), "a frame escaped with an unwritten token"
    print(f"check-decode PASS: {N} frames byte-identical between streaming and batch un-stagger")


@torch.inference_mode()
def cmd_check_vocode(args):
    model = load_model()
    context, tail = measure_receptive_field(model)
    print(f"measured CONTEXT={context} frames, TAIL={tail} samples, chunk={CHUNK_FRAMES} frames")

    codes = _fixed_codes(model)  # same codes -> only chunking differs; any loss is a CONTEXT/TAIL bug
    reference = model.autoencoder.decode(codes)

    # path 1: chunked_decode (batch overlap-save)
    cand1 = chunked_decode(model, codes, chunk_frames=CHUNK_FRAMES, context=context, tail=tail)
    v1 = _gate(reference, cand1, model, "chunked_decode vs one-shot")

    # path 2: stream_and_vocode (incremental -- the shipping path). Feed fixed code
    # columns so this is a pure windowing check, no generation randomness.
    frame_iter = (codes[..., t] for t in range(codes.shape[-1]))
    cand2 = torch.cat(list(stream_and_vocode(frame_iter, model, CHUNK_FRAMES, context, tail)), dim=-1)
    v2 = _gate(reference, cand2, model, "stream_and_vocode vs one-shot")

    assert v1 is not Verdict.RED and v2 is not Verdict.RED, (
        "a vocoder path diverges from one-shot decode -- CONTEXT/TAIL too small or windowing bug"
    )
    print("check-vocode PASS: both vocoder paths match one-shot decode within tolerance")


@torch.inference_mode()
def cmd_e2e(args):
    model = load_model()
    context, tail = measure_receptive_field(model)
    print(f"CONTEXT={context} frames, TAIL={tail} samples, chunk={CHUNK_FRAMES} frames", flush=True)
    conditioning = standard_conditioning(model)

    # Warmup: the FIRST backbone step pays one-time cuBLAS/cuDNN algo selection +
    # CUDA-graph capture (measure_receptive_field only warmed the DAC). Without
    # this, "TTFA" is dominated by ~7s of cold init, not real streaming latency.
    # Safe now that stream_frames guards the no-EOS overflow.
    print("warmup (priming backbone + graph; discarded)...", flush=True)
    for _ in stream_and_vocode(stream_frames(model, conditioning, max_new_tokens=64),
                               model, CHUNK_FRAMES, context, tail):
        pass
    torch.cuda.synchronize()

    # --- timed interleaved streaming run (TTFA + end-to-end streaming RTF) ---
    print("streaming (timed)...", flush=True)
    t_go = time.perf_counter()
    frame_gen = stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS)
    vs = {}
    chunks = []
    for i, wav_chunk in enumerate(stream_and_vocode(frame_gen, model, CHUNK_FRAMES, context, tail, session=vs)):
        if i == 0:
            print(f"TTFA (warm, single-shot; NOT yet p50/p90 over >=100 runs): "
                  f"{(time.perf_counter() - t_go) * 1000:.1f} ms", flush=True)
        chunks.append(wav_chunk.cpu())
    total = time.perf_counter() - t_go
    full = torch.cat(chunks, dim=-1)
    dur = full.shape[-1] / model.autoencoder.sampling_rate
    torchaudio.save(args.out, full[0], model.autoencoder.sampling_rate)

    # --- attribution: split the clock into generate vs vocode (both warm) ---
    # Is RTF < 1 the overlap-save recompute tax, or a decode regression? Time each
    # phase alone. generate-only drains stream_frames (no vocode); vocode-only
    # runs chunked_decode over the resulting codes.
    sess = {}
    t = time.perf_counter()
    for _ in stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS, session=sess):
        pass
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t
    codes = revert_delay_pattern(sess["delayed_codes"])[..., : sess["n_frames_yielded"]]
    codes = codes.masked_fill(codes >= 1024, 0)
    t = time.perf_counter()
    chunked_decode(model, codes, chunk_frames=CHUNK_FRAMES, context=context, tail=tail)
    torch.cuda.synchronize()
    t_voc = time.perf_counter() - t

    print(f"frames={vs['n_frames']}  chunks={len(chunks)}  audio={dur:.3f}s  wall={total:.3f}s", flush=True)
    print(f"end-to-end streaming RTF (generate+vocode; distinct from decode-only 1.15x): {dur / total:.2f}x", flush=True)
    print(f"  split: generate-only {t_gen:.3f}s ({dur / t_gen:.2f}x)   "
          f"vocode-only {t_voc:.3f}s ({dur / t_voc:.2f}x)", flush=True)
    print(f"wrote {args.out}", flush=True)


def _pct(xs, q):
    return statistics.quantiles(xs, n=100, method="inclusive")[q - 1] if len(xs) > 1 else xs[0]


@torch.inference_mode()
def cmd_bench(args):
    """M3: TTFA-vs-chunk sweep + p50/p90 for TTFA and steady-state inter-chunk
    latency. Finds the real minimum TTFA (the knee) instead of quoting one warm
    shot at chunk=20, and confirms steady-state stays under the per-frame budget."""
    model = load_model()
    context, tail = measure_receptive_field(model)
    hop = model.autoencoder.dac.config.hop_length
    sr = model.autoencoder.sampling_rate
    tail_frames = -(-tail // hop)
    conditioning = standard_conditioning(model)

    # Pin the delay-pattern floor off the code: frame 0 needs delayed index NQ=9,
    # of which position 1 is the prefill sample -> 8 decode steps.
    nq = model.autoencoder.num_codebooks
    delay_steps = nq - 1
    print(f"CONTEXT={context}f  TAIL={tail}smp ({tail_frames}f)  budget={FRAME_MS:.2f} ms/frame", flush=True)
    print(f"delay floor: NQ={nq} delayed positions, position 1 from prefill -> "
          f"first frame after {delay_steps} decode steps (~{delay_steps * FRAME_MS:.0f} ms); "
          f"min feasible chunk = {tail_frames + 1}", flush=True)

    # Process warmup once (prime cuBLAS/cuDNN + graph); reset graph so run 1 recaptures.
    print("warmup...", flush=True)
    for _ in stream_and_vocode(stream_frames(model, conditioning, max_new_tokens=64),
                               model, SWEEP_CHUNKS[0], context, tail):
        pass
    model._cg_graph = None
    torch.cuda.synchronize()

    def ttfa_once(chunk):
        gen = stream_and_vocode(stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS),
                                model, chunk, context, tail)
        t0 = time.perf_counter()
        next(gen) # first chunk only
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        gen.close()
        model._cg_graph = None # early-stop skipped stream_frames' reset; force recapture (stale-pointer safety)
        return dt

    def full_once(chunk):
        vs = {}
        gen = stream_and_vocode(stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS),
                                model, chunk, context, tail, session=vs)
        stamps, samples = [], []
        t0 = time.perf_counter()
        for wav_chunk in gen:
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
            samples.append(wav_chunk.shape[-1])
        total = stamps[-1] - t0
        audio = sum(samples) / sr
        # inter-chunk per-frame latency: gap between chunk k-1,k over frames committed in chunk k
        per_frame = [((stamps[k] - stamps[k - 1]) * 1000.0) / (samples[k] / hop)
                     for k in range(1, len(stamps))]
        return audio / total, per_frame  # (RTF, list of per-frame ms)

    rows = []
    for chunk in SWEEP_CHUNKS:
        ttfas = [ttfa_once(chunk) for _ in range(TTFA_RUNS)]
        rtfs, pf = [], []
        for _ in range(FULL_RUNS):
            rtf, per_frame = full_once(chunk)
            rtfs.append(rtf)
            pf.extend(per_frame)
        rows.append((chunk, _pct(ttfas, 50), _pct(ttfas, 90),
                     _pct(pf, 50), _pct(pf, 90), statistics.median(rtfs)))
        print(f"  chunk={chunk:2d}: TTFA p50={rows[-1][1]:6.1f} p90={rows[-1][2]:6.1f} ms | "
              f"per-frame p50={rows[-1][3]:5.2f} p90={rows[-1][4]:5.2f} ms (budget {FRAME_MS:.2f}) | "
              f"RTF={rows[-1][5]:.2f}x", flush=True)

    print(f"\n{'chunk':>5} {'TTFA_p50':>9} {'TTFA_p90':>9} {'pframe_p50':>11} {'pframe_p90':>11} {'RTF':>6} {'sustains':>9}")
    for chunk, t50, t90, f50, f90, rtf in rows:
        print(f"{chunk:>5} {t50:>9.1f} {t90:>9.1f} {f50:>11.2f} {f90:>11.2f} {rtf:>6.2f} "
              f"{'yes' if f90 < FRAME_MS else 'no':>9}")
    sustaining = [r for r in rows if r[4] < FRAME_MS]  # p90 per-frame under budget
    knee = min(sustaining, key=lambda r: r[1]) if sustaining else min(rows, key=lambda r: r[1])
    print(f"\nknee: chunk={knee[0]}  TTFA p50={knee[1]:.1f} ms / p90={knee[2]:.1f} ms  "
          f"(lowest TTFA that sustains p90 per-frame < {FRAME_MS:.2f} ms)", flush=True)


def main():
    global _PATCH_GEMV, _COMPILE_VOC
    ap = argparse.ArgumentParser(description="Phase 1 streaming pipeline")
    ap.add_argument("--patch", action="store_true",
                    help="swap projections to the CuTe GEMV (kernels/patch.py) after load")
    ap.add_argument("--compile-vocoder", action="store_true",
                    help="torch.compile the DAC decoder (2.5x eager decode, free)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("check-decode")
    sub.add_parser("check-vocode")
    e = sub.add_parser("e2e")
    e.add_argument("--out", default="streaming_e2e.wav")
    sub.add_parser("bench")
    args = ap.parse_args()
    _PATCH_GEMV = args.patch
    _COMPILE_VOC = args.compile_vocoder
    {
        "probe": cmd_probe,
        "check-decode": cmd_check_decode,
        "check-vocode": cmd_check_vocode,
        "e2e": cmd_e2e,
        "bench": cmd_bench,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
