import argparse
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


def load_model():
    torch.manual_seed(SEED)
    return Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)


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
    print(f"CONTEXT={context} frames, TAIL={tail} samples, chunk={CHUNK_FRAMES} frames")
    conditioning = standard_conditioning(model)

    t_go = time.perf_counter()
    frame_gen = stream_frames(model, conditioning, max_new_tokens=MAX_NEW_TOKENS)
    vs = {}
    chunks = []
    for i, wav_chunk in enumerate(stream_and_vocode(frame_gen, model, CHUNK_FRAMES, context, tail, session=vs)):
        if i == 0:
            print(f"TTFA (single-shot; NOT yet p50/p90 over >=100 runs): {(time.perf_counter() - t_go) * 1000:.1f} ms")
        chunks.append(wav_chunk.cpu())

    total = time.perf_counter() - t_go
    full = torch.cat(chunks, dim=-1)
    dur = full.shape[-1] / model.autoencoder.sampling_rate
    print(f"frames={vs['n_frames']}  chunks={len(chunks)}  audio={dur:.3f}s  wall={total:.3f}s")
    print(f"end-to-end streaming RTF (generate+vocode, distinct from the decode-only 1.15x): {dur / total:.2f}x")
    torchaudio.save(args.out, full[0], model.autoencoder.sampling_rate)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 streaming pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("check-decode")
    sub.add_parser("check-vocode")
    e = sub.add_parser("e2e")
    e.add_argument("--out", default="streaming_e2e.wav")
    args = ap.parse_args()
    {
        "probe": cmd_probe,
        "check-decode": cmd_check_decode,
        "check-vocode": cmd_check_vocode,
        "e2e": cmd_e2e,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
