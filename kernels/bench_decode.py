"""Phase B/C: does the GEMV win survive Zonos's CUDA graph, and does it move the
decode-step latency? Measures per-step GPU time under the stock CUDA-graph path,
stock vs Mamba2-projections-patched, on the same model.

Two things this proves at once:
  - integration: the patched cute launch is captured/replayed correctly inside
    Zonos's decode graph (a garbage-fast patched step or degenerate codes would
    mean the kernel was NOT captured -- empty-graph replay).
  - speed: the real end-to-end decode-step delta, vs the ~10% projected from the
    isolated kernel bench.
"""
import statistics
import sys
import time
from pathlib import Path

import torch
import torchaudio

_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *a, **k):
        d = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*a, **k)
        if d.type != "cpu":
            self.to(d)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from patch import patch_zonos_projections

SEED = 421
BUDGET_MS = 1000.0 / (44100.0 / 512.0)  # 11.61 ms/frame


def build():
    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
    spk = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text="The kernels run on time, frame by frame.", speaker=spk, language="en-us")
    return model, model.prepare_conditioning(cond)


def _stop(n):
    def cb(frame, step, max_steps):
        return step < n
    return cb


@torch.inference_mode()
def measure_cg(model, conditioning, warmup=50, steps=280):
    """Per-step GPU time under the stock CUDA-graph decode path. Wraps
    _decode_one_token with a tight event pair (outside capture) -> times the graph
    replay of one step."""
    evs = []
    orig = model._decode_one_token

    def wrapped(input_ids, ip, cfg, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig(input_ids, ip, cfg, *a, **k)
        e.record()
        evs.append((s, e))
        return out

    model._decode_one_token = wrapped
    try:
        model.generate(conditioning, max_new_tokens=warmup + 5, progress_bar=False, callback=_stop(warmup))
        evs.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(conditioning, max_new_tokens=steps + 5, progress_bar=False, callback=_stop(steps))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    finally:
        model._decode_one_token = orig
    n = len(evs)
    per = [s.elapsed_time(e) for s, e in evs][20:]  # ms, drop transient/recapture
    p50 = statistics.median(per)
    p90 = statistics.quantiles(per, n=100, method="inclusive")[89] if len(per) > 1 else per[0]
    return p50, p90, wall / n * 1000.0  # ms


@torch.inference_mode()
def sane_codes(model, conditioning):
    """Generate a short clip under the (patched) graph path; codes must be finite
    and non-degenerate, else the graph replayed without the cute kernel."""
    codes = model.generate(conditioning, max_new_tokens=60, progress_bar=False, callback=_stop(50))
    u = int(codes.unique().numel())
    return u


def main():
    model, cond = build()

    print("measuring stock (cuBLAS projections, CUDA graph) ...", flush=True)
    s50, s90, swall = measure_cg(model, cond)

    n = patch_zonos_projections(model, include_attn=True)
    print(f"patched {n} projections (Mamba2 + attention); measuring under CUDA graph ...", flush=True)
    p50, p90, pwall = measure_cg(model, cond)
    uniq = sane_codes(model, cond)

    print("\n" + "=" * 68)
    print(f"{'':16} {'GPU p50':>9} {'GPU p90':>9} {'wall/step':>10} {'RTF':>6}")
    print(f"{'stock (cuBLAS)':16} {s50:>8.3f} {s90:>8.3f} {swall:>9.3f}  {BUDGET_MS/swall:>5.2f}x")
    print(f"{'cute projections':16} {p50:>8.3f} {p90:>8.3f} {pwall:>9.3f}  {BUDGET_MS/pwall:>5.2f}x")
    print("-" * 68)
    print(f"decode-step GPU speedup: {s50/p50:.3f}x   ({100*(s50-p50)/s50:+.1f}% latency)")
    print(f"graph-capture sanity: patched clip has {uniq} unique codes "
          f"({'OK - kernel captured' if uniq > 5 else 'DEGENERATE - empty-graph replay!'})")
    print("=" * 68)
    # A garbage-fast step (e.g. < 0.6x) with few unique codes = the cute launch was
    # not captured; treat any >1.3x step speedup as suspicious, not a real win.


if __name__ == "__main__":
    main()
