import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torchaudio

# monkey patch torchaudio to make it work with 2.8.0
_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *args, **kwargs):
        outer_device = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if outer_device.type != "cpu":
            self.to(outer_device)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

from zonos.conditioning import make_cond_dict
from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

SEED = 421
TEXT = (
    "The kernels run on time, frame by frame, from rest, on time, with no wasted "
    "motion. Every step pays its budget and the audio arrives the moment it is "
    "owed, steady and unhurried, all the way to the end of the line."
)
WARMUP_STEPS = 50
MEASURE_STEPS = 540 # drop first 20 -> ~520 clean decode steps
DROP_FIRST = 20
CFG_SCALE = 2.0


def build_model_and_conditioning():
    """
    Build the model and conditioning for the experiment.
    """
    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    model.eval()
    wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text=TEXT, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond)
    return model, conditioning


def _seqlen_offset(args, kwargs):
    """
    Get the seqlen offset from the arguments.
    """
    for a in list(args) + list(kwargs.values()):
        if hasattr(a, "seqlen_offset"):
            return a.seqlen_offset
    return -1


class EventTimer:
    """CUDA-event pairs keyed by name; pairs are read after one synchronize().

    Hook closures index self.events[key] at call time (NOT at install time) so
    that reset() can clear the underlying lists in place between warmup/measure.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.events = defaultdict(list) # key -> [[start, end, offset], ...]

    def reset(self):
        for v in self.events.values():
            v.clear()

    def install_layer_hooks(self):
        """
        Install hooks for the layers and the backbone.
        """
        for idx, layer in enumerate(self.model.backbone.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre(idx), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._post(idx), with_kwargs=True))
        bb = self.model.backbone
        self.handles.append(bb.register_forward_pre_hook(self._pre("bb"), with_kwargs=True))
        self.handles.append(bb.register_forward_hook(self._post("bb"), with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, key):
        def hook(module, args, kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events[key].append([start, end, _seqlen_offset(args, kwargs)])
        return hook

    def _post(self, key):
        def hook(module, args, kwargs, output):
            self.events[key][-1][1].record()
        return hook

    def wrap_decode(self):
        """
        Wrap _decode_one_token with a tight event pair (the 'step' bucket).

        generate() does torch.compile(self._decode_one_token, disable=...) when
        disabled it returns the bound method as-is, so patching here is picked up.
        Brackets backbone + embed/heads/cfg (everything but sampling).
        """
        orig = self.model._decode_one_token

        def wrapped(input_ids, inference_params, *a, **k):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = orig(input_ids, inference_params, *a, **k)
            end.record()
            self.events["step"].append([start, end, inference_params.seqlen_offset])
            return out

        self.model._decode_one_token = wrapped
        self._orig_decode = orig

    def unwrap_decode(self):
        self.model._decode_one_token = self._orig_decode


def elapsed(triples, drop_first=DROP_FIRST):
    """ms list for decode-only (offset>0) events, dropping the first `drop_first`
    decode steps (by smallest seqlen_offset)."""
    dec = [(off, s.elapsed_time(e)) for s, e, off in triples if off > 0]
    if not dec:
        return []
    offsets = sorted({off for off, _ in dec})
    cutoff = offsets[drop_first] if len(offsets) > drop_first else offsets[-1]
    return [ms for off, ms in dec if off >= cutoff]


def p(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def stop_cb(target):
    def cb(frame, step, max_steps):
        return step < target
    return cb


def timed_generate(model, conditioning, steps, **gen_kwargs):
    """Run generate for `steps` decode steps; return wall seconds (GPU-synced)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(conditioning, max_new_tokens=steps + 5, cfg_scale=CFG_SCALE,
                   progress_bar=False, callback=stop_cb(steps), **gen_kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_eager(model, conditioning):
    orig_cug = model.can_use_cudagraphs
    model.can_use_cudagraphs = lambda: False # force eager so hooks fire
    timer = EventTimer(model)
    timer.install_layer_hooks()
    timer.wrap_decode()
    try:
        timed_generate(model, conditioning, WARMUP_STEPS, disable_torch_compile=True)
        timer.reset()
        wall = timed_generate(model, conditioning, MEASURE_STEPS, disable_torch_compile=True)
    finally:
        timer.remove()
        timer.unwrap_decode()
        model.can_use_cudagraphs = orig_cug
    return timer, wall


def run_cudagraph(model, conditioning):
    """Stock graph path. Per-step GPU time via a wrapper around _decode_one_token
    (outside the capture region), plus wall time."""
    timer = EventTimer(model)
    timer.wrap_decode()
    try:
        timed_generate(model, conditioning, WARMUP_STEPS)
        timer.reset()
        wall = timed_generate(model, conditioning, MEASURE_STEPS)
    finally:
        timer.unwrap_decode()
    return elapsed(timer.events["step"]), wall


def main():
    gpu = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"# GPU: {gpu} (sm{cc[0]}{cc[1]})  torch {torch.__version__}")

    model, conditioning = build_model_and_conditioning()
    attn_idx = set(model.config.backbone.attn_layer_idx)
    n_layer = len(model.backbone.layers)

    timer, eager_wall = run_eager(model, conditioning)
    cg_step_gpu, cg_wall = run_cudagraph(model, conditioning)

    # --- eager aggregation (GPU time) ---
    step_gpu = elapsed(timer.events["step"])
    bb = elapsed(timer.events["bb"])
    layer_ms = {i: elapsed(timer.events[i]) for i in range(n_layer)}
    n_dec = len(step_gpu)

    step_p50 = p(step_gpu, 50) # eager decode GPU time per step
    layer_p50 = {i: p(layer_ms[i], 50) for i in range(n_layer)}
    layer_p90 = {i: p(layer_ms[i], 90) for i in range(n_layer)}
    sum_layer_p50 = sum(layer_p50.values())
    bb_p50 = p(bb, 50)

    mamba_idx = [i for i in range(n_layer) if i not in attn_idx]
    attn_layers = [i for i in range(n_layer) if i in attn_idx]
    mamba_sum_p50 = sum(layer_p50[i] for i in mamba_idx)
    attn_sum_p50 = sum(layer_p50[i] for i in attn_layers)

    eager_wall_per = eager_wall / MEASURE_STEPS * 1000.0 # ms
    cg_wall_per = cg_wall / MEASURE_STEPS * 1000.0 # ms
    cg_gpu_p50 = p(cg_step_gpu, 50)
    cg_gpu_p90 = p(cg_step_gpu, 90)
    budget = 1000.0 / (44100.0 / 512.0)
    rtf = budget / cg_wall_per

    def pct(x):
        return 100 * x / step_p50 if step_p50 else float("nan")

    print(f"\n# decode steps measured: eager={n_dec} cudagraph={len(cg_step_gpu)}")
    print("\n================ HEADLINE (production = CUDA-graph) ================")
    print(f"Steady-state per-step decode latency on {gpu}:")
    print(f"  CUDA-graph wall/step (full production step incl. sampling): {cg_wall_per:.3f} ms")
    print(f"  CUDA-graph GPU/step (captured _compute_logits region): p50 {cg_gpu_p50:.3f}  p90 {cg_gpu_p90:.3f} ms")
    print(f"  Per-frame budget @ 1x RTF = {budget:.2f} ms  ->  RTF = {budget:.2f}/{cg_wall_per:.3f} = {rtf:.2f}x"
          f"  (backbone decode only, excludes vocode)")
    print("\n  Launch-bound picture (decode is launch-bound at batch 1; the shipped graph fixes it):")
    print(f"    eager GPU/step  = {step_p50:.3f} ms  (incl. inter-kernel idle bubbles; events span GPU stalls)")
    print(f"    eager wall/step = {eager_wall_per:.3f} ms")
    print(f"    graph GPU/step  = {cg_gpu_p50:.3f} ms  -> graph removes ~{step_p50 - cg_gpu_p50:.1f} ms of launch "
          f"bubbles ({100*(step_p50 - cg_gpu_p50)/step_p50:.0f}% of eager GPU time)")
    print(f"    graph wall/step = {cg_wall_per:.3f} ms  (residual {cg_wall_per - cg_gpu_p50:.2f} ms = sampling + Python loop)")
    print(f"    => true per-step compute is memory-bound at ~{cg_gpu_p50:.1f} ms; launch overhead is ALREADY removed by the existing graph")

    print("\n================ PER-LAYER (eager GPU time, microseconds) ================")
    print(f"{'idx':>3} {'type':>9} {'d_int':>6} {'p50us':>9} {'p90us':>9} {'%gpustep':>9}")
    for i in range(n_layer):
        typ = "attention" if i in attn_idx else "mamba2"
        d_int = 8192 if i in attn_idx else 0
        print(f"{i:>3} {typ:>9} {d_int:>6} {layer_p50[i]*1000:>9.1f} {layer_p90[i]*1000:>9.1f} {pct(layer_p50[i]):>8.1f}%")

    print("\n================ BUCKETS (eager GPU time) ================")
    print(f"mamba2  ({len(mamba_idx)} layers): sum p50 = {mamba_sum_p50:.3f} ms  "
          f"mean = {mamba_sum_p50/len(mamba_idx)*1000:.1f} us  ({pct(mamba_sum_p50):.1f}% of GPU step)")
    print(f"attention ({len(attn_layers)} layers): sum p50 = {attn_sum_p50:.3f} ms  "
          f"mean = {attn_sum_p50/len(attn_layers)*1000:.1f} us  ({pct(attn_sum_p50):.1f}% of GPU step)")
    print(f"backbone GPU time p50         = {bb_p50:.3f} ms  ({pct(bb_p50):.1f}% of GPU step)")
    print(f"backbone overhead (bb - Sigma)= {bb_p50 - sum_layer_p50:.3f} ms  ({pct(bb_p50 - sum_layer_p50):.1f}%)")
    print(f"embed+heads+cfg (step - bb)   = {step_p50 - bb_p50:.3f} ms  ({pct(step_p50 - bb_p50):.1f}%)")
    print(f"decode GPU step p50           = {step_p50:.3f} ms  (100% of GPU step)  p90 = {p(step_gpu,90):.3f}")


if __name__ == "__main__":
    sys.exit(main())
