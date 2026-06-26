"""Teacher-forced logit-divergence probe for weight-only fake quantization.

WHY THIS EXISTS
---------------
Free-running audio comparison is uninterpretable for this model: it samples
tokens autoregressively, so any sub-ULP numerical change (a driver/cuBLAS update,
or a 1% weight quant) sends the rollout onto a *different but valid* token
sequence. Frame-aligned SNR/mel then measure sequence divergence, not quality --
which is why even the unquantized clean run fails the audio gate.

This probe removes rollout divergence entirely:
  1. Run the clean model with GREEDY decoding (argmax) to obtain a fixed token
     tape S and the clean per-step logits.
  2. Teacher-force S through each quantized model: at every step the next input
     is S[t] (clean's token), never the quantized model's own prediction. The
     sequence cannot diverge, so the per-step logit difference is the *pure*
     numerical effect of quantization.

DECISION METRIC
---------------
  - token-flip rate: fraction of steps where argmax(quant logits) != argmax(clean
    logits). This is the go/no-go number. Low (<~1-2%) on codebook 0 => int8 is
    safe, green-light the kernel. High => quant changes the model's decisions too
    much; try group-wise or fall back to the bf16 roofline GEMV.
  - logit MSE and softmax-KL(clean || quant): magnitude of the perturbation.

The probe intercepts only `sample_from_logits` (the sampling boundary), so both
clean and quantized passes run the identical generate() machinery (same CFG,
logit_bias, EOS handling, CUDA-graph path); the sole difference is the weights.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
import torchaudio

import zonos.model as zmodel
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_quant import apply_fake_quant

SEED = 421
_REAL_SAMPLE = zmodel.sample_from_logits  # captured before any patching

# Configs to evaluate: (label, bits, group_size, target). Mirrors the knobs in
# run_reference.py / fake_quant.apply_fake_quant.
CONFIGS = [
    ("int8 per-channel  both",     8, None, "both"),
    ("int8 per-channel  in_proj",  8, None, "in_proj"),
    ("int8 per-channel  out_proj", 8, None, "out_proj"),
    ("int8 group=64     both",     8, 64,   "both"),
    ("int4 per-channel  both",     4, None, "both"),
    ("int4 group=64     both",     4, 64,   "both"),
]


# torchaudio>=2.7 MelSpectrogram device bug shim (same as run_reference.py).
_OriginalMelSpec = torchaudio.transforms.MelSpectrogram


class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *args, **kwargs):
        outer_device = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if outer_device.type != "cpu":
            self.to(outer_device)


torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec


class Recorder:
    """Drop-in replacement for sample_from_logits.

    Records the logits seen at every call. In greedy mode (forced=None) it
    delegates to the real sampler with temperature=0 (argmax + repetition
    penalty) and stores the chosen tokens. In teacher-forced mode it ignores the
    (quantized) logits for the decision and returns the pre-recorded clean token,
    pinning the rollout to the clean tape S.
    """

    def __init__(self, forced: list[torch.Tensor] | None = None):
        self.forced = forced
        self.logits: list[torch.Tensor] = []
        self.tokens: list[torch.Tensor] = []
        self.i = 0

    def __call__(self, logits, *args, **kwargs):
        self.logits.append(logits.detach().float().cpu().clone())
        if self.forced is None or self.i >= len(self.forced):
            # greedy, or teacher-forcing ran past the recorded tape (rollout-length
            # drift from forward non-determinism): fall back to greedy for this step.
            tok = _REAL_SAMPLE(logits, *args, **{**kwargs, "temperature": 0.0})
        else:
            tok = self.forced[self.i].to(logits.device)
        self.tokens.append(tok.detach().cpu().clone())
        self.i += 1
        return tok


@contextlib.contextmanager
def patched_sampler(rec: Recorder):
    old = zmodel.sample_from_logits
    zmodel.sample_from_logits = rec
    try:
        yield
    finally:
        zmodel.sample_from_logits = old


@torch.inference_mode()
def run_pass(model, conditioning, forced: list[torch.Tensor] | None) -> Recorder:
    """One full generate() pass with the sampler intercepted. Returns the Recorder
    holding per-step logits (and tokens). forced=None => greedy (defines the tape);
    forced=S => teacher-forced replay of S."""
    rec = Recorder(forced=forced)
    with patched_sampler(rec):
        model.generate(conditioning, sampling_params=dict(temperature=0.0), progress_bar=True)
    return rec


def snapshot_targets(model) -> dict[str, torch.Tensor]:
    """CPU copy of every Mamba in_proj/out_proj weight, so configs don't compound."""
    attn = set(model.backbone.config.attn_layer_idx)
    snap = {}
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(("in_proj", "out_proj")):
            idx = next((int(p) for p in name.split(".") if p.isdigit()), None)
            if idx in attn:
                continue
            snap[name] = mod.weight.detach().to("cpu").clone()
    return snap


def restore_targets(model, snap: dict[str, torch.Tensor]):
    mods = dict(model.backbone.named_modules())
    for name, w in snap.items():
        mods[name].weight.data = w.to(model.device, torch.bfloat16)


def divergence(L_clean: list[torch.Tensor], L_quant: list[torch.Tensor]) -> dict:
    """Per-step logit divergence stats. Each L[*] is [batch, 9, vocab] (post-CFG)."""
    n = min(len(L_clean), len(L_quant))
    C = torch.stack(L_clean[:n])  # [T, B, 9, V]
    Q = torch.stack(L_quant[:n])
    finite = torch.isfinite(C) & torch.isfinite(Q)

    ac, aq = C.argmax(-1), Q.argmax(-1)  # [T, B, 9]; argmax ignores -inf
    flip = ac != aq
    cb0_flip = flip[..., 0].float().mean().item()      # codebook 0: drives EOS + content
    all_flip = flip.float().mean().item()              # mean over all 9 codebooks
    frame_any = flip.any(-1).float().mean().item()     # any codebook flipped this frame

    diff = C - Q
    mse = (diff[finite] ** 2).mean().item()
    rms = (C[finite] ** 2).mean().sqrt().item()

    lc, lq = torch.log_softmax(C, -1), torch.log_softmax(Q, -1)
    p = lc.exp()
    kl = torch.where(p > 0, p * (lc - lq), torch.zeros_like(p)).sum(-1)  # [T, B, 9]
    kl = kl[torch.isfinite(kl)].mean().item()

    return dict(steps=n, cb0_flip=cb0_flip, all_flip=all_flip, frame_any=frame_any,
                mse=mse, rms=rms, rel_rms=(mse ** 0.5) / max(rms, 1e-9), kl=kl)


def main():
    torch.manual_seed(SEED)
    print(f"quant_logit_probe: seed={SEED} device={device}")

    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text="The kernels run on time, frame by frame.",
                          speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond)

    # 1) Clean greedy pass -> defines the tape S and the reference logits.
    print("\n[clean] greedy rollout to define teacher-forcing tape S ...")
    clean = run_pass(model, conditioning, forced=None)
    S = clean.tokens
    print(f"[clean] tape length = {len(S)} steps")

    snap = snapshot_targets(model)
    print(f"[snapshot] saved {len(snap)} target weights to CPU")

    rows = []

    # 2) Sanity: clean weights, teacher-forced through S. Must reproduce S exactly
    #    (flip ~= 0). Validates the forcing harness before trusting quant rows.
    print("\n[sanity] clean weights, teacher-forced (expect ~0 flips) ...")
    self_check = run_pass(model, conditioning, forced=S)
    rows.append(("clean (self-check)", divergence(clean.logits, self_check.logits)))

    # 3) Each quant config, teacher-forced through S.
    for label, bits, gs, target in CONFIGS:
        print(f"\n[{label}] applying quant + teacher-forced pass ...")
        restore_targets(model, snap)
        apply_fake_quant(model, bits=bits, group_size=gs, target=target, verbose=True)
        rec = run_pass(model, conditioning, forced=S)
        rows.append((label, divergence(clean.logits, rec.logits)))

    restore_targets(model, snap)

    # Report
    print("\n" + "=" * 100)
    print("TEACHER-FORCED LOGIT DIVERGENCE  (vs clean greedy tape S)")
    print(f"{'config':<24} {'cb0 flip%':>10} {'all-cb flip%':>13} {'frame-any%':>11} "
          f"{'logit MSE':>11} {'rel-RMS':>9} {'KL(c||q)':>10}")
    print("-" * 100)
    for label, m in rows:
        print(f"{label:<24} {100*m['cb0_flip']:>10.3f} {100*m['all_flip']:>13.3f} "
              f"{100*m['frame_any']:>11.3f} {m['mse']:>11.3e} {m['rel_rms']:>9.4f} {m['kl']:>10.4f}")
    print("=" * 100)
    print("go/no-go = cb0 flip%. Low (<~1-2%) + small KL -> int8 safe, green-light the kernel.")
    print("High -> quant changes decisions too much; try group-wise or bf16 roofline GEMV.")


if __name__ == "__main__":
    main()
