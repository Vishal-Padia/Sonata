"""Phase A quality gate: does swapping the Mamba2 projections to the CuTe GEMV
change the model's decisions? Reuses the teacher-forced flip probe from
scripts/quant_logit_probe.py -- the perturbation is the cute swap instead of
quantization. Runs EAGER (CUDA graph off) to isolate quality from graph capture.

Expect near-zero flips: unlike quant (which changed the weights ~1%), the kernel
is the same math as cuBLAS at bf16, so the only delta is accumulation-order
rounding. Near-zero -> quality GREEN, integration is numerically safe.
"""
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))                       # kernels/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))    # scripts/

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

from quant_logit_probe import Recorder, patched_sampler, divergence  # reuse harness + metrics
from patch import patch_zonos_projections

SEED = 421


@torch.inference_mode()
def run_pass_eager(model, conditioning, forced):
    """One greedy generate() pass, EAGER (no CUDA graph, no torch.compile), sampler
    intercepted. forced=None defines the tape S; forced=S teacher-forces it."""
    rec = Recorder(forced=forced)
    with patched_sampler(rec):
        model.generate(conditioning, sampling_params=dict(temperature=0.0),
                       progress_bar=False, disable_torch_compile=True)
    return rec


def main():
    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    model.can_use_cudagraphs = lambda: False  # force eager: isolate quality from graph capture

    wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text="The kernels run on time, frame by frame.",
                          speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond)

    print("[clean] greedy rollout (stock cuBLAS projections) -> tape S ...")
    clean = run_pass_eager(model, conditioning, forced=None)
    S = clean.tokens
    print(f"[clean] tape length = {len(S)} steps")

    print("[sanity] clean weights, teacher-forced (expect ~0 flips) ...")
    sc = run_pass_eager(model, conditioning, forced=S)
    m_sc = divergence(clean.logits, sc.logits)

    print("[cute] patch Mamba2 in_proj/out_proj -> teacher-forced ...")
    npatched = patch_zonos_projections(model, verbose=False)
    print(f"[cute] patched {npatched} projections")
    rec = run_pass_eager(model, conditioning, forced=S)
    m = divergence(clean.logits, rec.logits)

    print("\n" + "=" * 92)
    print("TEACHER-FORCED DIVERGENCE vs clean cuBLAS tape S  (eager, Mamba2 projections -> CuTe GEMV)")
    print(f"{'config':<22} {'cb0 flip%':>10} {'all-cb flip%':>13} {'frame-any%':>11} "
          f"{'logit MSE':>11} {'rel-RMS':>9} {'KL(c||q)':>10}")
    print("-" * 92)
    for label, mm in [("clean (self-check)", m_sc), ("cute projections", m)]:
        print(f"{label:<22} {100*mm['cb0_flip']:>10.3f} {100*mm['all_flip']:>13.3f} "
              f"{100*mm['frame_any']:>11.3f} {mm['mse']:>11.3e} {mm['rel_rms']:>9.4f} {mm['kl']:>10.4f}")
    print("=" * 92)
    # Verdict: cute should be indistinguishable from the self-check noise floor.
    verdict = "GREEN" if m["cb0_flip"] <= max(m_sc["cb0_flip"], 0.01) + 1e-6 else (
        "YELLOW" if m["cb0_flip"] < 0.02 else "RED")
    print(f"cb0 flip {100*m['cb0_flip']:.3f}% vs self-check {100*m_sc['cb0_flip']:.3f}%  ->  {verdict}")
    print("(same math as cuBLAS at bf16; near-noise-floor flips => numerically safe to integrate)")


if __name__ == "__main__":
    main()
