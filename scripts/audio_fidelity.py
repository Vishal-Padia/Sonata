from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path

import torch
import torchaudio

SEED = 421
GOLDEN_WAV = Path("tests/goldens/audio/baseline.wav")
GOLDEN_SHA = Path("tests/goldens/audio/baseline.sha256")
GOLDEN_META = Path("tests/goldens/audio/baseline.meta.json")
SOURCE_WAV = Path("baseline.wav")  # repo-root reference output (run_reference.py, seed 421)

# Goldens are frozen against these upstream versions. A golden minted under a
# different mamba_ssm/causal_conv1d/torch is a different ground truth — regen
# refuses to mint under drift, and check refuses to compare against a golden
# whose recorded versions differ from the current env. See docs/validation.md.
PINNED_LIBS = {
    "torch": "2.8.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "mamba_ssm": "2.3.2.post1",
    "causal_conv1d": "1.6.2.post1",
}

# Calibrated against baseline-vs-baseline (GREEN, SNR=inf/mel=0) and a deliberate
# perturbation (YELLOW/RED). See docs/validation.md. Refine as kernels land.
SNR_GREEN_DB = 50.0
MEL_GREEN_DB = 0.05 # log-mel L1, dB units
MEL_YELLOW_DB = 0.5

# Mel config — ALL params pinned (incl. the ones with version-varying defaults:
# mel_scale, norm, window_fn, center, pad_mode) so the metric is reproducible
# across torchaudio versions.
MEL_CFG = dict(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80,
               f_min=0.0, f_max=8000.0, power=2.0,
               mel_scale="htk", norm=None, center=True, pad_mode="reflect",
               window_fn=torch.hann_window)
LOG_FLOOR = 1e-10 # avoids -inf on silent tails; documented in docs/validation.md

_MEL_TRANSFORM = None   # built once, lazily (see _mel)


class Verdict(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclasses.dataclass
class AudioMetrics:
    sample_rate: int
    duration_s: float
    n_samples_compared: int
    length_mismatch: bool
    peak_abs_err: float
    snr_db: float
    mel_l1_db: float
    verdict: Verdict

    def fmt(self) -> str:
        warn = "  (length mismatch: truncated to shorter)" if self.length_mismatch else ""
        return (
            f"  sample_rate : {self.sample_rate} Hz\n"
            f"  duration : {self.duration_s:.3f} s ({self.n_samples_compared} samples){warn}\n"
            f"  peak_abs_err: {self.peak_abs_err:.3e}  (normalized to golden peak)\n"
            f"  SNR : {self.snr_db:.2f} dB   (green > {SNR_GREEN_DB})\n"
            f"  log-mel L1 : {self.mel_l1_db:.4f} dB (green < {MEL_GREEN_DB}, yellow < {MEL_YELLOW_DB})\n"
            f"  VERDICT : {self.verdict.value}"
        )


# Golden generation

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lib_versions() -> dict:
    import causal_conv1d
    import mamba_ssm
    return {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__version__}


def _env() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"seed": SEED, "libs": _lib_versions(),
            "gpu": gpu, "cuda": torch.version.cuda, "mel_cfg_sr": MEL_CFG["sample_rate"]}


def regen_golden(force: bool = False):
    """Pin baseline.wav (deterministic seed-421 reference output) as the golden,
    plus a meta sidecar (seed + lib versions + GPU). Refuses to mint under library
    drift unless --force, so a venv change can't silently mint a new ground truth."""
    cur, drift = _lib_versions(), {}
    for k, want in PINNED_LIBS.items():
        if cur.get(k) != want:
            drift[k] = f"{cur.get(k)} != pinned {want}"
    if drift and not force:
        raise SystemExit(f"refusing to regen golden under library drift: {drift}\n"
                         f"upgrade was intentional? re-run with --force and bump PINNED_LIBS.")
    GOLDEN_WAV.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE_WAV.exists():
        print(f"{SOURCE_WAV} missing — regenerating via scripts/run_reference.py --seed {SEED} ...")
        subprocess.run([sys.executable, "scripts/run_reference.py", "--seed", str(SEED),
                        "--out", str(SOURCE_WAV)], check=True)
    shutil.copyfile(SOURCE_WAV, GOLDEN_WAV)
    sha = _sha256(GOLDEN_WAV)
    GOLDEN_SHA.write_text(sha + "\n")
    GOLDEN_META.write_text(json.dumps(_env(), indent=2, sort_keys=True))
    print(f"golden pinned: {GOLDEN_WAV}  sha256={sha[:16]}...  meta={_env()['libs']}")


def _mel(wav: torch.Tensor) -> torch.Tensor:
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
        _MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(**MEL_CFG)  # built once
    return _MEL_TRANSFORM(wav)


# Metrics — pure functions of two waveforms; computed on CPU/fp32.

def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))  # (channels, samples)
    wav = wav.to(torch.float32).mean(dim=0)  # downmix to mono (samples,)
    return wav, sr


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


def _align(golden: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Truncate both to the shorter length (never zero-pad — that lies to SNR)."""
    n = min(golden.numel(), candidate.numel())
    return golden[:n], candidate[:n], golden.numel() != candidate.numel()


def waveform_metrics(golden: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    """Returns (peak_abs_err_normalized, snr_db). Inputs already aligned."""
    err = (golden - candidate)
    peak = golden.abs().max().clamp_min(1e-12)
    peak_abs_err = (err.abs().max() / peak).item()
    sig = (golden ** 2).sum()
    noise = (err ** 2).sum()
    snr_db = float("inf") if noise.item() == 0.0 else (10.0 * torch.log10(sig / noise)).item()
    return peak_abs_err, snr_db


def mel_l1_db(golden: torch.Tensor, candidate: torch.Tensor, native_sr: int) -> float:
    """Resample both to MEL_CFG sample_rate, log-mel, mean L1 in dB. Inputs aligned."""
    tsr = MEL_CFG["sample_rate"]
    g = _resample(golden, native_sr, tsr)
    c = _resample(candidate, native_sr, tsr)
    n = min(g.numel(), c.numel())
    g, c = g[:n], c[:n]
    lg = 10.0 * torch.log10(_mel(g) + LOG_FLOOR)
    lc = 10.0 * torch.log10(_mel(c) + LOG_FLOOR)
    return (lg - lc).abs().mean().item()


def classify(snr_db: float, mel_db: float) -> Verdict:
    if snr_db > SNR_GREEN_DB and mel_db < MEL_GREEN_DB:
        return Verdict.GREEN
    if mel_db < MEL_YELLOW_DB:
        return Verdict.YELLOW
    return Verdict.RED


# Top level

def _check_meta():
    """Refuse to compare against a golden minted under different upstream libs
    (a different ground truth). GPU/driver differences only warn — comparison is
    on CPU and env only affects how the golden was generated, not the metric."""
    if not GOLDEN_META.exists():
        print(f"  [WARN] no {GOLDEN_META}; provenance unknown (regen to create it)")
        return
    meta = json.loads(GOLDEN_META.read_text())
    cur = _lib_versions()
    drift = {k: f"golden {meta['libs'].get(k)} != current {cur.get(k)}"
             for k in cur if meta.get("libs", {}).get(k) != cur.get(k)}
    if drift:
        raise ValueError(f"golden was minted under different libraries: {drift} — re-pin or fix venv")
    if meta.get("seed") != SEED:
        raise ValueError(f"golden seed {meta.get('seed')} != harness SEED {SEED}")
    if meta.get("gpu") and torch.cuda.is_available() and meta["gpu"] != torch.cuda.get_device_name(0):
        print(f"  [WARN] golden GPU {meta['gpu']} != current {torch.cuda.get_device_name(0)} "
              f"(comparison still valid; generation env differed)")


def check_audio(candidate_path: Path) -> AudioMetrics:
    if not GOLDEN_WAV.exists() or not GOLDEN_SHA.exists():
        raise FileNotFoundError(f"no golden at {GOLDEN_WAV}; run --regen-golden")
    if _sha256(GOLDEN_WAV) != GOLDEN_SHA.read_text().strip():
        raise ValueError(f"golden sha mismatch — {GOLDEN_WAV} edited or corrupt")
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate not found: {candidate_path}")
    _check_meta()

    golden, g_sr = _load_mono(GOLDEN_WAV)
    cand, c_sr = _load_mono(candidate_path)
    if torch.isnan(cand).any() or torch.isinf(cand).any():
        raise ValueError("candidate waveform contains NaN/Inf")
    if g_sr != c_sr:
        # SNR needs a common rate; resample candidate -> golden SR (mel path resamples anyway).
        print(f"  [WARN] sample-rate mismatch: candidate {c_sr} -> resampled to golden {g_sr} for SNR")
        cand = _resample(cand, c_sr, g_sr)

    golden, cand, mismatch = _align(golden, cand)
    peak_abs_err, snr_db = waveform_metrics(golden, cand)
    mel_db = mel_l1_db(golden, cand, g_sr)
    verdict = classify(snr_db, mel_db)
    return AudioMetrics(g_sr, golden.numel() / g_sr, golden.numel(), mismatch,
                        peak_abs_err, snr_db, mel_db, verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-golden", action="store_true")
    ap.add_argument("--force", action="store_true", help="regen even under library drift (then bump PINNED_LIBS)")
    ap.add_argument("--candidate", type=Path, default=SOURCE_WAV)
    args = ap.parse_args()

    if args.regen_golden:
        regen_golden(force=args.force)
        return 0

    try:
        m = check_audio(args.candidate)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 3

    print(f"audio fidelity: {args.candidate} vs {GOLDEN_WAV}")
    print(m.fmt())
    if m.verdict is Verdict.RED:
        return 2
    if m.verdict is Verdict.YELLOW:
        print("  [WARN] YELLOW — token sequence diverged but perceptually close; review by listening.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
