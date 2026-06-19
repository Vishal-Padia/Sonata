# Numerical-validation harness (Phase 0, step 7)

The gate every Phase 1/2 change runs through. Two layers, because they catch
different failure modes:

- **Per-kernel** (`scripts/validation_harness.py`) — catches numerical drift at the source, in seconds, pointing at the offending op. Single decode step.
- **End-to-end audio** (`scripts/audio_fidelity.py`) — catches what per-kernel can't: drift accumulating across ~500 AR steps, an argmax flipped over a logit boundary, perceptual artifacts. A kernel can pass per-kernel and still wreck the audio; it canfail per-kernel by an irrelevant amount and produce identical audio. You need both.

Goldens are **frozen** against the current upstream lib versions (`mamba_ssm` 2.3.2, `causal_conv1d` 1.6.2). Re-pin only intentionally (`--regen-goldens` / `--regen-golden`), never in CI. The comparison target is the reference op **at the dtype the kernel runs at** (bf16 I/O, fp32 A) — not an fp64 math chain.

---

## Layer 1 — per-kernel

Four fused-op boundaries inside one Mamba2 decode layer (from `docs/baseline_op_level.md`). Projections (`nn.Linear`) are deliberately **not** tested here — they're checked by the audio gate plus generic linear tests. Shapes pinned to the checkpoint (`docs/architecture.md`): `batch=2` (cfg), `d_inner=4096`, `d_state=128`, `headdim=64`, `n_heads=64`, `d_conv=4`, `ngroups=1`. `conv_dim = d_ssm + 2·ngroups·d_state = 4352`. The harness
asserts `d_ssm == d_inner` (the no-gated-MLP case this checkpoint uses, `d_intermediate=0`,
`D_has_hdim=False`); a config with `d_ssm < d_inner` fails loudly rather than silently
validating the wrong shapes.

Each fixture mirrors exactly what `Mamba2.step()` passes (verified against the live signatures), including the subtleties that are easy to get wrong:

- **causal_conv1d_update** — `activation="silu"` is passed (the SiLU is *inside* the op). Outputs checked: returned `y` **and** the in-place-mutated `conv_state`.
- **selective_state_update** — `z=None` (rmsnorm path; the gate lives in the gated norm, not here). `A` is **fp32 and negative** (`-exp(A_log)`) — positive/bf16 A would explode the recurrence and make the golden meaningless. `A:(nh,hd,ds)`, `D/dt_bias:(nh,hd)`, `dt_softplus=True`. Outputs checked: `y` **and** the in-place `ssm_state` (a constant state offset passes per-step but blows the audio gate — so we assert it explicitly).
- **rmsnorm_gated** — `norm_before_gate=False`, `group_size=4096`, `bias=None` (matches `RMSNormGated` in `Mamba2.__init__`). Default `norm_before_gate=True` would be wrong.
- **layer_norm_fn** — `is_rms_norm=False` (the backbone is **LayerNorm**, `rms_norm=False`), `residual_in_fp32=False`, `prenorm=True`, **with bias**. Outputs: `(hidden_states, residual)`. This is the per-Block entry add-norm (×46 per pass).
- **layer_norm_fn_exit** — the **same kernel** on its `prenorm=False` branch: the once-per-pass stack-exit norm (`_mamba_ssm.py:49`) adds the residual but does **not** return it (single output). A fused replacement could be correct on the prenorm=True branch and buggy on this one, so it gets its own fixture.

### Tolerances (calibrated to the measured bf16 dtype floor)

A tolerance is a claim. The **empirical pass** (`--measure-floors`) runs each reference op at bf16 vs the same inputs upcast to fp32; that gap is the irreducible bf16 error. Tolerance is set to **~3× the measured floor**, so a legitimate bf16 kernel is not false-failed.

| op / output | measured floor (max / mean) | pinned tol (max / mean) |
|---|---|---|
| causal_conv1d_update `y` | 2.99e-2 / 1.29e-3 | 9e-2 / 4e-3 |
| causal_conv1d_update `conv_state` | 0 / 0 | 5e-3 / 1e-3 (kept tight) |
| selective_state_update `y` | 1.13e-1 / 8.6e-3 | 3.5e-1 / 3e-2 |
| selective_state_update `ssm_state` | 6.24e-2 / 1.17e-3 | 2e-1 / 4e-3 |
| rmsnorm_gated `y` | 2.78e-2 / 5.98e-4 | 9e-2 / 2e-3 |
| layer_norm_fn `hidden_states` | 1.56e-2 / 1.49e-3 | 5e-2 / 5e-3 |
| layer_norm_fn `residual` | 1.56e-2 / 1.42e-3 | 5e-2 / 5e-3 |

**Why `selective_state_update` is so loose (0.35):** with `randn` inputs the output magnitude is ~10 (sum over `d_state=128`), so ~1% bf16 error is ~0.1 absolute. This is expected, and it's exactly why per-kernel is single-step only — **accumulation is the audio gate's job**, not this one. (If we later want a magnitude-robust per-op check, add a relative-error tier.) Multi-step rollout bounds grow roughly linearly per step; not tested here (decode is single-step).

### Usage

```
uv run scripts/validation_harness.py --regen-goldens   # re-pin, manual, on lib version bump only
uv run scripts/validation_harness.py --measure-floors  # print bf16-vs-fp32 floors
uv run scripts/validation_harness.py                   # check candidate (reference for now) vs goldens
uv run scripts/validation_harness.py --op selective_state_update   # single op
```

Default mode currently runs **reference-vs-its-own-golden** (sanity: must be ~0, ALL GREEN). Once a fused kernel exists, swap it in as `candidate_fn` in `check_op`. Goldens live in `tests/goldens/kernels/*.pt` with a sha256 `MANIFEST.json`; `load_golden` verifies the hash, so a silently-edited golden fails loudly.

---

## Layer 2 — end-to-end audio

Golden = `tests/goldens/audio/baseline.wav` (the deterministic seed-421 reference output, pinned from repo-root `baseline.wav`) + a sha256 sidecar + a `baseline.meta.json` provenance sidecar (seed, lib versions, GPU, CUDA). Native 44.1 kHz mono.

**Provenance contract (so a golden can't silently become "whatever ran today"):**
- `regen_golden` runs `run_reference.py --seed 421` explicitly (the script now takes `--seed` so the contract is enforced at the source, not hidden in a hardcode), and **refuses to mint** a golden under upstream-library drift unless `--force` (then bump `PINNED_LIBS`).
- `check_audio` reads `baseline.meta.json` and **refuses to compare** against a golden whose recorded lib versions / seed differ from the current env (a different ground truth). GPU/driver differences only **warn** — the metric runs on CPU; the env only affected how the golden was generated.

### Metrics

- **Waveform SNR (dB)** = `10·log10(‖g‖² / ‖g−c‖²)` — strict; only high if the AR loop is ~numerically identical. Also reports peak-abs-err normalized to golden peak. Lengths are truncated to the shorter (never zero-padded — that lies to SNR). If candidate and golden sample rates differ, the candidate is resampled to the golden's rate for SNR (warned), rather than erroring.
- **log-mel L1 (dB)** — resample both to 16 kHz, log-mel, `10·log10(mel + 1e-10)`, mean |Δ|. The `+1e-10` floor stops silent tails producing −inf. **Every** `MelSpectrogram` param is pinned in `MEL_CFG` — including the ones with version-varying torchaudio defaults (`mel_scale="htk"`, `norm=None`, `window_fn=hann`, `center=True`, `pad_mode="reflect"`) — so the metric is reproducible across torchaudio versions. The transform is built once (module-level cache), not per call. Perceptual proxy: TTS audio can diverge sample-by-sample yet sound identical (a different-but-valid token sequence), and mel-L1 captures spectral content over time.

### Verdicts

| verdict | condition | meaning | exit |
|---|---|---|---|
| GREEN | SNR > 50 dB **and** mel-L1 < 0.05 | effectively bit-equivalent audio | 0 |
| YELLOW | SNR fails but mel-L1 < 0.5 | perceptually equiv, token seq diverged — review by ear | 0 (warns) |
| RED | mel-L1 ≥ 0.5 | audible regression — block | 2 |
| (error) | missing golden, sha mismatch, NaN | harness bug | 3 |

### Calibration (build-order step 5)

| candidate | SNR | mel-L1 | verdict |
|---|---|---|---|
| baseline vs baseline | ∞ | 0.0000 | GREEN |
| +1e-4 noise | 62.96 | 0.0660 | YELLOW |
| +5e-3 noise | 29.65 | 4.0753 | RED |
| +5e-2 noise | 9.65 | 14.2961 | RED |

**Open calibration note:** `MEL_GREEN_DB=0.05` is tight — a 63 dB-SNR signal already reads YELLOW on mel alone. A truly numerically-equivalent kernel gives mel≈0 (GREEN), but a bf16-reordered fused kernel may land just above 0.05 and read YELLOW when it should be GREEN. **Revisit `MEL_GREEN_DB` (likely 0.1–0.2) once the first real fused kernel's mel is observed.** The YELLOW/RED boundary (0.5) discriminates well (0.066 → YELLOW, 4.08 → RED).

### Usage

```
uv run scripts/audio_fidelity.py --regen-golden                  # pin baseline.wav as golden
uv run scripts/audio_fidelity.py                                 # check repo-root baseline.wav
uv run scripts/audio_fidelity.py --candidate path/to/out.wav     # check a candidate run
```

---
