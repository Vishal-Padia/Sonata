from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch

SEED = 421
GOLDENS_DIR = Path("tests/goldens/kernels")
MANIFEST_PATH = GOLDENS_DIR / "MANIFEST.json"

# Pinned from the backbone config + Mamba2 defaults (docs/architecture.md).
S = {
    "batch": 2, # cfg-doubled decode
    "d_inner": 4096, # expand * d_model
    "d_ssm": 4096, # == d_inner for this checkpoint (d_intermediate=0, D_has_hdim=False)
    "d_state": 128,
    "headdim": 64,
    "n_heads": 64, # d_ssm / headdim
    "d_conv": 4,
    "ngroups": 1,
    "d_model": 2048,
    "eps": 1e-5,
}
# This fixture only covers the checkpoint where the gated-MLP branch is absent
# (d_ssm == d_inner). On a config with d_ssm < d_inner the conv width and n_heads
# differ — fail loudly rather than silently validate the wrong shapes.
assert S["d_ssm"] == S["d_inner"], "fixture assumes d_ssm == d_inner; re-derive for gated-MLP configs"
# conv operates on xBC = x || B || C  =  d_ssm + 2*ngroups*d_state
CONV_DIM = S["d_ssm"] + 2 * S["ngroups"] * S["d_state"]   # 4352

# Goldens are frozen against these upstream versions (see docs/validation.md).
PINNED_LIBS = {"mamba_ssm": "2.3.2.post1", "causal_conv1d": "1.6.2.post1", "torch": "2.8.0+cu128"}


@dataclasses.dataclass
class OpSpec:
    name: str
    build_inputs: Callable[[torch.device, torch.Generator], dict[str, Any]]
    load_reference: Callable[[], Callable[..., Any]]
    run: Callable[[Callable[..., Any], dict[str, Any]], tuple[torch.Tensor, ...]]
    out_names: list[str]
    tolerances: list[tuple[float, float]] # (max_abs, mean_abs) per output
    tolerance_notes: list[str]


def _gen(device, g, shape, dtype=torch.bfloat16, scale=1.0):
    return (torch.randn(shape, device=device, generator=g, dtype=torch.float32) * scale).to(dtype)


# Fixtures — shapes/dtypes exactly mirror what Mamba2.step() passes

def build_causal_conv1d_update(device, g):
    # x: (B, conv_dim); conv_state: (B, conv_dim, d_conv); weight: (conv_dim, d_conv); bias: (conv_dim,)
    return {
        "x": _gen(device, g, (S["batch"], CONV_DIM)),
        "conv_state": _gen(device, g, (S["batch"], CONV_DIM, S["d_conv"])),  # non-zero: hides nothing
        "weight": _gen(device, g, (CONV_DIM, S["d_conv"])),
        "bias": _gen(device, g, (CONV_DIM,)),
        "activation": "silu",
    }


def build_selective_state_update(device, g):
    nh, hd, ds = S["n_heads"], S["headdim"], S["d_state"]
    # A = -exp(A_log) per-head, fp32, expanded to (nheads, headdim, dstate). MUST be negative
    # for a stable recurrence (dA = exp(dt*A) in (0,1)); positive A would explode the golden.
    a_head = -torch.exp(torch.randn(nh, device=device, generator=g, dtype=torch.float32))
    A = a_head[:, None, None].expand(nh, hd, ds).contiguous()
    dtb = torch.randn(nh, device=device, generator=g, dtype=torch.float32)
    D_h = torch.randn(nh, device=device, generator=g, dtype=torch.float32)
    return {
        "state": _gen(device, g, (S["batch"], nh, hd, ds)), # non-zero seeded state
        "x": _gen(device, g, (S["batch"], nh, hd)),
        "dt": _gen(device, g, (S["batch"], nh, hd)),
        "A": A, # fp32, (nh, hd, ds)
        "B": _gen(device, g, (S["batch"], S["ngroups"], ds)),
        "C": _gen(device, g, (S["batch"], S["ngroups"], ds)),
        "D": D_h[:, None].expand(nh, hd).contiguous().to(torch.bfloat16), # (nh, hd)
        "z": None, # rmsnorm path => no gate here
        "dt_bias": dtb[:, None].expand(nh, hd).contiguous().to(torch.bfloat16), # (nh, hd)
        "dt_softplus": True,
    }


def build_rmsnorm_gated(device, g):
    # gated RMSNorm over d_inner; norm_before_gate=False, group_size = d_inner (ngroups=1); no bias.
    return {
        "x": _gen(device, g, (S["batch"], S["d_inner"])),
        "weight": _gen(device, g, (S["d_inner"],)),
        "bias": None,
        "z": _gen(device, g, (S["batch"], S["d_inner"])),
        "eps": S["eps"],
        "group_size": S["d_inner"] // S["ngroups"],
        "norm_before_gate": False,
    }


def build_layer_norm_fn(device, g):
    # entry fused add-norm: LayerNorm (is_rms_norm=False), residual_in_fp32=False, WITH bias.
    return {
        "x": _gen(device, g, (S["batch"], 1, S["d_model"])),
        "residual": _gen(device, g, (S["batch"], 1, S["d_model"])),
        "weight": _gen(device, g, (S["d_model"],)),
        "bias": _gen(device, g, (S["d_model"],)),
        "eps": S["eps"],
    }


# Reference loaders

def load_ref_causal_conv1d_update():
    from causal_conv1d import causal_conv1d_update
    return causal_conv1d_update


def load_ref_selective_state_update():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    return selective_state_update


def load_ref_rmsnorm_gated():
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    return rmsnorm_fn


def load_ref_layer_norm_fn():
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
    return layer_norm_fn


# Runners — call the PASSED op (so a candidate kernel can be swapped in), clone
# any in-place-mutated state, and return (returned_output, *mutated_states).

def run_causal_conv1d_update(op, inp):
    conv_state = inp["conv_state"].clone() # mutated in place — check it too
    y = op(inp["x"], conv_state, inp["weight"], inp["bias"], inp["activation"])
    return (y, conv_state)


def run_selective_state_update(op, inp):
    state = inp["state"].clone() # mutated in place — check it too
    y = op(state, inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"],
           z=inp["z"], dt_bias=inp["dt_bias"], dt_softplus=inp["dt_softplus"])
    return (y, state)


def run_rmsnorm_gated(op, inp):
    y = op(inp["x"], inp["weight"], inp["bias"], z=inp["z"], eps=inp["eps"],
           group_size=inp["group_size"], norm_before_gate=inp["norm_before_gate"])
    return (y,)


def run_layer_norm_fn(op, inp):
    out, residual = op(inp["x"], inp["weight"], inp["bias"], residual=inp["residual"],
                       eps=inp["eps"], prenorm=True, residual_in_fp32=False, is_rms_norm=False)
    return (out, residual)


def run_layer_norm_fn_exit(op, inp):
    # The stack-exit norm calls layer_norm_fn WITHOUT prenorm=,
    # so prenorm=False: residual is added but NOT returned, single output. Different
    # code path through the same kernel (the residual-return branch is gated off).
    out = op(inp["x"], inp["weight"], inp["bias"], residual=inp["residual"],
             eps=inp["eps"], prenorm=False, residual_in_fp32=False, is_rms_norm=False)
    return (out,)


OPS: dict[str, OpSpec] = {
    # Tolerances = ~3x the measured bf16-vs-fp32 dtype floor (run --measure-floors).
    # They are deliberately >= the irreducible bf16 error so a legitimate bf16 kernel
    # is not false-failed; single-step only.
    "causal_conv1d_update": OpSpec(
        "causal_conv1d_update", build_causal_conv1d_update, load_ref_causal_conv1d_update,
        run_causal_conv1d_update, ["y", "conv_state"],
        tolerances=[(9e-2, 4e-3), (5e-3, 1e-3)],
        tolerance_notes=["y: floor 2.99e-2/1.29e-3 (conv width 4 + SiLU)",
                         "conv_state: floor 0 (roll+overwrite, no arithmetic) — kept tight"],
    ),
    "selective_state_update": OpSpec(
        "selective_state_update", build_selective_state_update, load_ref_selective_state_update,
        run_selective_state_update, ["y", "ssm_state"],
        tolerances=[(3.5e-1, 3e-2), (2e-1, 4e-3)],
        tolerance_notes=["y: floor 1.13e-1/8.6e-3 — large because randn output magnitude ~10 (sum over d_state=128)",
                         "ssm_state: floor 6.24e-2/1.17e-3; a constant offset here passes per-step but blows the audio gate"],
    ),
    "rmsnorm_gated": OpSpec(
        "rmsnorm_gated", build_rmsnorm_gated, load_ref_rmsnorm_gated,
        run_rmsnorm_gated, ["y"],
        tolerances=[(9e-2, 2e-3)],
        tolerance_notes=["y: floor 2.78e-2/5.98e-4 (norm over 4096 + SiLU(z) gate, norm_before_gate=False)"],
    ),
    "layer_norm_fn": OpSpec(
        "layer_norm_fn", build_layer_norm_fn, load_ref_layer_norm_fn,
        run_layer_norm_fn, ["hidden_states", "residual"],
        tolerances=[(5e-2, 5e-3), (5e-2, 5e-3)],
        tolerance_notes=["hidden_states: floor 1.56e-2/1.49e-3 (LayerNorm(x+residual))",
                         "residual: floor 1.56e-2/1.42e-3 (x+residual passthrough, prenorm)"],
    ),
    # Same kernel, prenorm=False branch — the once-per-pass stack-exit norm (_mamba_ssm.py:49).
    "layer_norm_fn_exit": OpSpec(
        "layer_norm_fn_exit", build_layer_norm_fn, load_ref_layer_norm_fn,
        run_layer_norm_fn_exit, ["out"],
        tolerances=[(5e-2, 5e-3)],
        tolerance_notes=["out: LayerNorm(x+residual), prenorm=False (residual not returned)"],
    ),
}


# Golden I/O (frozen tensors + sha256 manifest)

def _golden_path(name): return GOLDENS_DIR / f"{name}.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_golden(name, inputs, outputs):
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    path = _golden_path(name)
    torch.save({"inputs": _to_cpu(inputs), "outputs": _to_cpu(outputs)}, path)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    manifest[name] = {"file": path.name, "sha256": _sha256(path)}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_golden(name):
    path = _golden_path(name)
    manifest = json.loads(MANIFEST_PATH.read_text())
    if name not in manifest:
        raise FileNotFoundError(f"no golden for {name}; run --regen-goldens")
    if _sha256(path) != manifest[name]["sha256"]:
        raise ValueError(f"sha256 mismatch for {name} golden — file edited or corrupt")
    blob = torch.load(path, weights_only=False)
    return blob["inputs"], blob["outputs"]


# Diff

@dataclasses.dataclass
class DiffResult:
    op: str
    out_name: str
    max_abs: float
    mean_abs: float
    tol_max: float
    tol_mean: float
    passed: bool

    def fmt(self):
        flag = "PASS" if self.passed else "FAIL"
        return (f"  [{flag}] {self.op}.{self.out_name:<12} "
                f"max={self.max_abs:.2e} (tol {self.tol_max:.1e})  "
                f"mean={self.mean_abs:.2e} (tol {self.tol_mean:.1e})")


def diff_tensors(op, out_name, candidate, golden, tol_max, tol_mean) -> DiffResult:
    c = candidate.detach().float().cpu()
    gold = golden.detach().float().cpu()
    if c.shape != gold.shape:
        raise ValueError(f"{op}.{out_name}: shape {tuple(c.shape)} vs golden {tuple(gold.shape)}")
    bad = bool(torch.isnan(c).any() or torch.isinf(c).any())
    diff = (c - gold).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    passed = (not bad) and max_abs <= tol_max and mean_abs <= tol_mean
    return DiffResult(op, out_name, max_abs, mean_abs, tol_max, tol_mean, passed)


# Top level

def assert_pinned_libs(force=False):
    import causal_conv1d
    import mamba_ssm
    cur = {"mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__version__,
           "torch": torch.__version__}
    drift = {k: f"{cur.get(k)} != pinned {v}" for k, v in PINNED_LIBS.items() if cur.get(k) != v}
    if drift and not force:
        raise SystemExit(f"refusing to regen goldens under library drift: {drift}\n"
                         f"intentional upgrade? re-run with --force and bump PINNED_LIBS.")


def regen_golden(name, device):
    g = torch.Generator(device=device).manual_seed(SEED)
    spec = OPS[name]
    inputs = spec.build_inputs(device, g)
    outputs = spec.run(spec.load_reference(), inputs)
    save_golden(name, inputs, outputs)
    print(f"regen {name}: " + ", ".join(
        f"{n}{tuple(o.shape)}/{o.dtype}".replace("torch.", "") for n, o in zip(spec.out_names, outputs)))


def check_op(name, candidate_fn, device) -> list[DiffResult]:
    inputs, golden_outs = load_golden(name)
    spec = OPS[name]
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    op = candidate_fn if candidate_fn is not None else spec.load_reference()
    cand_outs = spec.run(op, inputs)
    if len(cand_outs) != len(golden_outs):
        raise ValueError(f"{name}: candidate returned {len(cand_outs)} outputs, golden has {len(golden_outs)}")
    return [diff_tensors(name, spec.out_names[i], cand_outs[i], golden_outs[i],
                         spec.tolerances[i][0], spec.tolerances[i][1])
            for i in range(len(golden_outs))]


def measure_floors(name, device):
    """bf16-vs-fp32 dtype floor: run the reference on native inputs and on the
    same inputs upcast to fp32, diff. The tolerance must be >= this floor."""
    g = torch.Generator(device=device).manual_seed(SEED)
    spec = OPS[name]
    ref = spec.load_reference()
    inp = spec.build_inputs(device, g)
    fp32 = {k: (v.float() if (torch.is_tensor(v) and v.is_floating_point()) else v) for k, v in inp.items()}
    native_outs = spec.run(ref, inp)
    fp32_outs = spec.run(ref, fp32)
    print(f"{name} dtype floor (bf16 vs fp32):")
    for n, a, b in zip(spec.out_names, native_outs, fp32_outs):
        d = (a.float().cpu() - b.float().cpu()).abs()
        print(f"  {n:<12} max={d.max().item():.2e}  mean={d.mean().item():.2e}")


def check_gemv(device) -> int:
    """Regression guard for the CuTeDSL projection GEMV (kernels/gemv.py), live vs
    cuBLAS and fp32 at the decode shapes (M=2). Matched-reference discipline like the
    audio gate: recompute the reference, no frozen golden. Gates cute-vs-fp32 (the
    kernel must stay >= cuBLAS accuracy); reports cute-vs-cuBLAS. Tight tolerance so a
    future kernel change that silently breaks accuracy fails here, not by ear."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
    from gemv import candidate_cute_splitk

    TOL = 3e-3  # ~2x the measured cute-vs-fp32 floor (1.66e-3); catches real breakage
    shapes = {"in_proj": (8512, 2048), "out_proj": (2048, 4096),
              "attn fc1": (16384, 2048), "attn fc2": (2048, 8192)}
    g = torch.Generator(device=device).manual_seed(SEED)
    rms = lambda a: (a.float() ** 2).mean().sqrt().item()
    print(f"GEMV regression (M=2, cute vs cuBLAS/fp32; PASS if cute-vs-fp32 < {TOL:.0e} rel-RMS)")
    failed = 0
    for name, (N, K) in shapes.items():
        x = _gen(device, g, (2, K))
        W = _gen(device, g, (N, K))
        ref = x.float() @ W.float().t()
        r_fp32 = rms(candidate_cute_splitk(x, W).float() - ref) / rms(ref)
        r_cublas = rms(candidate_cute_splitk(x, W).float() - (x @ W.t()).float()) / rms(ref)
        ok = r_fp32 < TOL
        failed += not ok
        print(f"  {name:<10} cute-vs-fp32={r_fp32:.2e}  cute-vs-cuBLAS={r_cublas:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{'GEMV GREEN' if failed == 0 else f'{failed} GEMV FAILURE(S)'}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-goldens", action="store_true")
    ap.add_argument("--measure-floors", action="store_true")
    ap.add_argument("--gemv", action="store_true", help="regression-check the CuTeDSL projection GEMV vs cuBLAS/fp32")
    ap.add_argument("--force", action="store_true", help="regen even under library drift (then bump PINNED_LIBS)")
    ap.add_argument("--op", default=None, help="single op by name")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [args.op] if args.op else list(OPS.keys())

    if args.regen_goldens:
        assert_pinned_libs(force=args.force)
        for name in targets:
            regen_golden(name, device)
        return 0
    if args.measure_floors:
        for name in targets:
            measure_floors(name, device)
        return 0
    if args.gemv:
        return 1 if check_gemv(device) else 0

    failed = 0
    for name in targets:
        results = check_op(name, candidate_fn=None, device=device) # candidate = reference (sanity)
        n_pass = sum(r.passed for r in results)
        failed += len(results) - n_pass
        print(f"{name}: {n_pass}/{len(results)} outputs within tolerance")
        for r in results:
            print(r.fmt())
    print(f"\n{'ALL GREEN' if failed == 0 else f'{failed} FAILURE(S)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
