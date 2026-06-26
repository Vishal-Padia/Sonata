"""Weight-only fake quantization helpers.

Factored out of run_reference.py so the same code can be imported by the
teacher-forced logit probe (importing run_reference would execute its top-level
generation pipeline as a side effect). "Fake" quant = quantize then dequantize
back to the original dtype, so a normal bf16 matmul reproduces exactly the
numerics a real weight-only integer kernel would emit. No speedup; this measures
the quality ceiling only.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def fake_quantize_weights(w: torch.Tensor, bits: int, group_size: int | None) -> torch.Tensor:
    """Simulate weight-only symmetric integer quantization.

    w: (out_features, in_features). Scale is per output row, per K-group.
    group_size=None -> one scale per row (per-output-channel).

    Symmetric signed: qmax = 2**(bits-1) - 1 (127 for int8, 7 for int4). Scales
    are accumulated in fp32 then the dequantized weight is cast back to w.dtype,
    preserving device/contiguity so it can drop straight into mod.weight.data.
    """
    orig_dtype = w.dtype
    w = w.float()  # accumulate scales in fp32, never bf16
    out_f, in_f = w.shape
    if group_size is None:
        group_size = in_f  # single group == per-output-channel
    assert in_f % group_size == 0, f"in_features {in_f} not divisible by group_size {group_size}"

    qmax = (1 << (bits - 1)) - 1  # 127 (int8) / 7 (int4), symmetric signed
    wg = w.view(out_f, in_f // group_size, group_size)

    amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # avoid /0 on dead channels
    scale = amax / qmax  # fp32 scale per (row, group)

    q = torch.round(wg / scale).clamp_(-qmax, qmax)
    wdq = (q * scale).view(out_f, in_f)
    return wdq.to(orig_dtype)


@torch.no_grad()
def apply_fake_quant(model, bits: int, group_size: int | None, mamba_only: bool = True,
                     target: str = "both", verbose: bool = True) -> list[int | None]:
    """Patch projection GEMV weights in-place with fake-quantized versions.

    Walk is scoped to model.backbone so Linears named in_proj/out_proj elsewhere
    (autoencoder, conditioner) can never be touched by accident. With mamba_only,
    attention-layer projections are skipped via config.attn_layer_idx. `target`
    restricts to in_proj / out_proj / both, to localize where quant hurts.

    Returns the list of layer indices patched (for logging/inspection).
    """
    attn_idx = set(model.backbone.config.attn_layer_idx)
    want = ("in_proj", "out_proj") if target == "both" else (target,)
    patched: list[int | None] = []
    for name, mod in model.backbone.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not name.endswith(want):
            continue
        parts = name.split(".")
        layer_idx = next((int(p) for p in parts if p.isdigit()), None)
        if mamba_only and layer_idx in attn_idx:
            continue
        mod.weight.data = fake_quantize_weights(mod.weight.data, bits, group_size)
        patched.append(layer_idx)
    if verbose:
        idxs = sorted({i for i in patched if i is not None})
        print(f"fake-quant applied: bits={bits} group_size={group_size} target={target} "
              f"mamba_only={mamba_only} -> {len(patched)} linears across {len(idxs)} layers {idxs}")
    return patched
