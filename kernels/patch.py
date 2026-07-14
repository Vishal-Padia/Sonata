"""Swap Zonos's decode-shaped projection Linears to the CuTeDSL skinny-GEMV.

Monkeypatches the live nn.Linear modules after load (leaves third_party untouched).
Graph-safe: the kernel is pre-compiled once per (M,N,K) at patch time, the launch
reuses a pinned output buffer + prebound weight dlpack, and runs on the current
(capture) stream -- so it can sit inside Zonos's decode CUDA graph.

Prefill trap: the projections also run at M = prefill seqlen (a real GEMM where
cuBLAS is optimal). The patched forward only routes M in {1,2} (2D input) to the
GEMV and falls back to the original nn.Linear otherwise, so prefill is untouched.
"""
import sys
from pathlib import Path

import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemv import skinny_gemv_splitk_m1, skinny_gemv_splitk_m2

# Compiled kernels are shape-specialized but tensor-agnostic, so one per (M,N,K)
# serves every layer of that shape (34 mamba layers share 2 projection shapes).
_KCACHE: dict[tuple[int, int, int], object] = {}


def _compiled(M: int, N: int, K: int, device):
    key = (M, N, K)
    if key not in _KCACHE:
        jit = skinny_gemv_splitk_m2 if M == 2 else skinny_gemv_splitk_m1
        y = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        x = torch.empty(M, K, device=device, dtype=torch.bfloat16)
        W = torch.empty(N, K, device=device, dtype=torch.bfloat16)
        _KCACHE[key] = cute.compile(
            jit, from_dlpack(y), from_dlpack(x), from_dlpack(W), M, N, K,
            cutlass_torch.current_stream(),
        )
    return _KCACHE[key]


def patch_linear(lin: torch.nn.Linear, batches=(1, 2)):
    """Route decode-shaped (M in `batches`) calls of `lin` through the GEMV kernel."""
    assert lin.bias is None, "GEMV kernel has no bias path"
    W = lin.weight.detach()  # (N, K) bf16, row-major; detach so dlpack can export the Parameter
    N, K = W.shape
    device = W.device
    comp = {M: _compiled(M, N, K, device) for M in batches}
    ybuf = {M: torch.empty(M, N, device=device, dtype=torch.bfloat16) for M in batches}
    Wd = from_dlpack(W)                                   # weight is fixed -> bind once
    yd = {M: from_dlpack(ybuf[M]) for M in batches}       # output buffers pinned -> bind once
    orig = lin.forward

    def fwd(x):
        # Mamba step() feeds 2D (M, K); attention (MHA/MLP) feeds 3D (M, 1, K) at
        # decode. Route both single-token shapes to the GEMV; anything else
        # (prefill seqlen>1, unexpected M) falls back to cuBLAS.
        three_d = x.dim() == 3 and x.shape[1] == 1
        if x.dim() == 2 or three_d:
            x2 = x.squeeze(1) if three_d else x
            M = int(x2.shape[0])
            if M in comp and x2.shape[1] == K:
                xc = x2.detach().contiguous()             # kernel expects contiguous (M,K); detach for dlpack
                comp[M](yd[M], from_dlpack(xc), Wd, M, N, K, cutlass_torch.current_stream())
                return ybuf[M].unsqueeze(1) if three_d else ybuf[M]
        return orig(x)                                    # prefill / unsupported -> cuBLAS

    lin.forward = fwd
    lin._gemv_orig = orig
    return lin


def unpatch_linear(lin: torch.nn.Linear):
    if hasattr(lin, "_gemv_orig"):
        lin.forward = lin._gemv_orig
        del lin._gemv_orig


def patch_zonos_projections(model, include_attn=False, verbose=False) -> int:
    """Patch decode-shaped, bias-free projection Linears across the backbone:
    Mamba2 in_proj/out_proj always; the attention qkv/out_proj and MLP fc1/fc2 when
    include_attn=True. Returns the number of Linears patched."""
    attn = set(model.backbone.config.attn_layer_idx)
    targets = ("in_proj", "out_proj", "fc1", "fc2")
    n = 0
    for name, mod in model.backbone.named_modules():
        if not (isinstance(mod, torch.nn.Linear) and name.endswith(targets)):
            continue
        if mod.bias is not None:            # GEMV has no bias path
            continue
        idx = next((int(p) for p in name.split(".") if p.isdigit()), None)
        if (idx in attn) and not include_attn:
            continue
        patch_linear(mod)
        n += 1
        if verbose:
            print(f"  patched {name}: {tuple(mod.weight.shape)}")
    return n
