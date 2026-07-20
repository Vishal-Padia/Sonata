"""Fused snake+conv (naive per-output kernel) vs separate snake -> cuDNN conv,
at the REAL DAC decoder shapes (the high-sample-rate late blocks that dominate the
profile). This quantifies the cost of recomputing Snake ~out_channels*K times per
input element in the current fused kernel, so we know where it lands before the
tiled (snake-once) rewrite.

Reproduce: python kernels/bench_snake_conv.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_snake_conv import fused_snake_conv1d, reference_snake_conv1d

# (in_ch, out_ch, seq_len, kernel, dilation). Seq lengths for a 228-frame clip:
# latent 228 -> x8 -> x8 -> x4 (block.2 = 228*256) -> x2 (block.3 = 228*512).
SHAPES = {
    "block.2 res1 (C=192, L=58368, K7 d1)":  (192, 192, 58368, 7, 1),
    "block.2 res3 (C=192, L=58368, K7 d9)":  (192, 192, 58368, 7, 9),
    "block.3 res1 (C=96,  L=116736, K7 d1)": (96, 96, 116736, 7, 1),
    "block.3 res3 (C=96,  L=116736, K7 d9)": (96, 96, 116736, 7, 9),
}


def bench(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    print(f"# {torch.cuda.get_device_name(0)}  fp32  (snake->cuDNN conv is the baseline to beat)")
    print(f"{'shape':38} {'fused ms':>9} {'separate ms':>12} {'speedup':>8} {'max_err':>9}")
    for name, (cin, cout, L, K, d) in SHAPES.items():
        pad = ((K - 1) * d) // 2
        x = torch.randn(1, cin, L, device="cuda", dtype=torch.float32)
        alpha = torch.rand(cin, device="cuda", dtype=torch.float32) * 2 + 0.3
        W = torch.randn(cout, cin, K, device="cuda", dtype=torch.float32)
        b = torch.randn(cout, device="cuda", dtype=torch.float32)

        fused = lambda: fused_snake_conv1d(x, alpha, W, b, padding=pad, dilation=d)
        sep = lambda: reference_snake_conv1d(x, alpha, W, b, padding=pad, dilation=d)
        err = (fused().float() - sep().float()).abs().max().item()
        t_f = bench(fused)
        t_s = bench(sep)
        print(f"{name:38} {t_f:9.2f} {t_s:12.3f} {t_s / t_f:7.2f}x {err:9.2e}")
    print("\nspeedup > 1 => fused wins; < 1 => the Snake recompute is costing more than the HBM pass it saves.")


if __name__ == "__main__":
    sys.exit(main())
