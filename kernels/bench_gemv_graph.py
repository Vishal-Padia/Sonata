"""CUDA-graph timing bench: capture each kernel launch once, time replay.

The plain bench (bench_gemv.py) measures per-call Python/CuTe dispatch, which for
the fast small-N shapes dwarfs the kernel (NCU: out_proj M=2 kernel is 42 us but
the wall was 87 us). Under a CUDA graph the dispatch is captured once and replayed,
so this measures the kernel itself -- the fair cuBLAS-vs-CuTe comparison, and the
exact launch shape the Zonos decode loop needs (it runs under a graph too).
"""
import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from gemv import skinny_gemv_coalesced, skinny_gemv_splitk_m1, skinny_gemv_splitk_m2

SHAPES = {"out_proj": (2048, 4096), "in_proj": (8512, 2048), "fc1": (16384, 2048)}
BATCHES = [1, 2]
PEAK = 600.0


def measured_ceiling(nbytes=512 * 1024 * 1024):
    src = torch.empty(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    call = lambda: dst.copy_(src)
    return graph_time(call, iters=100, warmups=5) and (2 * nbytes) / (graph_time(call) * 1e-6) / 1e9


def graph_time(call, iters=300, warmups=5):
    """call() must only enqueue GPU work on the current stream (no host malloc/sync)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmups):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # us


def prep_cute(jit_fn, x, W, y):
    M, K = x.shape
    N = W.shape[0]
    xd, Wd, yd = from_dlpack(x), from_dlpack(W), from_dlpack(y)
    # The jit launcher now takes `stream` as its last arg and forwards it to
    # .launch(stream=...). Pass the CURRENT torch stream so the launch lands on the
    # capture stream (and skips the default-stream sync). Without it, CuTe launches
    # on stream 0 and syncs -> the kernel is never recorded (empty-graph replay).
    s = cutlass_torch.current_stream()
    compiled = cute.compile(jit_fn, yd, xd, Wd, M, N, K, s)   # compile ONCE, outside capture
    def call():
        compiled(yd, xd, Wd, M, N, K, cutlass_torch.current_stream())
    return call


def main():
    roof = 482.0  # measured copy ceiling from bench_gemv.py (~80% of 600 spec)
    print(f"# roofline ~{roof:.0f} GB/s (measured copy ceiling)")
    print(f"{'shape':10} {'M':>2} {'cand':>16} {'kernel us':>10} {'GB/s':>7} {'%roof':>6} {'vs cuBLAS':>10} {'ok':>5}")
    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        for M in BATCHES:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            truth = (x.float() @ W.float().t())

            cands = {}
            # cuBLAS: preallocated out, transposed-view rhs (both graph-safe)
            y_t = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            cands["torch"] = (lambda x=x, W=W, y=y_t: torch.matmul(x, W.t(), out=y), y_t)
            # CuTe: pick the M-specialized split-K kernel + the coalesced variant
            y_c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            splitk = skinny_gemv_splitk_m2 if M == 2 else skinny_gemv_splitk_m1
            cands["cute_splitk"] = (prep_cute(splitk, x, W, y_c), y_c)
            y_co = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            cands["cute_coalesced"] = (prep_cute(skinny_gemv_coalesced, x, W, y_co), y_co)

            t_cublas = None
            for label, (call, y) in cands.items():
                try:
                    us = graph_time(call)
                except Exception as e:
                    print(f"{name:10} {M:>2} {label:>16} {'CAPTURE FAIL':>10}  ({type(e).__name__}: {str(e)[:40]})")
                    continue
                err = (y.float() - truth).abs().max().item()
                rel = err / max(truth.abs().max().item(), 1e-12)
                gbps = (N * K + M * K + M * N) * 2 / (us * 1e-6) / 1e9
                if label == "torch":
                    t_cublas = us
                sp = f"{t_cublas / us:.2f}x" if t_cublas else "-"
                ok = "PASS" if rel < 5e-2 else "FAIL"
                print(f"{name:10} {M:>2} {label:>16} {us:10.2f} {gbps:7.0f} {100 * gbps / roof:6.1f} {sp:>10} {ok:>5}")


if __name__ == "__main__":
    main()
