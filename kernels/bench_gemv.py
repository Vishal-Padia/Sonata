import torch

from gemv import candidate_cute, candidate_cute_coalesced, candidate_cute_splitk

SHAPES = {
    "out_proj": (2048, 4096), # small-N: the split-k case later
    "in_proj": (8512, 2048),
    "fc1": (16384, 2048), # large-N
}

BATCHES = [1, 2] # M: 1 = true streaming, 2 = cfg-doubled
PEAK_GBPS = 600.0 # a10g spec
REL_TOL = 5e-2   # ~3x the measured bf16 output-rounding floor; magnitude-robust

def candidate_torch(x, W):
    return torch.matmul(x, W.t())

# Candidates share the (x, W) -> y contract so the harness treats them uniformly.
CANDIDATES = {
    "torch": candidate_torch,
    "cute":  candidate_cute,
    "cute_coalesced": candidate_cute_coalesced,
    "cute_splitk": candidate_cute_splitk,
}

def time_us(fn, x, W, iters=200, warmup=50, repeats=7):
    for _ in range(warmup): fn(x, W)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters): fn(x, W)   # batch stays pipelined (graph-like)
        end.record(); torch.cuda.synchronize()   # one sync per batch
        times.append(start.elapsed_time(end) * 1e3 / iters)  # us/call
    return float(torch.tensor(times).median())

def gbps(N, K, M, t_us):
    bytes_moved = (N*K + M*K + M*N) * 2      # bf16 = 2 bytes; W read + x read + y write
    return bytes_moved / (t_us * 1e-6) / 1e9

def make_inputs(N, K, M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, W

def correctness(candidate, x, W):
    truth = (x.float() @ W.float().t()) # fp32 ground truth
    out   = candidate(x, W).float()
    abs_err = (out - truth).abs().max().item()
    # Relative to output magnitude: the bf16 floor scales with sqrt(K), so an
    # absolute tolerance would false-fail the large-K shapes. See docs/validation.md.
    rel_err = abs_err / max(truth.abs().max().item(), 1e-12)
    return abs_err, rel_err

def measured_ceiling(nbytes=512*1024*1024):
    src = torch.empty(nbytes//2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    def cp(a, b): dst.copy_(src)
    t = time_us(cp, None, None)
    return 2*nbytes / (t*1e-6) / 1e9 # copy = 1 read + 1 write

def main():
    roof = measured_ceiling()
    print(f"# measured HBM ceiling: {roof:.0f} GB/s ({100*roof/PEAK_GBPS:.0f}% of spec)")
    print(f"{'shape':10} {'M':>2} {'cand':>6} {'us':>9} {'GB/s':>7} {'%roof':>6} {'maxerr':>9} {'ok':>5}")
    for name, (N, K) in SHAPES.items():
        for M in BATCHES:
            x, W = make_inputs(N, K, M)
            for label, fn in CANDIDATES.items():
                abs_err, rel_err = correctness(fn, x, W)
                t = time_us(fn, x, W)
                bw = gbps(N, K, M, t)
                ok = "PASS" if rel_err < REL_TOL else "FAIL"
                print(f"{name:10} {M:>2} {label:>6} {t:9.1f} {bw:7.0f} {100*bw/roof:6.1f} {abs_err:9.2e} {ok:>5}")

if __name__ == "__main__":
    main()