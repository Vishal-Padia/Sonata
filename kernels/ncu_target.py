"""NCU target: run ONLY the out_proj split-K M=2 kernel (the shape with real
headroom that our kernel currently loses on). Warmup compiles + primes, then a
few steady-state launches for the profiler to capture (skip the warmup with -s)."""
import torch

from gemv import candidate_cute_splitk

N, K, M = 2048, 4096, 2  # out_proj, cfg-doubled
g = torch.Generator(device="cuda").manual_seed(0)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)

for _ in range(15):          # compile + warmup (skip these in ncu)
    candidate_cute_splitk(x, W)
torch.cuda.synchronize()

for _ in range(3):           # the launches we actually profile
    candidate_cute_splitk(x, W)
torch.cuda.synchronize()
