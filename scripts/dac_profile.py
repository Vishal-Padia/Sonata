import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

SEED = 421
FRAMES = 228 # ~2.6 s clip, matches our streaming tests
WARMUP = 20
ITERS = 50


def wrap_region(module, name):
    """Wrap a submodule's forward in a record_function so the profiler attributes
    time per decoder block / stem / head."""
    orig = module.forward

    def fwd(*a, **k):
        with record_function(name):
            return orig(*a, **k)

    module.forward = fwd


@torch.inference_mode()
def main():
    torch.manual_seed(SEED)
    gpu = torch.cuda.get_device_name(0)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    ae = model.autoencoder
    dec = ae.dac.decoder
    sr, hop = ae.sampling_rate, ae.dac.config.hop_length
    print(f"# GPU {gpu}  frames={FRAMES}  hop={hop}  sr={sr}")

    codes = torch.randint(0, ae.codebook_size, (1, ae.num_codebooks, FRAMES), device=device)

    # label the stem, each upsampling block, and the head for per-region attribution
    wrap_region(dec.conv1, "conv1 (stem)")
    for i, blk in enumerate(dec.block):
        wrap_region(blk, f"block.{i}")
    wrap_region(dec.snake1, "snake1 (head)")
    wrap_region(dec.conv2, "conv2 (head)")

    def decode():
        return ae.decode(codes)

    for _ in range(WARMUP):
        decode()
    torch.cuda.synchronize()

    # wall-clock total + RTF
    t0 = time.perf_counter()
    for _ in range(ITERS):
        decode()
    torch.cuda.synchronize()
    per_call_ms = (time.perf_counter() - t0) / ITERS * 1000
    audio_s = FRAMES * hop / sr
    print(f"decode: {per_call_ms:.2f} ms/call  audio={audio_s:.3f}s  RTF={audio_s / (per_call_ms/1000):.1f}x")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            decode()
        torch.cuda.synchronize()

    ka = prof.key_averages()
    def region_ms(key):
        e = next((x for x in ka if x.key == key), None)
        t = getattr(e, "cuda_time_total", None) or getattr(e, "device_time_total", 0.0) if e else 0.0
        return t / ITERS / 1000  # ms/call

    print("\n=== per region (device time, ms/call) ===")
    total = 0.0
    for key in ["conv1 (stem)", "block.0", "block.1", "block.2", "block.3", "snake1 (head)", "conv2 (head)"]:
        ms = region_ms(key)
        total += ms
        print(f"  {key:16} {ms:7.3f} ms")
    print(f"  {'sum':16} {total:7.3f} ms")

    print("\n=== top CUDA kernels (self device time, ms/call) ===")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))


if __name__ == "__main__":
    sys.exit(main())
