import sys
import time

import torch

from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

SEED = 421
FRAMES = 228


def bench(fn, iters=50, warm=10):
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
    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    ae = model.autoencoder
    sr, hop = ae.sampling_rate, ae.dac.config.hop_length
    codes = torch.randint(0, ae.codebook_size, (1, ae.num_codebooks, FRAMES), device=device)
    audio_s = FRAMES * hop / sr

    decode = lambda: ae.decode(codes)
    ref = decode().clone()
    t_eager = bench(decode)
    print(f"eager    : {t_eager:6.2f} ms/call  RTF {audio_s / (t_eager / 1e3):5.0f}x", flush=True)

    # compile the decoder module (inductor fuses the pointwise Snake ops)
    for mode in ("default", "max-autotune"):
        ae.dac.decoder = torch.compile(ae.dac.decoder, mode=None if mode == "default" else mode)
        for _ in range(8):  # first calls compile (slow); discard
            decode()
        torch.cuda.synchronize()
        err = (ref - decode()).abs().max().item()
        t = bench(decode)
        print(f"compiled ({mode:12}): {t:6.2f} ms/call  RTF {audio_s / (t / 1e3):5.0f}x  "
              f"speedup {t_eager / t:.2f}x  max_err {err:.2e}", flush=True)
        ae.dac.decoder = ae.dac.decoder._orig_mod  # reset for the next mode


if __name__ == "__main__":
    sys.exit(main())
