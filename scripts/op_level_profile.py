import sys
from torch.profiler import ProfilerActivity, profile, record_function

import torch
import torchaudio

_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *args, **kwargs):
        outer_device = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if outer_device.type != "cpu":
            self.to(outer_device)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

from zonos.config import InferenceParams
from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device

SEED = 421
LAYER_IDX = 2 # a "typical" 833 us mamba2 layer (not the cheaper post-attn 717 us ones)
BATCH = 2 # cfg-doubled decode batch
WARMUP = 40 # cover triton autotune (selective_state_update, layernorm_gated)
PROFILE_ITERS = 200
SEQLEN_OFFSET = 100 # >0 => Mamba2.forward dispatches to step()


@torch.inference_mode()
def main():
    gpu = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"# GPU: {gpu} (sm{cc[0]}{cc[1]})  torch {torch.__version__}  layer_idx={LAYER_IDX} batch={BATCH}")

    torch.manual_seed(SEED)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    model.eval()

    layer = model.backbone.layers[LAYER_IDX]
    mixer = layer.mixer
    assert type(mixer).__name__ == "Mamba2", f"layer {LAYER_IDX} is {type(mixer).__name__}, pick a mamba2 layer"
    layer_idx = mixer.layer_idx

    d_model = model.config.backbone.d_model
    conv_state, ssm_state = mixer.allocate_inference_cache(BATCH, 2048, dtype=torch.bfloat16)
    ip = InferenceParams(
        max_seqlen=2048, max_batch_size=BATCH, seqlen_offset=SEQLEN_OFFSET, batch_size_offset=0,
        key_value_memory_dict={layer_idx: (conv_state, ssm_state)},
        lengths_per_sample=torch.full((BATCH,), SEQLEN_OFFSET, dtype=torch.int32, device=device),
    )
    hidden = torch.randn(BATCH, 1, d_model, dtype=torch.bfloat16, device=device)
    residual = torch.randn(BATCH, 1, d_model, dtype=torch.bfloat16, device=device)

    # sanity: confirm the step() path is taken
    h, r = layer(hidden, residual, ip)
    assert h.shape == hidden.shape

    def run_block():
        for _ in range(PROFILE_ITERS):
            with record_function("block.forward"):
                layer(hidden, residual, ip)

    def run_mixer():
        for _ in range(PROFILE_ITERS):
            with record_function("mixer.step"):
                mixer.step(hidden, conv_state, ssm_state)

    # warmup (both paths)
    for _ in range(WARMUP):
        layer(hidden, residual, ip)
        mixer.step(hidden, conv_state, ssm_state)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        run_block()
        run_mixer()
        torch.cuda.synchronize()

    print("\n================ TOP CUDA KERNELS (self CUDA time, summed over "
          f"{PROFILE_ITERS} iters of EACH of block+mixer) ================")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # total device time per iter for the two regions, via the record_function rows
    ka = {e.key: e for e in prof.key_averages()}
    def per_iter_us(key):
        e = ka.get(key)
        if e is None:
            return float("nan")
        # device_time_total is the wall of the labeled region's CUDA work
        t = getattr(e, "cuda_time_total", None)
        if t is None:
            t = getattr(e, "device_time_total", 0.0)
        return t / PROFILE_ITERS
    print("\n================ PER-ITER DEVICE TIME (us) ================")
    print(f"block.forward (norm + mixer.step) : {per_iter_us('block.forward'):.1f} us/iter")
    print(f"mixer.step    (mamba2 ops only)   : {per_iter_us('mixer.step'):.1f} us/iter")
    print(f"=> entry fused add-norm (block - mixer) ~ "
          f"{per_iter_us('block.forward') - per_iter_us('mixer.step'):.1f} us/iter")

    trace = "docs/op_level_trace.json"
    prof.export_chrome_trace(trace)
    print(f"\n# chrome trace -> {trace}")


if __name__ == "__main__":
    sys.exit(main())
