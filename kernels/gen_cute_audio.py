"""Generate audio with the Mamba2 projections swapped to the CuTe GEMV, for an
A/B listen against baseline.wav. The teacher-forced flip probe shows ~5% token
divergence, but the kernel is >= cuBLAS accuracy vs fp32, so those flips are the
model's brittleness (a different-but-valid rollout), not a quality regression.
The only honest quality check is the ear: does this sound clean?"""
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))

_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    def __init__(self, *a, **k):
        d = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*a, **k)
        if d.type != "cpu":
            self.to(d)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from patch import patch_zonos_projections

torch.manual_seed(421)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model.can_use_cudagraphs = lambda: False  # eager (graph integration is the next phase)

wav, sr = torchaudio.load("third_party/assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sr)
cond = make_cond_dict(text="The kernels run on time, frame by frame.", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond)

n = patch_zonos_projections(model, include_attn=True)
print(f"patched {n} projections (Mamba2 + attention) -> CuTe GEMV")

codes = model.generate(conditioning, disable_torch_compile=True)
out = model.autoencoder.decode(codes).cpu()
torchaudio.save("cute_stream.wav", out[0], model.autoencoder.sampling_rate)
print(f"wrote cute_stream.wav ({out.shape[-1] / model.autoencoder.sampling_rate:.2f}s) -- A/B against baseline.wav")
