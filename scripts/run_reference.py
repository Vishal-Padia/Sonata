import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

_OriginalMelSpec = torchaudio.transforms.MelSpectrogram
class _DeviceSafeMelSpec(_OriginalMelSpec):
    """Workaround for torchaudio>=2.7 filterbank precomputation bug
    inside a non-CPU torch.device() context. Construct on CPU, then
    move all registered buffers to wherever the outer context wanted."""
    def __init__(self, *args, **kwargs):
        outer_device = torch.empty(0).device
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if outer_device.type != "cpu":
            self.to(outer_device)
torchaudio.transforms.MelSpectrogram = _DeviceSafeMelSpec

torch.manual_seed(421)

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)

wav, sampling_rate = torchaudio.load("third_party/assets/exampleaudio.mp3")

speaker = model.make_speaker_embedding(wav, sampling_rate)

cond = make_cond_dict(
    text="The kernels run on time, frame by frame.",
    speaker=speaker,
    language="en-us",
)
conditioning = model.prepare_conditioning(cond)

codes = model.generate(conditioning)
wav_out = model.autoencoder.decode(codes).cpu()
torchaudio.save("baseline.wav", wav_out[0], model.autoencoder.sampling_rate)