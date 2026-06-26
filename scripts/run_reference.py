import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow importing sibling scripts
from fake_quant import apply_fake_quant

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=421, help="RNG seed; the golden contract pins 421")
parser.add_argument("--out", default="baseline.wav")
parser.add_argument("--greedy", action="store_true",
                    help="deterministic greedy decode (temperature=0, argmax); the calibratable control")
parser.add_argument("--quant-bits", type=int, default=None, choices=[4, 8],
                    help="enable weight-only fake quant (4 or 8); off by default (clean reference)")
parser.add_argument("--quant-group-size", type=int, default=None,
                    help="K-group size for fake quant; None = per-output-channel (one scale per row)")
parser.add_argument("--quant-include-attn", action="store_true",
                    help="also quantize attention-layer projections (default: Mamba layers only)")
parser.add_argument("--quant-target", choices=["both", "in_proj", "out_proj"], default="both",
                    help="which Mamba projection(s) to fake-quant; lets you localize sensitivity")
args = parser.parse_args()

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

torch.manual_seed(args.seed)
print(f"run_reference: seed={args.seed} device={device}")

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)

wav, sampling_rate = torchaudio.load("third_party/assets/exampleaudio.mp3")

speaker = model.make_speaker_embedding(wav, sampling_rate)

cond = make_cond_dict(
    text="The kernels run on time, frame by frame.",
    speaker=speaker,
    language="en-us",
)
conditioning = model.prepare_conditioning(cond)

if args.quant_bits is not None:
    # Never let a fake-quant run clobber the clean golden source (baseline.wav).
    if args.out == parser.get_default("out"):
        raise SystemExit("refusing to write fake-quant audio to the default baseline.wav; "
                         "pass an explicit --out (e.g. --out cand_int8.wav)")
    apply_fake_quant(model, bits=args.quant_bits, group_size=args.quant_group_size,
                     mamba_only=not args.quant_include_attn, target=args.quant_target)

# temperature=0 -> sample_from_logits takes argmax (deterministic greedy control).
sampling_params = dict(temperature=0.0) if args.greedy else dict(min_p=0.1)
codes = model.generate(conditioning, sampling_params=sampling_params)
wav_out = model.autoencoder.decode(codes).cpu()
torchaudio.save(args.out, wav_out[0], model.autoencoder.sampling_rate)
print(f"run_reference: wrote {args.out} ({wav_out.shape[-1]} samples @ {model.autoencoder.sampling_rate} Hz)")