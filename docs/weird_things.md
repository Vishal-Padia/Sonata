### Installation Shenanigans

At first I created a env, and installed basic libraries like torch, torchaudio, etc. But then I tried to run this command to install the dependencies from zonos repository:

```bash
uv pip install --no-build-isolation -e .[compile]
```
Which basically OOM-killed my machine lol. So I thought I'll try to get pre-built wheels from github repos and use that, but then the issue of version mismatch made me bash my head against the wall. I gave up doing it manually and just ask Claude to help me with the installation.

So here's how I installed it:
```bash
uv venv venv --python 3.12

source venv/bin/activate

uv pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

cat > /home/ubuntu/Sonata/constraints.txt <<'EOF'
torch==2.8.0
torchaudio==2.8.0
triton<3.5
EOF

uv pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

uv pip install --no-deps \
  https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

uv pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


uv pip install --constraint /home/ubuntu/Sonata/constraints.txt \
  einops ninja

uv pip install --constraint /home/ubuntu/Sonata/constraints.txt \
  -e /home/ubuntu/Sonata/third_party

python - <<'PY'
import torch, flash_attn, mamba_ssm, causal_conv1d
print("torch        :", torch.__version__, "cuda:", torch.cuda.is_available())
print("flash_attn   :", flash_attn.__version__)
print("mamba_ssm    :", mamba_ssm.__version__)
print("causal_conv1d:", causal_conv1d.__version__)

from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from flash_attn import flash_attn_func
print("trio symbol resolution ok")
PY
```

Don't know what would I have done if I didn't have Claude to help me with the installation lol

### Running the Baseline Reference Audio
So basically I wrote `scripts/run_reference.py` to get a baseline reference audio before I start to work on kernels and all, I took `sample.py` from zonos repository and tried to run it, and it gave me this error:
```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

And to be honest I was confused because I had use the same code given in the zonos repository. But one thing I didn't notice is that I was using `torchaudio` version `2.8.0` and `zonos` was using `2.7.0`. So I added a monkey patch to `torchaudio` to make it work with `2.8.0`.

```python
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
```
After adding this monkey patch, I was able to run the code without any errors and I got the baseline reference audio.