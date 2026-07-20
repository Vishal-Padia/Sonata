# DAC Architecture

```python
DacConfig {
  "architectures": [
    "DacModel"
  ],
  "codebook_dim": 8,
  "codebook_loss_weight": 1.0,
  "codebook_size": 1024,
  "commitment_loss_weight": 0.25,
  "decoder_hidden_size": 1536,
  "downsampling_ratios": [
    2,
    4,
    8,
    8
  ],
  "dtype": "float32",
  "encoder_hidden_size": 64,
  "hidden_size": 1024,
  "hop_length": 512,
  "model_type": "dac",
  "n_codebooks": 9,
  "quantizer_dropout": 0.0,
  "sampling_rate": 44100,
  "transformers_version": "5.10.2",
  "upsampling_ratios": [
    8,
    8,
    4,
    2
  ]
}
```
So from the config, we can see that the upsampling ratios are [8, 8, 4, 2], the decoder hidden size is 1536, and the decoder dimension is 1024. We can also see that the encoder hidden size is 64, and the codebook dimension is 8. We can also see that the hop length is 512, and the sampling rate is 44100. We can also see that the number of codebooks is 9, and the quantizer dropout is 0.0


```python
DacDecoder(
  (conv1): Conv1d(1024, 1536, kernel_size=(7,), stride=(1,), padding=(3,))
  (block): ModuleList(
    (0): DacDecoderBlock(
      (snake1): Snake1d()
      (conv_t1): ConvTranspose1d(1536, 768, kernel_size=(16,), stride=(8,), padding=(4,))
      (res_unit1): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(768, 768, kernel_size=(7,), stride=(1,), padding=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(768, 768, kernel_size=(1,), stride=(1,))
      )
      (res_unit2): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(768, 768, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(768, 768, kernel_size=(1,), stride=(1,))
      )
      (res_unit3): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(768, 768, kernel_size=(7,), stride=(1,), padding=(27,), dilation=(9,))
        (snake2): Snake1d()
        (conv2): Conv1d(768, 768, kernel_size=(1,), stride=(1,))
      )
    )
    (1): DacDecoderBlock(
      (snake1): Snake1d()
      (conv_t1): ConvTranspose1d(768, 384, kernel_size=(16,), stride=(8,), padding=(4,))
      (res_unit1): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(384, 384, kernel_size=(7,), stride=(1,), padding=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(384, 384, kernel_size=(1,), stride=(1,))
      )
      (res_unit2): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(384, 384, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(384, 384, kernel_size=(1,), stride=(1,))
      )
      (res_unit3): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(384, 384, kernel_size=(7,), stride=(1,), padding=(27,), dilation=(9,))
        (snake2): Snake1d()
        (conv2): Conv1d(384, 384, kernel_size=(1,), stride=(1,))
      )
    )
    (2): DacDecoderBlock(
      (snake1): Snake1d()
      (conv_t1): ConvTranspose1d(384, 192, kernel_size=(8,), stride=(4,), padding=(2,))
      (res_unit1): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(192, 192, kernel_size=(7,), stride=(1,), padding=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
      (res_unit2): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(192, 192, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
      (res_unit3): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(192, 192, kernel_size=(7,), stride=(1,), padding=(27,), dilation=(9,))
        (snake2): Snake1d()
        (conv2): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (3): DacDecoderBlock(
      (snake1): Snake1d()
      (conv_t1): ConvTranspose1d(192, 96, kernel_size=(4,), stride=(2,), padding=(1,))
      (res_unit1): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(96, 96, kernel_size=(7,), stride=(1,), padding=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(96, 96, kernel_size=(1,), stride=(1,))
      )
      (res_unit2): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(96, 96, kernel_size=(7,), stride=(1,), padding=(9,), dilation=(3,))
        (snake2): Snake1d()
        (conv2): Conv1d(96, 96, kernel_size=(1,), stride=(1,))
      )
      (res_unit3): DacResidualUnit(
        (snake1): Snake1d()
        (conv1): Conv1d(96, 96, kernel_size=(7,), stride=(1,), padding=(27,), dilation=(9,))
        (snake2): Snake1d()
        (conv2): Conv1d(96, 96, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (snake1): Snake1d()
  (conv2): Conv1d(96, 1, kernel_size=(7,), stride=(1,), padding=(3,))
  (tanh): Tanh()
)
```

Upsample strides: 8 × 8 × 4 × 2 = 512 (= hop_length).
All pads below are symmetric / SAME-style (centered), not causal.

| # | module path | type | in→out | kernel | stride | dilation | padding |
|---|---|---|---|---|---|---|---|
| 1 | `decoder.conv1` | Conv1d | 1024→1536 | 7 | 1 | 1 | 3 |
| 2 | `decoder.block.0.conv_t1` | ConvTranspose1d | 1536→768 | 16 | 8 | 1 | 4 |
| 3 | `decoder.block.0.res_unit1.conv1` | Conv1d | 768→768 | 7 | 1 | 1 | 3 |
| 4 | `decoder.block.0.res_unit1.conv2` | Conv1d | 768→768 | 1 | 1 | 1 | 0 |
| 5 | `decoder.block.0.res_unit2.conv1` | Conv1d | 768→768 | 7 | 1 | 3 | 9 |
| 6 | `decoder.block.0.res_unit2.conv2` | Conv1d | 768→768 | 1 | 1 | 1 | 0 |
| 7 | `decoder.block.0.res_unit3.conv1` | Conv1d | 768→768 | 7 | 1 | 9 | 27 |
| 8 | `decoder.block.0.res_unit3.conv2` | Conv1d | 768→768 | 1 | 1 | 1 | 0 |
| 9 | `decoder.block.1.conv_t1` | ConvTranspose1d | 768→384 | 16 | 8 | 1 | 4 |
| 10 | `decoder.block.1.res_unit1.conv1` | Conv1d | 384→384 | 7 | 1 | 1 | 3 |
| 11 | `decoder.block.1.res_unit1.conv2` | Conv1d | 384→384 | 1 | 1 | 1 | 0 |
| 12 | `decoder.block.1.res_unit2.conv1` | Conv1d | 384→384 | 7 | 1 | 3 | 9 |
| 13 | `decoder.block.1.res_unit2.conv2` | Conv1d | 384→384 | 1 | 1 | 1 | 0 |
| 14 | `decoder.block.1.res_unit3.conv1` | Conv1d | 384→384 | 7 | 1 | 9 | 27 |
| 15 | `decoder.block.1.res_unit3.conv2` | Conv1d | 384→384 | 1 | 1 | 1 | 0 |
| 16 | `decoder.block.2.conv_t1` | ConvTranspose1d | 384→192 | 8 | 4 | 1 | 2 |
| 17 | `decoder.block.2.res_unit1.conv1` | Conv1d | 192→192 | 7 | 1 | 1 | 3 |
| 18 | `decoder.block.2.res_unit1.conv2` | Conv1d | 192→192 | 1 | 1 | 1 | 0 |
| 19 | `decoder.block.2.res_unit2.conv1` | Conv1d | 192→192 | 7 | 1 | 3 | 9 |
| 20 | `decoder.block.2.res_unit2.conv2` | Conv1d | 192→192 | 1 | 1 | 1 | 0 |
| 21 | `decoder.block.2.res_unit3.conv1` | Conv1d | 192→192 | 7 | 1 | 9 | 27 |
| 22 | `decoder.block.2.res_unit3.conv2` | Conv1d | 192→192 | 1 | 1 | 1 | 0 |
| 23 | `decoder.block.3.conv_t1` | ConvTranspose1d | 192→96 | 4 | 2 | 1 | 1 |
| 24 | `decoder.block.3.res_unit1.conv1` | Conv1d | 96→96 | 7 | 1 | 1 | 3 |
| 25 | `decoder.block.3.res_unit1.conv2` | Conv1d | 96→96 | 1 | 1 | 1 | 0 |
| 26 | `decoder.block.3.res_unit2.conv1` | Conv1d | 96→96 | 7 | 1 | 3 | 9 |
| 27 | `decoder.block.3.res_unit2.conv2` | Conv1d | 96→96 | 1 | 1 | 1 | 0 |
| 28 | `decoder.block.3.res_unit3.conv1` | Conv1d | 96→96 | 7 | 1 | 9 | 27 |
| 29 | `decoder.block.3.res_unit3.conv2` | Conv1d | 96→96 | 1 | 1 | 1 | 0 |
| 30 | `decoder.conv2` | Conv1d | 96→1 | 7 | 1 | 1 | 3 |

Padding formulas (from source, matches the dump):
- residual `conv1`: `pad = ((7 - 1) * dilation) // 2` → 3 / 9 / 27
- stem / final `Conv1d(k=7)`: `padding=3`
- `ConvTranspose1d`: `kernel=2*stride`, `padding=ceil(stride/2)` → 4 / 4 / 2 / 1


1. **Upsampling stack (ConvTranspose1d strides)**

| block | ConvTranspose1d | stride s | kernel (=2s) | padding (=ceil(s/2)) | in→out ch | rate after block |
|-------|-----------------|----------|--------------|----------------------|-----------|------------------|
| 0 | `block.0.conv_t1` | 8 | 16 | 4 | 1536→768 | latent ×8 |
| 1 | `block.1.conv_t1` | 8 | 16 | 4 | 768→384  | ×8 again |
| 2 | `block.2.conv_t1` | 4 | 8  | 2 | 384→192  | ×4 |
| 3 | `block.3.conv_t1` | 2 | 4  | 1 | 192→96   | ×2 |

Product of strides: **8 × 8 × 4 × 2 = 512 = hop_length**. This is correct as the hop length is 512.

Samples of audio per feature at each stage (decode order):

| stage | location | samples / timestep |
|-------|----------|--------------------|
| latent | after `from_codes`, into `conv1` | 512 |
| after block 0 | into block 1 | 64 |
| after block 1 | into block 2 | 8 |
| after block 2 | into block 3 | 2 |
| after block 3 | into final `snake1`/`conv2` | 1 (44.1 kHz) |

One latent frame → 512 waveform samples. That is the vocoder’s frame clock.

2. **Causal or centered? (per layer)**

Padding is computed as **symmetric / SAME**, not causal left-pad.

| layer family | padding formula | L pad | R pad | verdict |
|--------------|-----------------|-------|-------|---------|
| stem `conv1`, final `conv2` (K=7, d=1) | hard-coded `padding=3` | 3 | 3 | **centered** |
| residual `conv1` (K=7, d∈{1,3,9}) | `pad = ((K-1)*d)//2` → 3 / 9 / 27 | pad | pad | **centered** |
| residual `conv2` (K=1) | `padding=0` | 0 | 0 | pointwise (no temporal RF) |
| `ConvTranspose1d` | `padding=ceil(s/2)`, `output_padding=0` | symmetric-ish | symmetric-ish | **non-causal** (needs future inputs to finalize edge outputs) |

Source of the residual pad (HF `DacResidualUnit`):
```python
pad = ((7 - 1) * dilation) // 2   # 3, 9, 27 — SAME, not (K-1)*d left-only
```

3. **Per-layer state size (Conv1d & ConvTranspose1d)**

### Conv1d layers

| module                              | K | d | full=(K-1)d | left=pad | right=pad | scale (samp/in) | left (audio) | right (audio) |
|--------------------------------------|---|---|-------------|----------|-----------|-----------------|--------------|---------------|
| conv1                               | 7 | 1 |      6      |    3     |     3     |      512        |   1536       |    1536       |
| block.0.res_unit1.conv1             | 7 | 1 |      6      |    3     |     3     |      64         |    192       |     192       |
| block.0.res_unit1.conv2             | 1 | 1 |      0      |    0     |     0     |      64         |      0       |       0       |
| block.0.res_unit2.conv1             | 7 | 3 |     18      |    9     |     9     |      64         |    576       |     576       |
| block.0.res_unit2.conv2             | 1 | 1 |      0      |    0     |     0     |      64         |      0       |       0       |
| block.0.res_unit3.conv1             | 7 | 9 |     54      |   27     |    27     |      64         |   1728       |    1728       |
| block.0.res_unit3.conv2             | 1 | 1 |      0      |    0     |     0     |      64         |      0       |       0       |
| block.1.res_unit1.conv1             | 7 | 1 |      6      |    3     |     3     |       8         |     24       |      24       |
| block.1.res_unit1.conv2             | 1 | 1 |      0      |    0     |     0     |       8         |      0       |       0       |
| block.1.res_unit2.conv1             | 7 | 3 |     18      |    9     |     9     |       8         |     72       |      72       |
| block.1.res_unit2.conv2             | 1 | 1 |      0      |    0     |     0     |       8         |      0       |       0       |
| block.1.res_unit3.conv1             | 7 | 9 |     54      |   27     |    27     |       8         |    216       |     216       |
| block.1.res_unit3.conv2             | 1 | 1 |      0      |    0     |     0     |       8         |      0       |       0       |
| block.2.res_unit1.conv1             | 7 | 1 |      6      |    3     |     3     |       2         |      6       |       6       |
| block.2.res_unit1.conv2             | 1 | 1 |      0      |    0     |     0     |       2         |      0       |       0       |
| block.2.res_unit2.conv1             | 7 | 3 |     18      |    9     |     9     |       2         |     18       |      18       |
| block.2.res_unit2.conv2             | 1 | 1 |      0      |    0     |     0     |       2         |      0       |       0       |
| block.2.res_unit3.conv1             | 7 | 9 |     54      |   27     |    27     |       2         |     54       |      54       |
| block.2.res_unit3.conv2             | 1 | 1 |      0      |    0     |     0     |       2         |      0       |       0       |
| block.3.res_unit1.conv1             | 7 | 1 |      6      |    3     |     3     |       1         |      3       |       3       |
| block.3.res_unit1.conv2             | 1 | 1 |      0      |    0     |     0     |       1         |      0       |       0       |
| block.3.res_unit2.conv1             | 7 | 3 |     18      |    9     |     9     |       1         |      9       |       9       |
| block.3.res_unit2.conv2             | 1 | 1 |      0      |    0     |     0     |       1         |      0       |       0       |
| block.3.res_unit3.conv1             | 7 | 9 |     54      |   27     |    27     |       1         |     27       |      27       |
| block.3.res_unit3.conv2             | 1 | 1 |      0      |    0     |     0     |       1         |      0       |       0       |
| conv2                               | 7 | 1 |      6      |    3     |     3     |       1         |      3       |       3       |

- **Naive sum of Conv1d left/right audio ≈ 4436 / 4436 samples** (upper bound if you just add per-layer; real composed receptive field is not exactly this sum).


### ConvTranspose1d layers (overlap/phase state)

Kernel support in output samples: length K, with padding=p shifting the support. Input-side left/right is determined by ceil of out-support / stride.

| module            | K  | s | p | out supp. L/R | ≈ left_in | ≈ right_in | scale_in | left ≈ audio | right ≈ audio |
|-------------------|----|---|---|--------------|-----------|------------|----------|--------------|---------------|
| block.0.conv_t1   | 16 | 8 | 4 |  4 / 11      |    1      |     2      |   512    |    512       |   1024        |
| block.1.conv_t1   | 16 | 8 | 4 |  4 / 11      |    1      |     2      |    64    |     64       |    128        |
| block.2.conv_t1   |  8 | 4 | 2 |  2 / 5       |    1      |     2      |     8    |      8       |     16        |
| block.3.conv_t1   |  4 | 2 | 1 |  1 / 2       |    1      |     1      |     2    |      2       |      2        |

- **Cached state for a ConvTranspose1d**:
  - Last few input frames (left), plus
  - An output overlap buffer of length related to K - s (for partial columns / overlap-add), not just a single (K-1)\*d vector.


### **Totals / streaming latency floor**

| quantity                | value               | note                                     |
|-------------------------|--------------------|------------------------------------------|
| Naive sum L/R           | ~5.0k / ~5.6k samp | (conv1d + transpose approx; not strictly a sum) |
| Measured CONTEXT        | 10 frames = 5120   | left (used as streaming context floor)   |
| Measured TAIL           | 5254               | right (true streaming latency floor)     |
| TAIL ms (44.1kHz)       | ~119.1 ms          | TAIL = ceil(5254/512)=11 frame hold-back |

Correction to checklist: **per-layer left cache is pad = ((K-1)\*d)//2, not the full (K-1)\*d.**  
The full receptive field width is left+right.

- **Using only left (K-1)\*d would over-cache history and still miss the lookahead that creates the right-side TAIL.**
- True streaming floor is measured as above until theory matches measurement.

4. **Snake1d is pointwise: `x + (1/(α+ε)) * sin(αx)²`. No temporal state.**

**Per DacDecoderBlock:**

| Site                                         | Count      |
|-----------------------------------------------|------------|
| block.i.snake1 (before transpose)             | 1          |
| each of 3 residual units: snake1 + snake2     | 3 × 2 = 6  |
| **Per block total**                           | **7**      |


**Full DacDecoder:**

| Site                                | Count        |
|--------------------------------------|--------------|
| 4 × blocks                          | 4 × 7 = 28   |
| final decoder.snake1 (before conv2) | 1            |
| **Decoder total**                   | **29**       |


**Typical fuse pairs**  
(Snake immediately precedes the conv):

- snake1 → conv_t1 (per block) × 4  
- res.snake1 → res.conv1 × 12  
- res.snake2 → res.conv2 × 12  
- final snake1 → conv2 × 1  

→ **29 Snake→conv edges** (every Snake in the decoder sits on a fuse boundary).