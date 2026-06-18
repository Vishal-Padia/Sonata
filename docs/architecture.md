# What does this contain?

This contains the things related to Mamba2 block internals. Basically all the things I learned about Mamba2 block internals. I have to read the Zonos hybrid's actual Mamba2 block code, in the checkpoint, with the actual dims, and write down what's there.

## Backbone config:
```
ZonosConfig(
    backbone=BackboneConfig(
        d_model=2048,
        d_intermediate=0,
        attn_mlp_d_intermediate=8192,
        n_layer=46,
        ssm_cfg={'layer': 'Mamba2'},
        attn_layer_idx=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        attn_cfg={
            'causal': True,
            'num_heads': 16,
            'num_heads_kv': 4,
            'rotary_emb_dim': 128,
            'qkv_proj_bias': False,
            'out_proj_bias': False
        },
        rms_norm=False,
        residual_in_fp32=False,
        norm_epsilon=1e-05
    ),
    prefix_conditioner=PrefixConditionerConfig(
        conditioners=[
            {'type': 'EspeakPhonemeConditioner', 'name': 'espeak'},
            {
                'cond_dim': 128,
                'uncond_type': 'learned',
                'projection': 'linear',
                'type': 'PassthroughConditioner',
                'name': 'speaker'
            },
            {
                'input_dim': 8,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'emotion'
            },
            {
                'min_val': 0,
                'max_val': 24000,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'fmax'
            },
            {
                'min_val': 0,
                'max_val': 400,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'pitch_std'
            },
            {
                'min_val': 0,
                'max_val': 40,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'speaking_rate'
            },
            {
                'min_val': -1,
                'max_val': 126,
                'uncond_type': 'learned',
                'type': 'IntegerConditioner',
                'name': 'language_id'
            },
            {
                'input_dim': 8,
                'min_val': 0.5,
                'max_val': 0.8,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'vqscore_8'
            },
            {
                'min_val': -1.0,
                'max_val': 1000,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'ctc_loss'
            },
            {
                'min_val': 1,
                'max_val': 5,
                'uncond_type': 'learned',
                'type': 'FourierConditioner',
                'name': 'dnsmos_ovrl'
            },
            {
                'min_val': 0,
                'max_val': 1,
                'uncond_type': 'learned',
                'type': 'IntegerConditioner',
                'name': 'speaker_noised'
            }
        ],
        projection='linear'
    ),
    eos_token_id=1024,
    masked_token_id=1025,
    pad_vocab_to_multiple_of=8
)
```

### What's in the backbone config?

- `d_model`: 2048
- `d_intermediate`: 0
- `attn_mlp_d_intermediate`: 8192
- `n_layer`: 46
- `ssm_cfg`: {'layer': 'Mamba2'}
- `attn_layer_idx`: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
- `attn_cfg`: {'causal': True, 'num_heads': 16, 'num_heads_kv': 4, 'rotary_emb_dim': 128, 'qkv_proj_bias': False, 'out_proj_bias': False}
- `rms_norm`: False
- `residual_in_fp32`: False
- `norm_epsilon`: 1e-05

This is much different then the config written in `third_party/zonos/config.py`
```
BackboneConfig(
    d_model=1024,
    d_intermediate=0,
    attn_mlp_d_intermediate=0,
    n_layer=16,
    ssm_cfg={'layer': 'Mamba2'},
    attn_layer_idx=[0, 4, 8, 12, 16],
    attn_cfg={'causal': True, 'num_heads': 16, 'num_heads_kv': 4, 'rotary_emb_dim': 128, 'qkv_proj_bias': False, 'out_proj_bias': False},
    rms_norm=False,
    residual_in_fp32=False,
    norm_epsilon=1e-05
)
```

## Mamba2 layer dims (resolved)

`ssm_cfg` is just `{'layer': 'Mamba2'}`, so every Mamba2 hyperparameter falls through to the defaults in `Mamba2.__init__` (`mamba_ssm/modules/mamba2.py`). Those defaults, against our `d_model = 2048`, resolve to concrete integers:

| param | value | source |
|---|---|---|
| `d_model` | 2048 | backbone config |
| `d_state` | 128 | `Mamba2.__init__` default |
| `headdim` | 64 | `Mamba2.__init__` default |
| `expand` | 2 | `Mamba2.__init__` default |
| `d_conv` | 4 | `Mamba2.__init__` default |
| `ngroups` | 1 | `Mamba2.__init__` default |
| `chunk_size` | 256 | `Mamba2.__init__` default |

Derived:

- `d_inner = expand * d_model = 2 * 2048 = `**`4096`** (also `d_ssm`, since `d_ssm=None`)
- `n_heads = d_inner // headdim = 4096 // 64 = `**`64`**

Layer tensors that follow from these:

- `in_proj`: out features `= 2*d_inner + 2*ngroups*d_state + n_heads = 8192 + 256 + 64 = `**`8512`**, split as `[z, xBC, dt] = [4096, 4352, 64]`
- `conv1d`: depthwise over `conv_dim = d_ssm + 2*ngroups*d_state = 4096 + 256 = `**`4352`**, kernel width 4, with bias
- `out_proj`: `4096 -> 2048`

Decode-time state (per sample, per Mamba2 layer):

- `ssm_state`: `(batch, n_heads, headdim, d_state) = (b, 64, 64, 128)` = 524,288 elts -> ~1 MB/sample in bf16
- `conv_state`: `(batch, conv_dim, d_conv) = (b, 4352, 4)` -> small

Note: `rmsnorm=True` is also a default, so each Mamba2 block has an internal **gated RMSNorm on d_ssm=4096**. This is a *different* norm from the backbone-level `rms_norm=False` (the inter-block norm is LayerNorm). The fused decode kernel has to reproduce both.

### Mamba_ssm (`third_party/zonos/backbone/_mamba_ssm.py`):

So this is the file we will be reimplementing, so I have to understand this like my life depends on it lol. Basically this file tells how Zonos composes Mamba2 blocks with layers, where the per-layer norms sit, how the residual stream flows, and crucially how the inference-time decode setup is wired (the `step()` call that drives the AR loop).

This is the model, it's the contract sheet that my future fused mamba2 decode kernel has to drop into cleanly. Specifically it declares:
- Who owns the residual stream
- Where add-norm fusion happens
- What the inference cache looks like and who allocates it
- What the per-layer forward contract is at decode time
- Where the heterogeneity between mamba2 layers and attention layers lives

#### Contract 1: Layer Heterogeneity:
`create_block(...)` is a dispatching factory: if `layer_idx in attn_layer_idx`, it returns an attention block; otherwise a Mamba2 block. So `self.layers[i]` is sometimes an attention block, and they share the same forward signature but have completely different internals. The fused kernel I will write will replace Mamba2 indices. The attention indices stay as-is. 

#### Contract 2: `fused_add_norm=True`
When the flag is on, the block's entry fuses the previous residual addition with its own input pre-norm into one kernel. That changes the residual contract: the block doesn't receive a pre-summed `hidden_states`, it receives `hidden_states` and `residual` separately, and the block decides when to sum them (right before its first norm). Each block takes (`hidden_states`, `residual`) and returns (`hidden_states`, `residual`). The residual is threaded alongside `hidden_states`, not folded into it. This is memory-traffic optimization. 

#### Contract 3: The residual-stream lifecycle
The residual stream is born at `None`, threaded through every block, and the loop ends with one final fused add-norm that finally sums the residual into `hidden_states` and normalizes the result. So there's an add-norm fusion at every block entry and another one at the exit of the whole stack. The `layer_norm_fn` is a triton kernel that fuses residual addition with norm in one pass. The implicit `residual_in_fp32` contract: residuals can be kept in fp32 even when activations are in bf16.

#### Contract 4: The inference cache
```
def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
    return {
        i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        for i, layer in enumerate(self.layers)
    }
```
Each layer allocates its own cache. The shape differs per layer type:
- Mamb2 layers cache: a recurrent state tensor `(batch, n_heads, head_dim, d_state)` plus a small conv1d state `(batch, conv_dim, d_conv)`. This is the tensor our fused decode kernel reads and writes every step.
- Attention layer's cache: KV cache, shape `(batch, n_heads_kv, max_seqlen, head_dim) * 2`. Standard transformer decode cache.

When we will do CUDA-graph capture, the pointer addresses of these caches must remain stable across graph replays. That's a constraint on how we will allocate.

#### Contract 5: `inference_params`
Every layer takes `inference_params` as thrid arg. This ist he `InferenceParams` dataclass imported from the config. It carries:
- `seqlen_offset`: where in the AR loop we are
- `lengths_per_sample`: per-sample sequence length (for variable-length batches; we'll have batch=1 in streamining so this is trivially `[seqlen_offset]`)
- `key_value_memory_dict`: the cache dict from Contract 4.

## The `step` function in Mamba2 (Thanks Claude for filling this up)

This is `Mamba2.step(hidden_states, conv_state, ssm_state)` (`mamba2.py:278`), the decode path taken when `inference_params.seqlen_offset > 0` (`mamba2.py:173-176`). Model is loaded in **bf16**; everything below is bf16 unless noted. `d_mlp = 0` here (`(8512 - 2*4096 - 256 - 64)//2 = 0`), so the `z0`/`x0` MLP-skip branch never fires and the `torch.cat` at the end is dead code.

**Existing kernel boundaries (the most important thing on this page).** Both heavy steps are *already* single fused ops in the installed build (`causal_conv1d` and `mamba_ssm` are both present in the venv, so the `... is None` fallbacks below are NOT taken):

- `causal_conv1d_update` (`mamba2.py:298`) — fuses the conv-state roll, depthwise conv, bias, and **SiLU** into one kernel. Updates `conv_state` in place.
- `selective_state_update` (`mamba2.py:333`) — fuses dt-discretization, the state recurrence, and the output read into one kernel. Updates `ssm_state` in place. This is the op my fused decode kernel replaces; **stages 5-7 below are all inside this one call**, not separate Python ops. They appear as explicit ops only in the `selective_state_update is None` fallback (`mamba2.py:310-322`).

| Stage | Op / Description              | Input tensors (shape, dtype)                                                                        | Output (shape, dtype)       | Notes                                                                                                                        |
|-------|------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 0     | input                        | hidden_states (b, 1, 2048) bf16                                                                     | -                           | squeezed to (b, 2048)                                                                                                        |
| 1     | `in_proj` GEMV               | hidden_states (b,2048) bf16; weight (8512,2048) bf16                                                | zxbcdt (b, 8512) bf16       | memory-bound GEMV at b=1                                                                                                     |
| 2     | split `[z, x, B, C, dt]`     | zxbcdt                                                                                              | z (b,4096), xBC (b,4352), dt (b,64) bf16 | d_mlp=0 -> z0,x0 empty. xBC = x∥B∥C still interleaved                                  |
| 3     | **`causal_conv1d_update`**   | xBC (b,4352) bf16; conv_state (b,4352,4) bf16; weight (4352,4) bf16; bias (4352) bf16               | xBC (b,4352) bf16; conv_state updated in place | SiLU activation is **inside** this op                                               |
| 4     | split                        | xBC                                                                                                 | x (b,4096), B (b,128), C (b,128) bf16 | B,C are ngroups·d_state = 1·128                                                       |
| —     | `A = -exp(A_log.float())`    | A_log (64) bf16 param                                                                               | A (64) **fp32** -> repeat-> (64,64,128) fp32 | A always fp32; sign flip here                                                |
| 5-7   | **`selective_state_update`** | ssm_state (b,64,64,128); x (b,64,64); dt (b,64,64); A (64,64,128) fp32; B (b,1,128); C (b,1,128); D (64,64); dt_bias (64,64); `dt_softplus=True`; `z=None` | y (b,64,64) bf16 -> (b,4096); ssm_state updated in place | discretize `softplus(dt+dt_bias)`, `dA=exp(dt·A)`, `state=state·dA+dB·x`, `y=⟨state,C⟩+D·x` - all internal. Internal math fp32; stored state dtype = ssm_state dtype (bf16). z **not** passed (rmsnorm path) |
| 8     | gated RMSNorm `self.norm(y, z)` | y (b,4096) bf16; z (b,4096) bf16; norm.weight (4096) bf16                                         | y (b,4096) bf16             | **SiLU(z) gating happens INSIDE this op** (RMSNormGated, `layernorm_gated`), not in stage 7                                 |
| 9     | `out_proj` GEMV              | y (b,4096) bf16; weight (2048,4096) bf16                                                            | out (b,2048) bf16 -> (b,1,2048) | memory-bound GEMV                                                                  |

### Verification notes (the points to get exactly right)

1. **`selective_state_update` is the existing fused-op boundary.** In the installed build it subsumes discretization + recurrence + output (stages 5-7). My fused decode kernel's job is to fuse *across* this boundary — pulling stages 3 (conv), 5-7 (SSM), 8 (gated norm), and ideally the 1/9 projections into one kernel so `ssm_state`/`conv_state` hit HBM once per step. The Python ops in the fallback (`mamba2.py:310-322`) are the reference for *what* it computes, not what actually runs.
2. **`causal_conv1d_update`** takes `xBC (b,4352)`, the cached `conv_state (b,4352,4)`, the depthwise weight `(4352,4)` and bias `(4352)`, applies SiLU, returns `xBC (b,4352)` and updates `conv_state` in place. The conv only touches the `x∥B∥C` stream (4352 = d_ssm + 2·ngroups·d_state); `z` and `dt` bypass it.
3. **Gating on `z` lives inside the gated RMSNorm.** Because `rmsnorm=True`, `z=None` is passed to `selective_state_update`, so the SiLU(z) gate is applied by `self.norm(y, z)` at stage 8, not in the SSM op. (Only in the `rmsnorm=False` fallback would gating be `y * silu(z)` outside the norm.)
4. **dt and A are exactly as you guessed.** dt discretization is `softplus(dt + dt_bias)` (carried by `dt_softplus=True` + the `dt_bias` arg into the fused op; explicit `F.softplus(dt + dt_bias)` in the fallback, `mamba2.py:313`). `A = -torch.exp(A_log.float())` (`mamba2.py:307`) — per-head, fp32, with the negative sign applied here.
5. **CORRECTION — `ssm_state` is bf16 in the reference, not fp32.** `allocate_inference_cache` (`mamba2.py:351-354`) sets `ssm_dtype = dtype` and Zonos calls `setup_cache(..., dtype=torch.bfloat16)` (`model.py:198`), so the **stored** `ssm_state` (and `conv_state`) is **bf16**. The fp32 only appears *inside* `selective_state_update`, which upcasts the recurrence math to fp32 and writes the result back down to the bf16 state tensor each step. So the plan's "keep state accumulation in fp32" is a prescription for **my new kernel** — persist the state tensor itself in fp32 across steps, going beyond the reference's bf16 round-trip — not a property of the stock code. A is the only tensor the reference forces to fp32 throughout.


## DAC Configuration

`third_party/zonos/autoencoder.py` is a DAC wrapper here.
- `sampling_rate` = 44100Hz
- `downsampling_factor` = 512 (this is standard, also look at `third_party/zonos/autoencoder.py:19`)
- `frame_rate` = `sampling_rate / downsampling_factor` = 44100 / 512 = 86.13Hz
- per-frame budget at 1x RTF = 1 / frame_rate = 1 / 86.13 = 11.61ms/frame
- `num_codebooks` = 9
- `current per-frame compute` = **10.07 ms wall / 9.00 ms GPU** per step, measured on A10G (sm86) — see `docs/baseline_per_block.md`.
- `headroom` = 11.61 - 10.07 = **1.54 ms/frame** at 1× → **RTF 1.15×**, backbone decode only (vocode not yet included)

>The 9 codebooks are emitted with a delay shift: codebook k at AR step t predicts the token for frame t - (k+1) (roll by k+1). The backbone still runs once per AR step and emits all 9 codebook logits in parallel (9 separate heads off the same hidden state). For streaming this means: one backbone step = one frame of progress after the initial n_codebooks prefill steps prime the delay buffer. The per-step budget is therefore the per-frame budget — 11.6 ms — not 11.6/9 ms. The delay pattern only affects how codebooks line up at the boundaries of the sequence (first/last few steps), not the steady-state step rate.