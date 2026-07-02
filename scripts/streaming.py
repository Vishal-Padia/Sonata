import time
from typing import Callable

import torch
from tqdm import tqdm

from zonos.codebook_pattern import apply_delay_pattern
from zonos.sampling import sample_from_logits


@torch.inference_mode()
def stream_frames(
    self,
    prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
    audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
    max_new_tokens: int = 86 * 30,
    cfg_scale: float = 2.0,
    batch_size: int = 1,
    sampling_params: dict = dict(min_p=0.1),
    progress_bar: bool = False,
    disable_torch_compile: bool = False,
    callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    session: dict | None = None,
):
    """Generator. Mirrors Zonos.generate()'s decode loop, but yields each true
    frame [batch, NQ] the moment its last (delayed) codebook is written, instead
    of returning the whole tensor at the end. `session`, if given, is populated
    with `delayed_codes` and `n_frames_yielded` for the token-level self-check.
    """
    assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
    prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
    device = self.device

    # Use CUDA Graphs if supported, and torch.compile otherwise.
    cg = self.can_use_cudagraphs()
    decode_one_token = self._decode_one_token
    decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

    unknown_token = -1
    audio_seq_len = prefix_audio_len + max_new_tokens
    seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

    with torch.device(device):
        inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
        codes = torch.full((batch_size, 9, audio_seq_len), unknown_token)

    if audio_prefix_codes is not None:
        codes[..., :prefix_audio_len] = audio_prefix_codes

    delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

    delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

    logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
    next_token = sample_from_logits(logits, **sampling_params)

    offset = delayed_prefix_audio_codes.shape[2]
    frame = delayed_codes[..., offset : offset + 1]
    frame.masked_scatter_(frame == unknown_token, next_token)

    prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
    inference_params.seqlen_offset += prefix_length
    inference_params.lengths_per_sample[:] += prefix_length

    logit_bias = torch.zeros_like(logits)
    logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS

    stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_steps = delayed_codes.shape[2] - offset
    remaining_steps = torch.full((batch_size,), max_steps, device=device)
    progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)
    cfg_scale = torch.tensor(cfg_scale)

    NQ = self.autoencoder.num_codebooks
    n_frames_yielded = 0

    step = 0
    while torch.max(remaining_steps) > 0:
        offset += 1
        input_ids = delayed_codes[..., offset - 1 : offset]
        logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
        logits += logit_bias

        next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
        eos_in_cb0 = next_token[:, 0] == self.eos_token_id

        remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
        stopping |= eos_in_cb0[:, 0]

        eos_codebook_idx = 9 - remaining_steps
        eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
        for i in range(next_token.shape[0]):
            if stopping[i]:
                idx = eos_codebook_idx[i].item()
                next_token[i, :idx] = self.masked_token_id
                next_token[i, idx] = self.eos_token_id

        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == unknown_token, next_token)
        inference_params.seqlen_offset += 1
        inference_params.lengths_per_sample[:] += 1

        remaining_steps -= 1

        # Un-stagger: the frame that just became complete is (offset - NQ). Its
        # codebook k lives at delayed position (i + k + 1) -- exactly one column
        # of revert_delay_pattern. torch.stack copies, so the yielded frame is a
        # frozen snapshot (a handed-out frame never changes later).
        i = offset - NQ
        if i >= 0:
            frame_tokens = torch.stack(
                [delayed_codes[:, k, i + k + 1] for k in range(NQ)], dim=1
            )  # [batch, NQ]
            n_frames_yielded += 1
            yield frame_tokens

        progress.update()
        step += 1

        if callback is not None and not callback(frame, step, max_steps):
            break

    self._cg_graph = None  # reset cuda graph to avoid cache changes

    if session is not None:
        session["delayed_codes"] = delayed_codes
        session["n_frames_yielded"] = n_frames_yielded


def measure_receptive_field(model, num_frames: int = 200, eps: float | None = None) -> tuple[int, int]:
    """Empirically measure how far one perturbed frame's effect reaches in the
    decoded waveform, in both directions.

    Returns:
        CONTEXT: frames of left context `chunked_decode` must prepend to a chunk.
        TAIL: samples on the right `chunked_decode` must hold back from a chunk
            (they lacked real right-context when decoded).
    """
    device = model.device
    codebook_size = model.autoencoder.codebook_size  # [0, 1024)
    hop = model.autoencoder.dac.config.hop_length  # 512 samples/frame

    g = torch.Generator(device=device).manual_seed(0)
    codes_A = torch.randint(0, codebook_size, (1, 9, num_frames), device=device, generator=g)

    if eps is None:
        # Calibrate the noise floor first (same "measure the dtype floor before
        # trusting a diff" discipline as validation_harness.py --measure-floors):
        # decode() runs under fp16 autocast, so decoding the SAME codes twice
        # should be bit-identical, but don't assume -- measure it, and require
        # the real signal to clear that floor by a wide margin.
        wav_same_1 = model.autoencoder.decode(codes_A)
        wav_same_2 = model.autoencoder.decode(codes_A)
        floor = (wav_same_1 - wav_same_2).abs().max().item()
        eps = max(1e-5, floor * 10)

    # Probe a few interior frames. Avoid the first/last ~1/7th of the tensor --
    # near the boundary, the *tensor edge* truncates the true receptive field
    # and would report a falsely small CONTEXT/TAIL.
    margin = max(1, num_frames // 7)
    probe_frames = [num_frames // 4, num_frames // 2, (3 * num_frames) // 4]

    contexts, tails = [], []
    for probe_frame in probe_frames:
        assert margin <= probe_frame <= num_frames - margin, "probe_frame too close to tensor edge"

        codes_B = codes_A.clone()
        offset = torch.randint(1, codebook_size, (1, 9), device=device, generator=g)
        codes_B[:, :, probe_frame] = (codes_A[:, :, probe_frame] + offset) % codebook_size

        wav_A = model.autoencoder.decode(codes_A)
        wav_B = model.autoencoder.decode(codes_B)
        diff = (wav_A - wav_B).abs().reshape(-1)

        affected = torch.nonzero(diff > eps, as_tuple=True)[0]
        assert affected.numel() > 0, f"perturbation at frame {probe_frame} had no measurable effect (eps={eps:.2e})"

        center = probe_frame * hop
        left_edge, right_edge = affected.min().item(), affected.max().item()

        contexts.append(-(-(center - left_edge) // hop))  # ceil division
        tails.append(right_edge - center)

    return max(contexts), max(tails)


def chunked_decode(model, codes: torch.Tensor, chunk_frames: int, context: int, tail: int) -> torch.Tensor:
    """Overlap-save decode of a full, already-cleaned codes tensor [B, 9, T].

    Equivalent (up to CONTEXT/TAIL correctness) to `model.autoencoder.decode(codes)`
    in one shot, but processes `chunk_frames` at a time -- the standalone building
    block that `stream_and_vocode` below drives incrementally.

    IMPORTANT: holding back `tail` samples does NOT mean discarding them forever --
    the frames underneath that held-back region must be re-decoded on the *next*
    window (where they'll finally have real right-context), not skipped. So the
    "safely emitted" frame boundary only advances by (chunk_frames - tail_frames)
    per non-final iteration, not by chunk_frames. Getting this wrong silently drops
    ~tail_frames worth of audio at every single chunk boundary.
    """
    hop = model.autoencoder.dac.config.hop_length
    tail_frames = -(-tail // hop)  # ceil to a whole number of frames: conservative
                                    # (discards a hair more than the measured minimum,
                                    # but keeps the frame-level bookkeeping exact)
    assert chunk_frames > tail_frames, (
        f"chunk_frames ({chunk_frames}) must exceed tail in frames ({tail_frames}) "
        f"or every non-final chunk makes zero forward progress"
    )

    T = codes.shape[-1]
    pieces = []
    pos = 0  # frames whose audio has been safely committed to the output so far
    while pos < T:
        end = min(pos + chunk_frames, T)
        window_start = max(0, pos - context)
        window = codes[..., window_start:end]

        wav_chunk = model.autoencoder.decode(window)
        warm_up = (pos - window_start) * hop

        is_last_chunk = end == T
        if is_last_chunk:
            # No more future frames are ever coming for this run -- nothing left
            # to hold back FOR. Commit the whole rest of the window.
            safe = wav_chunk[..., warm_up:]
            pos = end
        else:
            committed_end = end - tail_frames  # this chunk's uncertain tail stays pending
            safe = wav_chunk[..., warm_up : warm_up + (committed_end - pos) * hop]
            pos = committed_end

        pieces.append(safe)

    return torch.cat(pieces, dim=-1)


def stream_and_vocode(frame_iter, model, chunk_frames: int, context: int, tail: int, session: dict | None = None):
    """Consume an iterator of raw [batch, NQ] token frames (e.g. from stream_frames),
    buffer them, and yield waveform chunks via the same overlap-save windowing as
    chunked_decode, as frames actually become available. `session`, if given, is
    populated with wall-clock `ttfa_s` (time from the first `next()` pull to the
    first emitted audio chunk) and the total `n_frames` consumed.
    """
    hop = model.autoencoder.dac.config.hop_length
    tail_frames = -(-tail // hop)  # ceil; see chunked_decode for why this must be exact
    assert chunk_frames > tail_frames, (
        f"chunk_frames ({chunk_frames}) must exceed tail in frames ({tail_frames}) "
        f"or every non-final chunk makes zero forward progress"
    )
    t0 = time.perf_counter()
    first_chunk_emitted = False

    buffered = []  # list of [batch, NQ, 1] token columns
    pos = 0  # frames whose audio has been safely committed to the output so far
    for frame_tokens in frame_iter:
        frame_tokens = frame_tokens.clone()
        frame_tokens[frame_tokens >= 1024] = 0  # EOS/mask cleanup, per-frame (generate() does this once, batched, at the end)
        buffered.append(frame_tokens.unsqueeze(-1))

        if len(buffered) - pos >= chunk_frames:
            end = len(buffered)
            window_start = max(0, pos - context)
            window = torch.cat(buffered[window_start:end], dim=-1)

            wav_chunk = model.autoencoder.decode(window)
            warm_up = (pos - window_start) * hop
            committed_end = end - tail_frames  # tail stays pending, re-decoded next time with more context
            safe = wav_chunk[..., warm_up : warm_up + (committed_end - pos) * hop]

            if not first_chunk_emitted:
                first_chunk_emitted = True
                if session is not None:
                    session["ttfa_s"] = time.perf_counter() - t0

            yield safe
            pos = committed_end

    if pos < len(buffered):
        # Flush the remainder exactly like chunked_decode's last-chunk case: no
        # more frames are coming, ever, so nothing gets held back.
        window_start = max(0, pos - context)
        window = torch.cat(buffered[window_start:], dim=-1)
        wav_chunk = model.autoencoder.decode(window)
        warm_up = (pos - window_start) * hop
        safe = wav_chunk[..., warm_up:]

        if not first_chunk_emitted:
            first_chunk_emitted = True
            if session is not None:
                session["ttfa_s"] = time.perf_counter() - t0

        yield safe

    if session is not None:
        session["n_frames"] = len(buffered)
