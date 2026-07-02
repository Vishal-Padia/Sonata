# What Streaming Means?

Streaming means that the model will generate audio in chunks and the audio will be emitted as it's generated.

Batching: generate all 5 seconds, and then play it. The listener waits for the whole 5 seconds to be generated before it can hear anything.

Streaming: generate a chunk of audio, and then play it. The listener can hear the audio as it's generated.


The number that captures this is TTFA - Time To First Audio: how long from "go" until the listenr hears anything. Batch TTFA is terrible ( because we have to wait for the whole thing to be generated before we can hear anything), Streaming TTFA is good ( because we can hear the audio as it's generated).

# Current Scene

Our pipeline has two stages that run strictly one-then-the-other
1. generate() - runs the model once per frame and accumulates all the frames into one big array called codes. It returns only after the last frame is done.
2. autoencoder.decode(codes) takes that whole array and turns it into a sound waveform, all in one shot at the end.

So the first sound doesn't exist until stage 1 completely finishes and stage 2 runs. Streaming means interleaving them: the moment a handful of frames are ready, vocode just those and emit the audio and then keep generating the rest.

# Things we would be working on:

- the decode loop (the backbone): this predicts the tokens for the next frame, one frame at a time. It's autoregressive - frame 100 depends on frame 0-99. This is already a loop, we just need it to hand out each frame as it's finished instead of hoarding them all.
- the vocoder (the DAC decoder): turns tokens into actual sound samples. Today it eats the whole array, we need it to eat small chunks

# States

the model is autoregressive, to predict frame 100, it needs to "remember" frames 0-99. It would be insane to re-read all of them every step, so the model keeps a running summary that it updates each step:
- attention layers keep a KV cache: a growing list, one entry per past frame
- Mamba / SSM layers keep an SSM state: a small, fixed-size tensor that summarizes all history (that's the magic of SSMs - constant memory no matter how long the audio). Plus a tiny buffer of the last few frames for it's little convolution.

the existing generate() loop already creates and updates all these buffers every single step - that's what the inference_params object is. When we stream, we run the exact same loop with the exact same state updates. We''re only changing what we do with each frame after the step (emit it now vs store it for later)

As of now, I think we should just focus on token-streaming, we don't need to touch SSM state buffers at all. 

# the delay pattern

Zonos emits 9 token streams per frame (think of them as 9 layres of audio detail, coarse to fine). For training reason, it staggers them: for a given frame, stream 0 comes out first, stream 1 one step later, ...., stream 8 nine steps later. So a complete frame (all 9 stream assembled) is only read about 9 steps after it started.

Consequence: even a perfect streaming system can't produce the first complete frame until ~9 steps in ~= 100ms. That's a floor we cannot remove - good to know, so we aim at a realistic TTFA, not an impossible one.

# what we need to do:

Step 1: Make the loop hand out frames as they finish

Take the generation loop and have it give us each complete frame the moment it's ready, instead of only at the end. The only real work is un-staggering the delay pattern live.

How would we know if it's right: the frames we hand out, stitched back together, must be exactly identical to what the batch streaming produces - checked within a single run so there's no randomness to fgiht.

Step 2: vocode in chunks so audio flows

Feed those frames to the vocoder a small chunk at a time. There's a quick-and-good-enough way to do it now and then we can do it "perfectly clean" way later. 

How we know if it's correct: the streamed audio measures close to (and sounds like) the batch audio - using the audio gate we have already build.

Step 3: measure TTFA

Time from "go" to first sound, and check that audio keeps up with the playback. That's the headline result.

Step 4: Phase 2, later:

make the vocoder click-free across chunks - the "cached conv state" work.

