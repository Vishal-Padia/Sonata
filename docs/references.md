# References

I mean, I searched "has anyone, anywhere, published fused streaming decode kernels for zonos mamba2-hybrid backbone?", Why am I doing this? because I feel the fastest way to write good kernels is to read someone else's kernels and understand how they work. We can understand things like register allocation, smem layout for SSM state, and other design decisions. By doing so, I think I'll be able to write better kernels and understand the underlying principles of the kernels.

I thought about skipping this step because I didn't think it matters whether someone has wrote something similar or not, because the whole of point of this project is to learn and challenge myself to write good kernels and show that I have the necessary skills to optimize end-to-end. But then claude convinced me to do it anyways because this will teach me a lot and give me reference to look at when I'm stuck.

Found this, https://github.com/Zyphra/Zonos/pull/49, which has done some work regarding streaming support for zonos. But the last push to this was in July of 2025 (ie a year ago). I can still use this as a reference to look at when I'm stuck. And damn it has some good code in it, and it will be definitely helpful when I'll add streaming to zonos

Found this, https://github.com/Zyphra/Zonos/pull/51, Removes the mamba-ssm dependency so the transformer model can run on CPU/Mac. This is python level architecture change, the comments on this PR are super informative, we have the god `ezyang` himself reviewing the PR, so it's definetly worth reading.

What does not exist anywhere:

- A fused decode attention kernel (Triton or CUDA) for Zonos's transformer backbone
- A fused RMSNorm+residual kernel
- A fused RoPE kernel
- Any KV-cache-aware decode kernel
- CUDA graph capture for the decode loop
- Any CuteDSL work on Zonos

So yea, it seems like no one has done something like this before and I'll be first one do it. I would it's scary because no one has done it before, but I think I am gonna be alright.