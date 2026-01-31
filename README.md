# SGL-JAX: High-Performance LLM Inference on JAX/TPU

SGL-JAX is a high-performance, JAX-based inference engine for Large Language Models (LLMs), specifically optimized for Google TPUs. It is engineered from the ground up to deliver exceptional throughput and low latency for the most demanding LLM serving workloads.

The engine incorporates state-of-the-art techniques to maximize hardware utilization and serving efficiency, making it ideal for deploying large-scale models in production on TPUs.

[![Pypi](https://img.shields.io/badge/pypi-sglang--jax-orange.svg)](https://pypi.org/project/sglang-jax) [![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/sgl-project/sglang-jax?tab=Apache-2.0-1-ov-file#readme) [![View Code Wiki](https://www.gstatic.com/_/boq-sdlc-agents-ui/_/r/YUi5dj2UWvE.svg)](https://codewiki.google/github.com/sgl-project/sglang-jax)


## Key Features

- **High-Throughput Continuous Batching**: Implements a sophisticated scheduler that dynamically batches incoming requests, maximizing TPU utilization and overall throughput.
- **Optimized KV Cache with Radix Tree**: Utilizes a Radix Tree for KV cache management (conceptually similar to PagedAttention), enabling memory-efficient prefix sharing between requests and significantly reducing computation for prompts with common prefixes.
- **FlashAttention Integration**: Leverages a high-performance FlashAttention kernel for faster and more memory-efficient attention calculations, crucial for long sequences.
- **Tensor Parallelism**: Natively supports tensor parallelism to distribute large models across multiple TPU devices, enabling inference for models that exceed the memory of a single accelerator.
- **OpenAI-Compatible API**: Provides a drop-in replacement for the OpenAI API, allowing for seamless integration with a wide range of existing clients, SDKs, and tools (e.g., LangChain, LlamaIndex).
- **Native Qwen Support**: Includes first-class, optimized support for the Qwen model family, including recent Mixture-of-Experts (MoE) variants.

## Architecture Overview

![SGLang-JAX Architecture](docs/_static/image/architecture.svg)

SGL-JAX operates on a distributed architecture designed for scalability and performance:

1.  **HTTP Server**: The entry point for all requests, compatible with the OpenAI API standard.
2. **TokenizerManager**: Runs in the main process, handles text tokenization
3.  **Scheduler**: The core of the engine. It receives requests, manages prompts, and schedules token generation in batches. It intelligently groups requests to form optimal batches for the model executor.
4.  **TP Worker (Tensor Parallel Worker)**: A set of distributed workers that host the model weights, distributed via tensor parallelism. They execute the forward pass for the model.
5.  **Model Runner(Included in TP Worker)**: Manages the actual JAX-based model execution, including the forward pass, attention computation, and KV cache operations.
6. **DetokenizerManager**: Runs in a subprocess, handles output token decoding

More details in [architecture](https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/project-core-structure.md).

---

## Getting Started

- [Install SGL-JAX](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Quick Start](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [Benchmark and Profiling](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)

## Documentation

For more features and usage details, please read the documents in the [`docs`](https://github.com/sgl-project/sglang-jax/tree/main/docs) directory.

## Supported Models

SGL-JAX is designed for easy extension to new model architectures. It currently provides first-class, optimized support for:

-   **Qwen**: Performance needs to improve.
-   **Qwen 2**: Performance needs to improve.
-   **Qwen 2 MoE**: Performance needs to improve.
-   **Qwen 3**: Currently these series have achieved our best performance.
-   **Qwen 3 MoE**: Apart from models like Qwen-coder3-480B with large parameters, these series have achieved our best performance.
-   **Llama**: Performance needs to improve.
-   **Bailing MoE**: Performance needs to improve.
-   **MiMo-7B**: Support Eagle's Speculative Decoding, Performance needs to improve.
-   **Qwen3-ASR**: Audio transcription via OpenAI-compatible chat API (base64 audio).

Currently, SGL-JAX already supports MultiModal Models, and its usage is compatible with LLMs. The architecture has been adapted to support flexible multimodal model architectures.

-   **Wan 2.1 T2V**: Text-to-Video generation model.
-   **Wan 2.2 T2V**: Text-to-Video generation model. Uses different DiT models at different noise stages for denoising, achieving higher generation quality.

For multimodal model usage, see the [Usage Guide](docs/mutlimodal/multimodal_usage.md) and [Architecture Design](docs/mutlimodal/design/[RFC]multimodal_architechure.md).

## ASR (Qwen3-ASR)

SGL-JAX supports Qwen3-ASR with the OpenAI-compatible Chat API. Audio is sent as base64 via `audio_url` and transcribed by the model.

### Dependencies

Audio decoding requires extra packages:

```bash
pip install soundfile
```

Tokenizer note:
- Qwen3-ASR uses a Qwen2-style tokenizer. If you are using a local checkpoint, ensure `vocab.json` and `merges.txt` are present. SGL-JAX will auto-patch missing entries in `tokenizer_config.json` when these files exist.

### Example (base64 audio)

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-ASR-1.7B",
    "messages":[
      {"role":"user","content":[
        {"type":"text","text":"<|audio_pad|>"},
        {"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,AAA..."}}
      ]}
    ],
    "temperature":0.0,
    "max_tokens":256
  }'
```

Notes:
- Only one audio input is supported per request.
- The prompt must include an audio placeholder token (e.g. `<|audio_pad|>`), otherwise the server will reject the request.
 
For a full launch guide, see `docs/basic_usage/asr.md`.

### ASR Warmup (Avoid First-Request JAX Compile Latency)

On TPU, the first ASR request for a new (token padding, audio length bucket) can be slow due to JAX/XLA compilation of the audio feature extraction + audio encoder path.

You can precompile this path during server startup:

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache  # optional but recommended

python -u -m sgl_jax.launch_server \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --trust-remote-code \
  --device tpu \
  --tp-size 4 \
  --dp-size 2 \
  --dtype bfloat16 \
  --page-size 32 \
  --asr-warmup-audio \
  --asr-warmup-audio-frame-paddings 64 128 256 512 1024 2048 3000 \
  --asr-warmup-audio-token-paddings 256 512 1024 2048 4096 8192 16384
```

Options:
- `--asr-warmup-audio`: enable ASR audio-path warmup at startup.
- `--asr-warmup-audio-frame-paddings`: which audio frame buckets to warm up (frames at hop_length=160 by default). If unset, uses `SGLANG_ASR_AUDIO_FRAME_PADDINGS`.
- `--asr-warmup-audio-token-paddings`: which token padding buckets to warm up for ASR prefill (default `[256]`). If you still see `#cache_miss: 1` on prefill, it is usually because your real request hit a new token padding bucket (e.g. 4096/8192/16384) that was not warmed up.

Notes:
- On TPU v6e-8 (8 cores), `--tp-size 4 --dp-size 2` uses all 8 cores. On smaller slices (e.g. v6e-4), keep `--dp-size 1`.
- If you pass `--disable-precompile`, you should expect runtime `#cache_miss: 1` logs when a request hits a new bucket. For throughput testing, keep precompile enabled and use `JAX_COMPILATION_CACHE_DIR` so restarts can reuse compiled executables.
- `#cache_miss` here means a JAX/XLA compilation cache miss (not a KV/prefix-cache miss).
- Audio frame paddings are in "frames", not samples. With 16kHz audio and hop_length=160, 1 frame ~= 10ms, so 100 frames ~= 1 second.
- If `--asr-warmup-audio-frame-paddings` is set, SGL-JAX will automatically set `SGLANG_ASR_AUDIO_FRAME_PADDINGS` to the same list so runtime bucketing matches warmup.
- If your audio can exceed ~30s, increase the buckets (e.g. add 4096 and/or 6000) and keep the warmup buckets consistent with your runtime bucket list.

### Debugging ASR Cache Misses

Enable cache-miss breakdown logs:

```bash
export SGLANG_LOG_CACHE_MISS=1
```

This prints:
- `forward_miss`: misses from model forward / attention kernels
- `sample_miss`: misses from sampling
- `padded_tokens` / `padded_bs`: which padding bucket triggered a (re)compile


## Performance and Benchmarking

For detailed performance evaluation and to run the benchmarks yourself, please see the scripts located in the `benchmark/` and `python/sgl_jax/` directories (e.g., `bench_serving.py`).

## Testing

The project includes a comprehensive test suite to ensure correctness and stability. To run the full suite of tests:

```bash
python test/srt/run_suite.py
```

## Contributing

Contributions are welcome! If you would like to contribute, please feel free to open an issue to discuss your ideas or submit a pull request.

Before contributing, please read our [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
 for setup instructions, coding standards, and contribution workflow.

You can also join our community on Slack to discuss ideas, get help, or collaborate with other contributors:
👉 Join the SGLang Slack workspace (https://slack.sglang.io/), then participate in discussions in the [SGL-JAX Slack Channel](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X).
