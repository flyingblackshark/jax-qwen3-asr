# ASR (Qwen3-ASR)

SGL-JAX supports Qwen3-ASR through the OpenAI-compatible Chat API. Audio is sent as base64 via `audio_url` and transcribed by the model.

## Dependencies

```bash
pip install librosa soundfile
```

## Tokenizer Notes

Qwen3-ASR uses a Qwen2-style tokenizer. If you are using a local checkpoint, make sure `vocab.json` and `merges.txt` are present next to `tokenizer_config.json`. SGL-JAX will auto-patch missing `vocab_file`/`merges_file` entries when these files exist.

## Launch Server

Example launch command (TPU):

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache uv run python -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup
```

## Example (base64 audio)

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-ASR-1.7B",
    "messages":[
      {"role":"user","content":[{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,AAA..."}}]}
    ],
    "temperature":0.0,
    "max_tokens":256
  }'
```

## Notes

- Only one audio input is supported per request.
- The prompt must include a single audio placeholder token (handled automatically by the Qwen3-ASR chat template).
