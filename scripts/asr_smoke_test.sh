#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
MODEL_REVISION="${MODEL_REVISION:-}"
DEVICE="${DEVICE:-cpu}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
PORT="${PORT:-30000}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
MODEL_LAYER_NUMS="${MODEL_LAYER_NUMS:-1}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-2048}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
SKIP_WARMUP="${SKIP_WARMUP:-1}"
DISABLE_PRECOMPILE="${DISABLE_PRECOMPILE:-1}"

AUDIO_PATH="${AUDIO_PATH:-/root/jax-qwen3-asr/test.mp3}"
LOG_PATH="${LOG_PATH:-/tmp/asr_server.log}"
REQ_PATH="${REQ_PATH:-/tmp/asr_request.json}"
RESP_PATH="${RESP_PATH:-/tmp/asr_response.json}"

export AUDIO_PATH
export MODEL_PATH
export REQ_PATH

if [[ ! -f "${AUDIO_PATH}" ]]; then
  echo "Audio file not found: ${AUDIO_PATH}" >&2
  exit 1
fi

export PYTHONPATH="/root/sglang-jax/python${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

if [[ "${DEVICE}" == "cpu" ]]; then
  export JAX_PLATFORM_NAME=cpu
  export JAX_PLATFORMS=cpu
fi

SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --device "${DEVICE}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --load-format "${LOAD_FORMAT}"
  --model-layer-nums "${MODEL_LAYER_NUMS}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --host 127.0.0.1
  --port "${PORT}"
)

if [[ -n "${MODEL_REVISION}" ]]; then
  SERVER_ARGS+=(--revision "${MODEL_REVISION}")
fi
if [[ "${SKIP_WARMUP}" == "1" ]]; then
  SERVER_ARGS+=(--skip-server-warmup)
fi
if [[ "${DISABLE_PRECOMPILE}" == "1" ]]; then
  SERVER_ARGS+=(--disable-precompile)
fi

python -u -m sgl_jax.launch_server "${SERVER_ARGS[@]}" >"${LOG_PATH}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "Server not ready after timeout." >&2
  tail -n 200 "${LOG_PATH}" || true
  exit 1
fi

python - <<'PY'
import base64
import json
import pathlib
import os

audio_path = os.environ.get("AUDIO_PATH", "/root/jax-qwen3-asr/test.mp3")
audio_b64 = base64.b64encode(pathlib.Path(audio_path).read_bytes()).decode()

payload = {
    "model": os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-1.7B"),
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|audio_pad|>"},
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/mp3;base64," + audio_b64},
                }
            ],
        }
    ],
    "temperature": 0.0,
    "max_tokens": 32,
}

pathlib.Path(os.environ.get("REQ_PATH", "/tmp/asr_request.json")).write_text(
    json.dumps(payload)
)
PY

http_code=$(curl -s --max-time 600 -o "${RESP_PATH}" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d @"${REQ_PATH}" \
  "http://127.0.0.1:${PORT}/v1/chat/completions" || true)

echo "HTTP_CODE=${http_code}"
cat "${RESP_PATH}"
