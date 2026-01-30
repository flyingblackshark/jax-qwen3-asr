#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
AUDIO_PATH="${AUDIO_PATH:-/root/jax-qwen3-asr/test.mp3}"
REQ_PATH="${REQ_PATH:-/tmp/asr_request.json}"
RESP_PATH="${RESP_PATH:-/tmp/asr_response.json}"

if [[ ! -f "${AUDIO_PATH}" ]]; then
  echo "Audio file not found: ${AUDIO_PATH}" >&2
  exit 1
fi

export AUDIO_PATH
export MODEL_PATH
export REQ_PATH

python - <<'PY'
import base64
import json
import os
import pathlib

audio_path = os.environ.get("AUDIO_PATH", "/root/jax-qwen3-asr/test.mp3")
audio_b64 = base64.b64encode(pathlib.Path(audio_path).read_bytes()).decode()

payload = {
    "model": os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-1.7B"),
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/mp3;base64," + audio_b64},
                }
            ],
        }
    ],
    "temperature": 0.0,
    "max_tokens": 256,
}

pathlib.Path(os.environ.get("REQ_PATH", "/tmp/asr_request.json")).write_text(
    json.dumps(payload)
)
PY

http_code=$(curl -s --max-time 120 -o "${RESP_PATH}" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d @"${REQ_PATH}" \
  "http://${HOST}:${PORT}/v1/chat/completions" || true)

echo "HTTP_CODE=${http_code}"
cat "${RESP_PATH}"
