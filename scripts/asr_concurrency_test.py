#!/usr/bin/env python3
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from statistics import mean

import requests

try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "soundfile is required for audio duration. Install with: pip install soundfile"
    ) from e


def _load_audio_b64(audio_path: str) -> tuple[str, float]:
    raw = open(audio_path, "rb").read()
    audio, sr = sf.read(BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = float(audio.shape[0]) / float(sr)
    b64 = base64.b64encode(raw).decode()
    return b64, duration


def _send_one(url: str, payload: dict, timeout_s: float) -> tuple[bool, float, int, int]:
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        latency = time.perf_counter() - start
        if resp.status_code != 200:
            return False, latency, resp.status_code, 0
        data = resp.json()
        usage = data.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        return True, latency, resp.status_code, total_tokens
    except Exception:
        latency = time.perf_counter() - start
        return False, latency, 0, 0


def main():
    parser = argparse.ArgumentParser(description="ASR concurrency throughput test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--audio", default="/root/jax-qwen3-asr/test.mp3")
    parser.add_argument("--requests", type=int, default=40, dest="num_requests")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    b64, duration = _load_audio_b64(args.audio)
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|audio_pad|>"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/mp3;base64," + b64},
                    }
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    latencies = []
    statuses = []
    token_counts = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(_send_one, url, payload, args.timeout)
            for _ in range(args.num_requests)
        ]
        for fut in as_completed(futures):
            ok, latency, status, total_tokens = fut.result()
            latencies.append(latency)
            statuses.append((ok, status))
            token_counts.append(total_tokens)

    elapsed = time.perf_counter() - start_time
    ok_count = sum(1 for ok, _ in statuses if ok)
    err_count = len(statuses) - ok_count

    req_tput = ok_count / elapsed if elapsed > 0 else 0.0
    audio_tput = (ok_count * duration) / elapsed if elapsed > 0 else 0.0
    token_tput = sum(token_counts) / elapsed if elapsed > 0 else 0.0

    print(f"Requests: {ok_count} ok / {err_count} err")
    if latencies:
        print(f"Latency avg: {mean(latencies):.2f}s  p50: {percentile(latencies, 50):.2f}s  p90: {percentile(latencies, 90):.2f}s")
    print(f"Throughput: {req_tput:.2f} req/s  |  {audio_tput:.2f} audio-sec/s")
    if sum(token_counts) > 0:
        print(f"Token throughput: {token_tput:.2f} tokens/s")


def percentile(values, pct):
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[k]


if __name__ == "__main__":
    main()
