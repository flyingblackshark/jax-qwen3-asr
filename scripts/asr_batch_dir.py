#!/usr/bin/env python3
import argparse
import base64
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


DEFAULT_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".wma",
}


LANG_PATTERNS = [
    re.compile(r"^\s*language\s*([a-zA-Z\-]{2,32})\s*<asr_text>\s*(.*)$", re.I),
    re.compile(r"^\s*<\|([a-zA-Z\-]{2,8})\|>\s*(.*)$"),
    re.compile(r"^\s*\[([a-zA-Z\-]{2,8})\]\s*(.*)$"),
    re.compile(r"^\s*lang(?:uage)?\s*[:=]\s*([a-zA-Z\-]{2,8})\s*(.*)$", re.I),
    re.compile(r"^\s*([a-zA-Z\-]{2,8})\s*:\s*(.*)$"),
]


def _extract_language_and_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    for pat in LANG_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).lower(), m.group(2).strip()
    return "", text.strip()


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
    except Exception:
        return ""
    try:
        return detect(text)
    except Exception:
        return ""


def _build_payload(model: str, b64_audio: str, max_tokens: int, temperature: float) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/mp3;base64," + b64_audio},
                    }
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _send_one(
    url: str,
    model: str,
    audio_path: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    detect_lang: bool,
) -> tuple[bool, str, str]:
    try:
        raw = open(audio_path, "rb").read()
        b64 = base64.b64encode(raw).decode()
        payload = _build_payload(model, b64, max_tokens, temperature)
        resp = requests.post(url, json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            return False, "", f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        lang, transcript = _extract_language_and_text(text)
        if detect_lang and not lang:
            lang = _detect_language(transcript)
        return True, lang, transcript
    except Exception as e:
        return False, "", str(e)


def _scan_files(root: str, exts: set[str]) -> list[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                matches.append(os.path.join(dirpath, name))
    matches.sort()
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch ASR over a directory")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Use langdetect (if installed) when model output has no language tag.",
    )
    parser.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Comma-separated list of audio extensions to include.",
    )
    args = parser.parse_args()

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if not exts:
        exts = DEFAULT_EXTS

    files = _scan_files(args.input_dir, exts)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _send_one,
                url,
                args.model,
                path,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.detect_language,
            ): path
            for path in files
        }
        for fut in as_completed(futures):
            path = futures[fut]
            ok, lang, transcript = fut.result()
            rel = os.path.relpath(path, args.input_dir)
            if not ok:
                lang = ""
                transcript = f"[ERROR] {transcript}"
            rows.append((rel, lang, transcript))

    rows.sort(key=lambda r: r[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "language", "transcript"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
