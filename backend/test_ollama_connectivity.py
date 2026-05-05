#!/usr/bin/env python3
"""
Quick Ollama diagnostics for remote servers.

Usage:
  python backend/test_ollama_connectivity.py --model qwen3.5:4b
  python backend/test_ollama_connectivity.py --base-url http://127.0.0.1:11434 --timeout 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import requests


def normalize_base_url(url: str) -> str:
    value = (url or "").strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    return value


def check_ollama_tags(base_url: str, timeout_s: float) -> List[str]:
    url = f"{base_url}/api/tags"
    start = time.perf_counter()
    resp = requests.get(url, timeout=timeout_s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()

    payload: Dict[str, Any] = resp.json() or {}
    models = payload.get("models", []) or []
    names = [m.get("name", "") for m in models if isinstance(m, dict)]

    print(f"[OK] Ollama tags endpoint reachable: {url} ({elapsed_ms:.1f} ms)")
    print(f"[INFO] Models found: {len(names)}")
    for name in names:
        print(f"  - {name}")
    return names


def check_chat_completion(base_url: str, model: str, timeout_s: float, prompt: str) -> None:
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }

    start = time.perf_counter()
    resp = requests.post(url, json=body, timeout=timeout_s)
    elapsed_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()

    payload = resp.json() or {}
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    print(f"[OK] Chat completion returned in {elapsed_ms:.1f} ms")
    print(f"[INFO] Response preview: {str(content)[:300]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Ollama connectivity and response behavior")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        help="Ollama base URL. Supports values with or without /v1",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_TEST_MODEL", "qwen3.5:4b"),
        help="Model to test",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OLLAMA_TEST_TIMEOUT_S", "45")),
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Test prompt",
    )
    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)
    print(f"[INFO] Base URL: {base_url}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Timeout: {args.timeout}s")

    try:
        model_names = check_ollama_tags(base_url, args.timeout)
    except requests.Timeout:
        print("[FAIL] Timeout while calling /api/tags. Ollama may be down, blocked, or too slow.")
        return 2
    except requests.RequestException as exc:
        print(f"[FAIL] Cannot reach Ollama /api/tags: {exc}")
        return 2

    if args.model not in model_names:
        print(f"[WARN] Requested model '{args.model}' was not listed by /api/tags.")
        print("[WARN] The completion test may fail unless the model name is correct or already pulled.")

    try:
        check_chat_completion(base_url, args.model, args.timeout, args.prompt)
    except requests.Timeout:
        print("[FAIL] Timeout during chat completion. Server is reachable, but model inference is hanging/slow.")
        return 3
    except requests.RequestException as exc:
        print(f"[FAIL] Chat completion request failed: {exc}")
        return 3
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"[FAIL] Unexpected response payload format: {exc}")
        return 4

    print("[PASS] Ollama is reachable and returned a completion within timeout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
