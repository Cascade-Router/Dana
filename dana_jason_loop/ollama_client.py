"""Minimal Ollama chat client for the Donna/Jason discovery loop (no agent.py imports)."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT_SEC = 180.0


def ask_ollama(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
) -> str:
    """Single isolated chat turn against local Ollama."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot reach Ollama at http://localhost:11434. "
            "Ensure Ollama is running and llama3.2 is pulled."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Ollama timed out after {OLLAMA_TIMEOUT_SEC:.0f}s") from exc

    content = str((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Ollama returned empty content: {data!r}")
    return content


def _repair_pseudo_json(raw: str) -> str:
    """Fix common llama3.2 JSON hygiene failures (triple-quotes, Trailing commas)."""
    text = raw
    # """...""" → "..." with escaped newlines/quotes (non-greedy, multiline).
    def _tri(m: re.Match[str]) -> str:
        inner = m.group(1)
        inner = inner.replace("\\", "\\\\").replace('"', '\\"')
        inner = inner.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        return f'"{inner}"'

    text = re.sub(r'"""([\s\S]*?)"""', _tri, text)
    text = re.sub(r"'''([\s\S]*?)'''", _tri, text)
    # Trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def extract_json_payload(text: str) -> Any:
    """Best-effort extract of a JSON array/object from an LLM reply."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model reply; cannot parse JSON")

    # Strip common markdown fences.
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I)
    if fenced:
        raw = fenced.group(1).strip()

    candidates = [raw, _repair_pseudo_json(raw)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # First array or object span.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start >= 0 and end > start:
            span = raw[start : end + 1]
            for candidate in (span, _repair_pseudo_json(span)):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Could not parse JSON from model reply:\n{text[:800]}")
