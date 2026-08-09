"""Optional cloud-assisted DAG / epic planning (Gemini). Workers stay local."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

_logger = logging.getLogger("dana.graph.cloud_planner")

T = TypeVar("T", bound=BaseModel)

_CLOUD_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY")
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

# Exponential backoff for 429 / 503 (Worker Escalation + planner).
_CLOUD_THROTTLE_MAX_RETRIES = 5
_CLOUD_THROTTLE_MAX_WAIT_S = 30.0
_THROTTLE_HITS = 0
_THROTTLE_RETRIES_OK = 0


def reset_cloud_throttle_stats() -> None:
    """Reset process-local 429/503 counters (Suite diagnostics)."""
    global _THROTTLE_HITS, _THROTTLE_RETRIES_OK
    _THROTTLE_HITS = 0
    _THROTTLE_RETRIES_OK = 0


def get_cloud_throttle_stats() -> dict[str, int]:
    """Return counts of throttle hits and successful post-retry completions."""
    return {
        "throttle_hits": int(_THROTTLE_HITS),
        "throttle_retries_ok": int(_THROTTLE_RETRIES_OK),
    }


def ensure_dotenv_loaded() -> None:
    """Best-effort ``.env`` load so API keys are visible in ``os.environ``."""
    try:
        from dotenv import load_dotenv

        from dana.paths import ENV_PATH

        load_dotenv(ENV_PATH)
        load_dotenv()
    except Exception:  # noqa: BLE001
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:  # noqa: BLE001
            pass


def cloud_planner_api_key() -> str:
    """Return the first configured Gemini/Google AI key, or empty string."""
    ensure_dotenv_loaded()
    for name in _CLOUD_KEY_ENVS:
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return raw
    return ""


def cloud_planner_key_present() -> bool:
    return bool(cloud_planner_api_key())


def hybrid_cloud_planner_active() -> bool:
    """True only when the UI toggle is on AND a cloud API key is present."""
    if (os.environ.get("DANA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    try:
        from dana.settings import is_hybrid_planner_enabled

        if not is_hybrid_planner_enabled():
            return False
    except Exception:  # noqa: BLE001
        return False
    return cloud_planner_key_present()


def planner_mode_label() -> str:
    """UI / monitor label: LOCAL vs HYBRID CLOUD (or hybrid requested but keyless)."""
    if (os.environ.get("DANA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "LOCAL"
    try:
        from dana.settings import is_hybrid_planner_enabled

        wanted = bool(is_hybrid_planner_enabled())
    except Exception:  # noqa: BLE001
        wanted = False
    if wanted and cloud_planner_key_present():
        return "HYBRID CLOUD"
    return "LOCAL"


def publish_planner_mode(*, warn_missing_key: bool = False) -> str:
    """Push planner mode to the DAG monitor bus; optionally warn about missing keys."""
    mode = planner_mode_label()
    try:
        from dana.graph.monitor_bus import get_monitor_bus, publish_tool_line

        bus = get_monitor_bus(create=True)
        if bus is not None:
            bus.publish("status", status=f"planner_{mode.lower().replace(' ', '_')}")
            bus.publish("telemetry", planner_mode=mode)
        publish_tool_line(f"Planner Mode: [{mode}]")
        if warn_missing_key:
            try:
                from dana.settings import is_hybrid_planner_enabled

                wanted = bool(is_hybrid_planner_enabled())
            except Exception:  # noqa: BLE001
                wanted = False
            if wanted and not cloud_planner_key_present():
                msg = (
                    "Hybrid Broker enabled but no GEMINI_API_KEY / GOOGLE_API_KEY "
                    "in environment — falling back to LOCAL Ollama planner."
                )
                _logger.warning(msg)
                publish_tool_line(f"[GRAPH WARNING] {msg}")
    except Exception:  # noqa: BLE001
        pass
    return mode


def _gemini_model_id() -> str:
    return (
        (os.environ.get("DANA_GEMINI_MODEL") or "").strip()
        or (os.environ.get("GEMINI_MODEL") or "").strip()
        or _DEFAULT_GEMINI_MODEL
    )


def ask_gemini_text(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    response_mime_type: str | None = "application/json",
) -> str:
    """Call Gemini generateContent (REST) and return the text response.

    ``response_mime_type`` defaults to JSON for structured planner calls.
    Pass ``None`` (or ``\"text/plain\"``) for free-form code generation.
    """
    from dana.system_health import llm_lock

    with llm_lock:
        return _ask_gemini_text_unlocked(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )


def _ask_gemini_text_unlocked(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_output_tokens: int,
    response_mime_type: str | None,
) -> str:
    key = cloud_planner_api_key()
    if not key:
        raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY configured")

    system_bits: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        text = str(m.get("content") or "")
        if role == "system":
            system_bits.append(text)
            continue
        gem_role = "model" if role in {"assistant", "model"} else "user"
        contents.append({"role": gem_role, "parts": [{"text": text}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "\n".join(system_bits) or ""}]}]

    model = _gemini_model_id()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    gen_cfg: dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    if response_mime_type:
        gen_cfg["responseMimeType"] = str(response_mime_type)
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": gen_cfg,
    }
    if system_bits:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_bits)}]
        }

    global _THROTTLE_HITS, _THROTTLE_RETRIES_OK
    throttle_retries = 0
    while True:
        resp = requests.post(
            url,
            params={"key": key},
            json=payload,
            timeout=90,
        )
        if resp.status_code in {429, 503}:
            _THROTTLE_HITS += 1
            if throttle_retries >= _CLOUD_THROTTLE_MAX_RETRIES:
                resp.raise_for_status()
            wait = min(
                _CLOUD_THROTTLE_MAX_WAIT_S,
                float(2**throttle_retries),
            )
            # 2**0=1 … 2**4=16, capped at 30s; up to 5 retries.
            print(
                f"[Cloud API Throttled] Retrying in {wait:g}s...",
                flush=True,
            )
            time.sleep(wait)
            throttle_retries += 1
            continue
        resp.raise_for_status()
        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict)]
            out = "".join(texts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {exc}") from exc
        if not out:
            raise RuntimeError("Gemini returned empty content")
        if throttle_retries > 0:
            _THROTTLE_RETRIES_OK += 1
            print(
                f"[Cloud API Throttled] recovered after {throttle_retries} retry(ies)",
                flush=True,
            )
        return out


def ask_cloud_structured(
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> T:
    """Schema-validated cloud planner invoke (Gemini JSON + retry middleware)."""
    from dana.middleware.json_schema_retry import parse_with_schema_retry

    schema = response_model.model_json_schema()
    schema_hint = json.dumps(schema, ensure_ascii=False)[:3500]

    def _invoke(msgs: list[dict[str, str]]) -> str:
        augmented = list(msgs)
        if not any("JSON Schema" in str(m.get("content") or "") for m in augmented):
            augmented = [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON matching this JSON Schema "
                        f"(no markdown):\n{schema_hint}"
                    ),
                },
                *augmented,
            ]
        return ask_gemini_text(
            augmented,
            temperature=temperature,
            max_output_tokens=2048,
        )

    return parse_with_schema_retry(
        messages,
        response_model,
        invoke=_invoke,
        max_retries=max_retries,
    )


__all__ = (
    "ask_cloud_structured",
    "ask_gemini_text",
    "cloud_planner_api_key",
    "cloud_planner_key_present",
    "ensure_dotenv_loaded",
    "get_cloud_throttle_stats",
    "hybrid_cloud_planner_active",
    "planner_mode_label",
    "publish_planner_mode",
    "reset_cloud_throttle_stats",
)
