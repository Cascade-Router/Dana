"""Retry-parser middleware for strict JSON / Pydantic LLM outputs.

If the model emits malformed JSON or fails schema validation, append an error
observation to a temporary context and force another generation (up to
``max_retries`` times) before failing the node.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_RETRIES = 3
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    """Raised when structured parsing fails after all retries."""

    def __init__(self, message: str, *, attempts: int, last_raw: str = "") -> None:
        super().__init__(message)
        self.attempts = int(attempts)
        self.last_raw = last_raw or ""


def extract_json_payload(text: str) -> str:
    """Best-effort unwrap of fenced / prose-wrapped JSON."""
    raw = (text or "").strip()
    if not raw:
        raise json.JSONDecodeError("empty model output", raw, 0)
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    # If the payload already starts with a JSON value, keep the matching span.
    if raw.startswith("{"):
        end = raw.rfind("}")
        if end > 0:
            return raw[: end + 1]
    if raw.startswith("["):
        end = raw.rfind("]")
        if end > 0:
            return raw[: end + 1]
    # Prose-wrapped: prefer the earliest JSON value opener.
    obj_i = raw.find("{")
    arr_i = raw.find("[")
    candidates: list[tuple[int, str, str]] = []
    if obj_i >= 0:
        candidates.append((obj_i, "{", "}"))
    if arr_i >= 0:
        bracket_open = "["
        bracket_close = "]"
        candidates.append((arr_i, bracket_open, bracket_close))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        _, opener, closer = candidates[0]
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start >= 0 and end > start:
            return raw[start : end + 1]
    return raw


def parse_model(text: str, model: type[T]) -> T:
    """Parse ``text`` into ``model``; raise JSONDecodeError / ValidationError."""
    payload = extract_json_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise
    # Allow bare TaskNode lists for DAGPlan convenience.
    if model.__name__ == "DAGPlan" and isinstance(data, list):
        data = {"tasks": data}
    return model.model_validate(data)


InvokeFn = Callable[[list[dict[str, str]]], str]


def parse_with_schema_retry(
    messages: list[dict[str, str]],
    model: type[T],
    *,
    invoke: InvokeFn,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> T:
    """Invoke ``invoke(messages)`` and parse; retry on JSON/schema failures.

    On failure, appends the assistant dump + a user error observation to a
    temporary copy of ``messages`` (original list is not mutated) and retries.
    """
    retries = max(0, int(max_retries))
    temp = [dict(m) for m in messages]
    last_raw = ""
    last_err = "unknown parse error"
    attempts = 0

    for attempt in range(retries + 1):
        attempts = attempt + 1
        raw = invoke(temp)
        last_raw = raw if isinstance(raw, str) else str(raw)
        try:
            return parse_model(last_raw, model)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            temp.append({"role": "assistant", "content": last_raw})
            topo_hint = ""
            if "Topological Error" in last_err:
                topo_hint = (
                    " Topological fix required: at least one task must have "
                    "dependencies: []; every dependency id must exist; "
                    "remove circular dependencies (A -> B -> A)."
                )
            temp.append(
                {
                    "role": "user",
                    "content": (
                        "ERROR OBSERVATION: previous output failed strict JSON schema "
                        f"validation ({last_err}).{topo_hint} "
                        f"Return ONLY valid JSON matching schema {model.__name__}. "
                        "No markdown fences, no prose."
                    ),
                }
            )

    raise StructuredOutputError(
        f"structured output failed after {attempts} attempt(s): {last_err}",
        attempts=attempts,
        last_raw=last_raw,
    )


__all__ = (
    "DEFAULT_MAX_RETRIES",
    "StructuredOutputError",
    "extract_json_payload",
    "parse_model",
    "parse_with_schema_retry",
)
