"""OpenAI `/v1/chat/completions` wire-format bridge — tool calling + vision.

One HTTP call site shared by ``dana.core.model_provider.ModelProvider`` for
three use cases that all speak the same OpenAI-compatible schema: plain-text
completion, native ``tools=[...]``/``tool_calls`` function calling, and
multimodal (``image_url``) vision prompts. Works against any endpoint that
implements this wire format — OpenAI, Groq, and local Ollama's
``/v1/chat/completions`` surface alike — the caller only supplies a
different ``base_url``/``model``/``api_key`` triple.

This module has no Dana-internal dependencies besides the tool IR
(``dana.tools.schema``), so it is always safe to import at module scope.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any


def build_image_content_part(image_b64: str, *, mime_type: str = "image/png") -> dict[str, Any]:
    """OpenAI ``image_url`` content part from raw base64 image data."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
    }


def build_multimodal_messages(
    prompt: str,
    *,
    image_b64: str | None = None,
    mime_type: str = "image/png",
    system: str | None = None,
) -> list[dict[str, Any]]:
    """One user turn with an optional inline image, as an OpenAI content array."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append(build_image_content_part(image_b64, mime_type=mime_type))
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def encode_image_bytes(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode("ascii")


def complete_openai_with_tools(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    num_predict: int = 512,
    temperature: float = 0.1,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """POST one ``/chat/completions`` turn; return the raw response ``message``.

    Returns ``{"content": str | None, "tool_calls": list[dict]}`` — the
    exact shape ``dana.tools.schema.openai_tool_calls_to_ir`` and
    plain-text callers both need, so there is a single HTTP call site for
    text, tool-calling, and vision requests alike.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(num_predict),
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"cloud HTTP {exc.code}: {exc.reason}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("cloud response missing choices")
    message = (choices[0] or {}).get("message") or {}
    return {
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }


__all__ = (
    "build_image_content_part",
    "build_multimodal_messages",
    "complete_openai_with_tools",
    "encode_image_bytes",
)
