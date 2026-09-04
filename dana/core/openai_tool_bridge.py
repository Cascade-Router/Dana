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
import sys
import time
import urllib.error
import urllib.request
from typing import Any


# Cloudflare (fronting Groq's and many other providers' API endpoints)
# returns a 1010 "Forbidden" and drops the connection for requests with no
# User-Agent, or urllib's own default ("Python-urllib/x.y") — both read as
# a bot signature. Every external-cloud-API request in this module sends
# this explicitly; local Ollama calls (never behind Cloudflare) don't need
# it, but sending it there too is harmless.
_USER_AGENT = "Dana-Agent/1.0 (+https://github.com/; Python urllib)"

# Client-side TPM 429 throttle-and-retry (sleeping out Groq's own
# retry-after hint, e.g. "Please try again in 19.0725s") used to live here.
# Removed now that cloud tool-calling routes directly to a single provider
# (dana.core.model_provider.tool_calling_provider — OpenRouter by default),
# whose own server-side ``models`` fallback array (see
# complete_openai_with_tools's ``fallback_models``) retries the next model
# upstream in milliseconds — a 429/5xx reaching this bridge means that was
# already exhausted, so sleeping and retrying the identical request here
# would just be waiting out a limit already tried and failed upstream. Any
# HTTP error (429 included) is now treated as a standard fast failure — see
# _complete_openai_with_tools_once's HTTPError handling, and
# ModelProvider.complete_with_tool_calls's own Ollama fallback for what
# happens next.


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


def _scan_embedded_json_values(text: str) -> list[Any]:
    """Best-effort scan for JSON object/array literals embedded ANYWHERE in
    ``text`` — a further-degraded variant of the same quirk
    ``_fallback_tool_calls_from_content`` recovers from, where the model
    prefixes its structured call with prose (e.g. "Here's the coordinate...
    {"name": "create_freecad_cylinder", ...}") instead of emitting only
    JSON. Uses ``JSONDecoder.raw_decode`` at each ``{``/``[`` so trailing
    prose/other content after a valid JSON value doesn't break the parse —
    a plain ``json.loads`` would reject the whole string outright.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    idx = 0
    length = len(text)
    while idx < length:
        brace = text.find("{", idx)
        bracket = text.find("[", idx)
        candidates = [p for p in (brace, bracket) if p != -1]
        if not candidates:
            break
        start = min(candidates)
        try:
            value, end = decoder.raw_decode(text, start)
            values.append(value)
            idx = end
        except json.JSONDecodeError:
            idx = start + 1
    return values


def _fallback_tool_calls_from_content(content: str | None) -> list[dict[str, Any]]:
    """Recover a tool call some local Ollama models emit as plain JSON text
    in ``message.content`` instead of populating ``message.tool_calls`` —
    a real, observed quirk of qwen2.5-coder over the OpenAI-compat
    ``/v1/chat/completions`` shim (verified live against a running Ollama
    daemon), not a hypothetical. Tries the whole content as one JSON
    object/array first; falls back to scanning for JSON values embedded in
    surrounding prose (also observed live) before giving up. Returns ``[]``
    when nothing recoverable looks like a ``{"name": ..., "arguments": {...}}``
    shape.
    """
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        candidates: list[Any] = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, ValueError):
        candidates = _scan_embedded_json_values(text)

    calls: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fn = candidate.get("function") if isinstance(candidate.get("function"), dict) else candidate
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = fn.get("arguments")
        if not isinstance(arguments, (dict, str)):
            arguments = {}
        calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    return calls


def _complete_openai_with_tools_once(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    num_predict: int = 512,
    temperature: float = 0.1,
    timeout: float = 600.0,
    extra_headers: dict[str, str] | None = None,
    fallback_models: list[str] | None = None,
) -> dict[str, Any]:
    """Stream one ``/chat/completions`` turn; return the assembled ``message``.

    Returns ``{"content": str | None, "tool_calls": list[dict], "ttft_ms": float | None,
    "usage": dict | None}`` — the exact shape ``dana.tools.schema.openai_tool_calls_to_ir``
    and plain-text callers both need, so there is a single HTTP call site for
    text, tool-calling, and vision requests alike. ``ttft_ms`` is ``None``
    only if the stream ended with no content/tool-call delta at all (an
    empty completion). ``usage`` is ``None`` unless the endpoint actually
    honors ``stream_options.include_usage`` (OpenRouter and OpenAI both do;
    an endpoint that ignores the field simply never sends that final chunk,
    so this degrades to "cost unknown" rather than raising).

    Streamed (``"stream": True``) rather than one blocking request so
    ``ttft_ms`` reflects the model's REAL time-to-first-token — the signal
    a caller actually needs to detect a stalling local model (e.g. Ollama
    VRAM pressure) — instead of however long the entire turn takes to
    finish. ``timeout`` bounds each individual socket read, so a connection
    that goes silent mid-stream fails within ``timeout`` seconds of its
    LAST byte, not ``timeout`` seconds after the request started; callers
    wanting a hard ceiling on total turn latency (e.g. dana.core.
    react_dispatch's ``_call_llm_once``) wrap this call in their own
    ``asyncio.wait_for`` instead of relying on this parameter for that.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(num_predict),
        "stream": True,
        # Cost Tracking: asks for one extra SSE chunk at the end of the
        # stream carrying a "usage" object (prompt/completion token counts)
        # with an EMPTY "choices" array — OpenRouter and OpenAI both honor
        # this; see the "usage" capture below for why it must be read
        # BEFORE the `if not choices: continue` skip.
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if fallback_models:
        # OpenRouter's native server-side model cascade — equivalent to the
        # official OpenAI SDK's ``extra_body={"models": [...]}`` for a
        # ``chat.completions.create`` call, just written directly into this
        # bridge's own raw JSON body since it has no SDK client underneath.
        # ``payload["model"]`` stays the primary; on a 429/5xx OpenRouter
        # itself retries each entry in ``models`` next, in order, with no
        # round trip back to this process.
        payload["models"] = list(fallback_models)

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _USER_AGENT,
    }
    # Provider-specific attribution/routing headers (e.g. OpenRouter's
    # recommended HTTP-Referer/X-Title) — this module stays provider-
    # agnostic on purpose, so the caller (dana.core.model_provider, which
    # already knows which provider it resolved) decides what goes here.
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    ttft_ms: float | None = None
    content_parts: list[str] = []
    # Keyed by the streamed delta's own "index" (OpenAI's multi-tool-call
    # streaming convention) — a tool call's name/arguments can arrive split
    # across many chunks, so each index accumulates independently until the
    # stream ends.
    tool_call_parts: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # The usage-only final chunk (stream_options.include_usage)
                # carries "usage" alongside an EMPTY "choices" array — must
                # be captured here, before the empty-choices skip below
                # would otherwise silently discard it every time.
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}

                piece = delta.get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    content_parts.append(piece)

                for tc_delta in delta.get("tool_calls") or []:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    idx = int(tc_delta.get("index") or 0)
                    entry = tool_call_parts.setdefault(
                        idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    fn_delta = tc_delta.get("function") or {}
                    if fn_delta.get("name"):
                        entry["function"]["name"] += fn_delta["name"]
                    if fn_delta.get("arguments"):
                        entry["function"]["arguments"] += fn_delta["arguments"]
    except urllib.error.HTTPError as exc:
        # The generic exception handler upstream (dana.core.react_dispatch's
        # next_react_turn) only ever sees str(exc) get discarded into a UI
        # message like "I ran into a problem talking to the model" — the
        # provider's OWN rejection reason (a 400 "too many tools"/context
        # error, a 429 rate-limit body with the exact TPM numbers, etc.)
        # lives in the response body, which urllib does NOT include in
        # exc.reason. Read it here (best-effort — a already-consumed or
        # unreadable body must never mask the original HTTPError) and log
        # it to stderr so it survives even when the caller only logs
        # str(exception), then fold it into the raised message too.
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — body read is best-effort diagnostics only
            body = "<no response body available>"
        print(
            f"[openai_tool_bridge] cloud HTTP {exc.code} from {base_url!r} model={model!r}: "
            f"{exc.reason}\nresponse body: {body}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(f"cloud HTTP {exc.code}: {exc.reason} -- {body}") from exc
    except urllib.error.URLError as exc:
        # Covers a stalled/silent connection (socket.timeout surfaces here,
        # wrapped by urllib) as well as connection-refused — both are a
        # "the endpoint didn't respond in time" failure from this caller's
        # point of view, so both raise the same TimeoutError a caller's
        # asyncio.wait_for-based fallback logic already expects.
        raise TimeoutError(f"model endpoint unreachable or stalled: {exc.reason}") from exc

    content = "".join(content_parts) or None
    tool_calls = [tool_call_parts[i] for i in sorted(tool_call_parts)]
    if tools and not tool_calls:
        tool_calls = _fallback_tool_calls_from_content(content)
        if tool_calls:
            content = ""  # it was a function call, not a reply meant for the user
    return {"content": content, "tool_calls": tool_calls, "ttft_ms": ttft_ms, "usage": usage}


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
    timeout: float = 600.0,
    extra_headers: dict[str, str] | None = None,
    fallback_models: list[str] | None = None,
) -> dict[str, Any]:
    """Public entry point every caller (``dana.core.model_provider``)
    actually uses. A plain passthrough to ``_complete_openai_with_tools_once``
    — no client-side sleep/retry loop. A 429/5xx here means the request to
    the resolved provider failed outright, so this raises immediately as a
    standard failure (a plain ``RuntimeError``) rather than sleeping and
    retrying; whatever caller-side fallback exists for a real outage (e.g.
    ``ModelProvider.complete_with_tool_calls`` routing to local Ollama) sees
    it right away.

    ``fallback_models``, when given, rides in the request body as
    OpenRouter's own ``models`` cascade array — this runs entirely on
    OpenRouter's servers for a single ``model=`` provider choice, so a
    429/5xx on the primary model retries the next one upstream in
    milliseconds with no round trip back to this process.
    """
    return _complete_openai_with_tools_once(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        num_predict=num_predict,
        temperature=temperature,
        timeout=timeout,
        extra_headers=extra_headers,
        fallback_models=fallback_models,
    )


__all__ = (
    "build_image_content_part",
    "build_multimodal_messages",
    "complete_openai_with_tools",
    "encode_image_bytes",
)
