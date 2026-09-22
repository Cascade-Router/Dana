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
    images: list[tuple[str, str]] | None = None,
    system: str | None = None,
) -> list[dict[str, Any]]:
    """One user turn with zero or more inline images, as an OpenAI content
    array. ``images`` is an ordered list of ``(image_b64, mime_type)`` pairs
    — multiple images let a multi-view VLM prompt (e.g. orthographic
    front/top/side projections of the same part) cross-reference them in a
    single turn instead of describing each in isolation.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_b64, mime_type in images or ():
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
    "usage": dict | None, "finish_reason": str | None}`` — the exact shape
    ``dana.tools.schema.openai_tool_calls_to_ir``
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
    # Token-Truncation Trap: only the LAST chunk of a completion carries a
    # non-null finish_reason (every intermediate delta chunk has it as
    # null/absent) — captured here so a caller can distinguish "the model
    # finished naturally" from "cut off by max_tokens" (finish_reason ==
    # "length", or "MAX_TOKENS" for some OpenAI-compatible Gemini endpoints).
    finish_reason: str | None = None

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
                choice0 = choices[0] or {}
                delta = choice0.get("delta") or {}
                if choice0.get("finish_reason"):
                    finish_reason = str(choice0["finish_reason"])

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
                    # Gemini's OpenAI-compat endpoint attaches its opaque
                    # thought_signature here as {"google": {"thought_signature":
                    # "..."}} — an atomic signature, not text to accumulate
                    # char-by-char like name/arguments, so this takes whatever
                    # arrives whole rather than concatenating. Absent for
                    # every other provider (OpenAI, Groq, Ollama).
                    tc_extra = tc_delta.get("extra_content")
                    if isinstance(tc_extra, dict):
                        entry["extra_content"] = tc_extra
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
    return {
        "content": content,
        "tool_calls": tool_calls,
        "ttft_ms": ttft_ms,
        "usage": usage,
        "finish_reason": finish_reason,
    }


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


def _messages_for_ollama_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ollama's native ``/api/chat`` template expects a REPLAYED assistant
    ``tool_calls[].function.arguments`` to be a real JSON object — sending
    back the OpenAI-wire JSON-STRING encoding
    ``dana.core.react_dispatch.build_assistant_tool_call_message`` always
    produces (the shared ``messages`` history is built once, in that
    shape, for every provider including cloud ones) made Ollama's own
    parser choke on the very next turn (``HTTP 400: "Value looks like
    object, but can't find closing '}' symbol"``) — confirmed live during
    this function's own validation. Returns a NEW list; the caller's own
    ``messages``, shared with every other provider, is never mutated.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            out.append(message)
            continue
        new_calls: list[Any] = []
        for entry in tool_calls:
            fn = entry.get("function") if isinstance(entry, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {}
                new_calls.append({**entry, "function": {**fn, "arguments": parsed_args}})
            else:
                new_calls.append(entry)
        out.append({**message, "tool_calls": new_calls})
    return out


def complete_ollama_native_with_tools(
    messages: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    num_predict: int = 512,
    num_ctx: int = 32768,
    num_gpu: int | None = None,
    temperature: float = 0.1,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Ollama's NATIVE ``/api/chat`` surface — distinct from the OpenAI-
    compat ``/v1/chat/completions`` bridge above. Exists for one reason:
    the OpenAI-compat surface has no way to request a context-window size
    (``max_tokens`` only maps to output length, ``num_predict``); the
    native endpoint accepts ``num_ctx`` directly inside its ``options``
    object, which is the only way this process can raise it without every
    machine hand-editing a Modelfile. See ``dana.core.model_provider.
    ollama_num_ctx``.

    ``num_gpu`` (``dana.core.model_provider.ollama_num_gpu`` — number of
    model layers to offload to GPU) is ``None`` by default and then simply
    omitted from ``options``, leaving Ollama's own automatic VRAM-fit
    heuristic in charge exactly as before this parameter existed; only a
    caller that explicitly resolved ``DANA_OLLAMA_NUM_GPU`` passes a
    concrete value here.

    Returns the SAME ``{"content", "tool_calls", "ttft_ms", "usage",
    "finish_reason"}`` shape ``complete_openai_with_tools`` does, so
    ``ModelProvider.complete_with_tool_calls`` can call either
    interchangeably. ``tool_calls[i]["function"]["arguments"]`` is left as
    the dict Ollama itself returns, never re-serialized to a JSON string —
    ``dana.tools.schema.openai_tool_calls_to_ir`` already accepts either
    shape, and no wire ``id`` is preserved either way (Dana synthesizes its
    own — see ``dana.core.react_dispatch.build_assistant_tool_call_message``'s
    own docstring), so none is fabricated here.

    No ``tool_choice``: Ollama's native API has no equivalent knob (every
    call behaves like the OpenAI wire format's ``"auto"``).

    Streamed as newline-delimited JSON (Ollama's own convention, distinct
    from OpenAI's ``data: `` SSE framing) — one object per line, the last
    carrying ``"done": true`` plus ``prompt_eval_count``/``eval_count``
    (used for ``usage``). Unlike OpenAI's per-token tool-call deltas,
    Ollama emits a chunk's ``message.tool_calls`` whole, not
    character-by-character — accumulated by extending a list rather than
    concatenating strings.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        # Accept either the bare Ollama origin or the "/v1"-suffixed form
        # dana.core.model_provider._resolve_openai_endpoint already builds
        # for the OpenAI-compat path, so callers don't need a second,
        # native-specific base_url to track.
        root = root[: -len("/v1")]
    url = root + "/api/chat"

    payload: dict[str, Any] = {
        "model": model,
        "messages": _messages_for_ollama_native(messages),
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
        },
    }
    if num_gpu is not None:
        payload["options"]["num_gpu"] = int(num_gpu)
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )

    start = time.perf_counter()
    ttft_ms: float | None = None
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = chunk.get("message") or {}
                piece = message.get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    content_parts.append(piece)
                chunk_tool_calls = message.get("tool_calls")
                if chunk_tool_calls:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    tool_calls.extend(chunk_tool_calls)
                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason") or ("tool_calls" if tool_calls else "stop")
                    prompt_tokens = chunk.get("prompt_eval_count")
                    completion_tokens = chunk.get("eval_count")
                    if prompt_tokens is not None or completion_tokens is not None:
                        usage = {
                            "prompt_tokens": int(prompt_tokens or 0),
                            "completion_tokens": int(completion_tokens or 0),
                        }
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — body read is best-effort diagnostics only
            body = "<no response body available>"
        print(
            f"[openai_tool_bridge] ollama-native HTTP {exc.code} from {url!r} model={model!r}: "
            f"{exc.reason}\nresponse body: {body}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(f"ollama-native HTTP {exc.code}: {exc.reason} -- {body}") from exc
    except urllib.error.URLError as exc:
        raise TimeoutError(f"ollama-native endpoint unreachable or stalled: {exc.reason}") from exc

    content = "".join(content_parts) or None
    if tools and not tool_calls:
        # The exact qwen2.5-coder quirk _complete_openai_with_tools_once
        # already recovers from on the OpenAI-compat path (see
        # _fallback_tool_calls_from_content's own docstring — "verified
        # live against a running Ollama daemon") reproduces identically
        # over this native endpoint: confirmed live during this function's
        # own validation, the model emitted its tool call as a bare JSON
        # object in message.content with an empty native tool_calls array.
        tool_calls = _fallback_tool_calls_from_content(content)
        if tool_calls:
            content = ""  # it was a function call, not a reply meant for the user
    return {
        "content": content,
        "tool_calls": tool_calls,
        "ttft_ms": ttft_ms,
        "usage": usage,
        "finish_reason": finish_reason,
    }


__all__ = (
    "build_image_content_part",
    "build_multimodal_messages",
    "complete_ollama_native_with_tools",
    "complete_openai_with_tools",
    "encode_image_bytes",
)
