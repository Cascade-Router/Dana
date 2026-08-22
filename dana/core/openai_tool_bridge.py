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
import re
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

# Retrying a Groq TPM 429 against a DIFFERENT model used to live here
# (mixtral-8x7b-32768, before it llama-3.1-8b-instant, llama3-8b-8192,
# llama-3.3-70b-versatile, openai/gpt-oss-20b) — a whack-a-mole anti-pattern:
# every one of those names has since been decommissioned/404'd on Groq, or
# (openai/gpt-oss-20b) turned out to share the SAME 8,000 TPM pool as the
# primary model, so it never actually recovered the turn. A model name is
# not a stable thing to hardcode a retry path on. Groq's 429 body itself
# already tells us exactly how long the SAME model's TPM window needs to
# drain (e.g. "Please try again in 19.0725s") — waiting that out and
# retrying the identical request against the identical (primary) model is
# both simpler and actually reliable, since it depends on nothing but the
# provider's own math. See _parse_retry_after_seconds/complete_openai_with_tools.
_MAX_TPM_RETRIES = 2

# Groq's own retry hint is trusted verbatim up to this ceiling — past it,
# sleeping the turn that long is worse than just degrading gracefully now
# (and guards against ever blocking on a nonsensical/corrupted hint).
_MAX_REASONABLE_RETRY_AFTER_SEC = 45.0

# Groq's tokens-per-minute 429 body's "error.message" ends with this exact
# phrase (live-observed) — e.g. "...Requested 3200. Please try again in
# 19.0725s. Visit https://console.groq.com/...". Anchored loosely (no
# trailing "s" requirement variant beyond this) since it's the only shape
# ever seen; returns None rather than guessing when the phrasing changes.
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*s\b", re.IGNORECASE)


def _parse_retry_after_seconds(body: str) -> float | None:
    """Extract Groq's own suggested wait time from a TPM 429 body's
    ``error.message`` (see ``_RETRY_AFTER_RE``). Returns ``None`` for
    anything that isn't the exact expected shape — an unparseable body, a
    missing message, or phrasing that doesn't contain "try again in Ns" —
    so the caller never sleeps on a guessed number.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return None
    match = _RETRY_AFTER_RE.search(message)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class _TpmRateLimitError(RuntimeError):
    """Raised only for a confirmed Groq tokens-per-minute 429 — deliberately
    a DIFFERENT exception type than the plain RuntimeError every other
    HTTPError raises, so complete_openai_with_tools can catch exactly this
    case for its throttle-and-retry loop without accidentally swallowing
    (and retrying) a 400 Bad Request or an unrelated 429 (e.g. a
    requests-per-minute cap, which waiting out a TPM window would not fix).

    Carries the raw response ``body`` (not just a formatted message string)
    so the retry loop can parse the provider's own retry-after hint out of
    it without re-deriving it from a str(exc) that also has the HTTP status
    line mixed in.
    """

    def __init__(self, message: str, *, body: str = "") -> None:
        super().__init__(message)
        self.body = body


def _is_tpm_rate_limit(status_code: int, body: str) -> bool:
    """True only for Groq's tokens-per-minute rate limit shape:
    ``{"error": {"code": "rate_limit_exceeded", "type": "tokens", ...}}``
    on a 429. Deliberately narrow — a 429 with a different ``type`` (e.g.
    ``"requests"``, a requests-per-minute cap) is NOT a case waiting out a
    TPM window fixes, so it must fall through to the normal RuntimeError
    path, not trigger the throttle-and-retry loop.
    """
    if status_code != 429:
        return False
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(error, dict):
        return False
    return error.get("code") == "rate_limit_exceeded" and error.get("type") == "tokens"


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
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Stream one ``/chat/completions`` turn; return the assembled ``message``.

    Returns ``{"content": str | None, "tool_calls": list[dict], "ttft_ms": float | None}``
    — the exact shape ``dana.tools.schema.openai_tool_calls_to_ir`` and
    plain-text callers both need, so there is a single HTTP call site for
    text, tool-calling, and vision requests alike. ``ttft_ms`` is ``None``
    only if the stream ended with no content/tool-call delta at all (an
    empty completion).

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
            "User-Agent": _USER_AGENT,
        },
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
        if _is_tpm_rate_limit(exc.code, body):
            raise _TpmRateLimitError(f"cloud HTTP {exc.code}: {exc.reason} -- {body}", body=body) from exc
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
    return {"content": content, "tool_calls": tool_calls, "ttft_ms": ttft_ms}


# Shown to the user only once the TPM throttle-and-retry loop below has
# exhausted _MAX_TPM_RETRIES retries against the SAME (primary) model — see
# _degraded_summary_response. Deliberately hedged rather than asserting
# success: this can fire on the VERY FIRST ReAct turn (before any tool has
# run at all), not only after a later summarization turn, so the message
# must not claim the task completed.
_DEGRADED_SUMMARY_MESSAGE = (
    "The LLM rate limit was reached. Some operations may not have completed successfully."
)


def _degraded_summary_response() -> dict[str, Any]:
    """A synthesized stand-in for a real model turn, in the exact shape
    every other return value from this module uses. Used ONLY as the last
    resort once the primary model has kept hitting its TPM rate limit
    through ``_MAX_TPM_RETRIES`` retries, or handed back a retry-after hint
    too large/absent to safely wait out. This can happen on ANY ReAct turn,
    not just a trailing summarization step, so _DEGRADED_SUMMARY_MESSAGE
    deliberately does NOT assert the task succeeded. Surfacing a hard crash
    here instead (which dana.core.react_dispatch's next_react_turn would
    turn into the generic "I ran into a problem talking to the model" UI
    message) would still be worse: it gives the user nothing to act on,
    where this at least tells them the turn is uncertain rather than
    silently wrong in either direction.

    ``tool_calls`` is deliberately empty: dana.core.react_dispatch.
    next_react_turn treats an empty-tool_calls result exactly like any
    other plain-text reply and ends the loop with a "final" turn, so the
    DAG completes cleanly instead of erroring. ``ttft_ms`` is ``None``
    (not 0.0) since no real request actually completed — a fabricated
    duration would be a lie to whatever telemetry reads this field.
    """
    return {"content": _DEGRADED_SUMMARY_MESSAGE, "tool_calls": [], "ttft_ms": None}


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
    """Same contract as ``_complete_openai_with_tools_once`` — this is the
    public entry point every caller (``dana.core.model_provider``) actually
    uses. On top of a plain passthrough, this implements a state-aware
    pause/throttle for Groq's tokens-per-minute 429: rather than switching
    to a different (and inevitably, eventually, decommissioned/repooled)
    model name, it reads the provider's OWN retry-after hint out of the 429
    body (``_parse_retry_after_seconds`` — e.g. "Please try again in
    19.0725s"), sleeps that long, and retries the IDENTICAL request against
    the SAME model — up to ``_MAX_TPM_RETRIES`` times.

    Only a confirmed TPM 429 (``_TpmRateLimitError``) is retried here. Every
    other failure shape — a 400, a non-TPM 429, a 5xx, or a network
    timeout (``TimeoutError``) — re-raises immediately, unretried, so
    whatever caller-side fallback exists for an actual outage (e.g. routing
    to local Ollama) still sees it and can act; this function never masks
    those behind a throttle-retry loop.

    The retry-after hint is only trusted when it's actually present and
    below ``_MAX_REASONABLE_RETRY_AFTER_SEC`` — an absent, unparseable, or
    implausibly large hint goes straight to ``_degraded_summary_response()``
    instead of blocking the turn on a guess. Exhausting
    ``_MAX_TPM_RETRIES`` retries against the primary model does the same:
    this never raises a second failure into
    ``dana.core.react_dispatch.next_react_turn``'s generic "I ran into a
    problem talking to the model" UI message — it returns a synthesized
    plain-text reply so the ReAct loop still completes the turn.
    """
    attempt = 0
    while True:
        try:
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
            )
        except _TpmRateLimitError as exc:
            attempt += 1
            retry_after = _parse_retry_after_seconds(exc.body)
            if (
                retry_after is None
                or retry_after >= _MAX_REASONABLE_RETRY_AFTER_SEC
                or attempt > _MAX_TPM_RETRIES
            ):
                print(
                    f"[openai_tool_bridge] TPM limit on {model!r} -- giving up after "
                    f"{attempt} attempt(s) (retry_after={retry_after!r}) -- returning a "
                    "degraded summary instead of crashing the ReAct loop.",
                    file=sys.stderr,
                    flush=True,
                )
                return _degraded_summary_response()
            sleep_for = retry_after + 1.0
            print(
                f"[openai_tool_bridge] TPM limit on {model!r} (attempt {attempt}/"
                f"{_MAX_TPM_RETRIES}) -- throttling for {sleep_for:.1f}s per Groq's own "
                "retry hint, then retrying the identical request against the same model.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_for)


__all__ = (
    "build_image_content_part",
    "build_multimodal_messages",
    "complete_openai_with_tools",
    "encode_image_bytes",
)
