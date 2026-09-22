"""Hybrid ModelProvider — local Ollama with optional encrypted cloud fallback."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Any, Literal

import requests

from dana.config import LLM_MAX_OUTPUT_TOKENS
from dana.core.openai_tool_bridge import (
    _USER_AGENT,
    build_multimodal_messages,
    complete_ollama_native_with_tools,
    complete_openai_with_tools,
)
from dana.core.pricing import estimate_cost_usd
from dana.system_health import llm_lock
from dana.tools.schema import openai_tool_calls_to_ir

ProviderKind = Literal["local", "cloud", "auto"]

# Providers whose tool-calling / vision schema is not OpenAI-wire-compatible.
# Gemini and Anthropic each use their own function-calling and image payload
# shapes; bridging them is out of scope for the OpenAI tool-calling bridge.
_NON_OPENAI_SCHEMA_PROVIDERS = frozenset({"gemini", "google", "anthropic"})

_DEFAULT_LOCAL_MODEL = "qwen2.5-coder:14b"
# Separate default from _DEFAULT_LOCAL_MODEL on purpose: that one is a
# text/tool-calling model (e.g. Qwen2.5-Coder), and Ollama's own OpenAI-
# compat surface rejects a multimodal request outright with an HTTP 400
# when the resolved model isn't a VLM ("Multimodal data provided, but model
# does not support multimodal requests.") — confirmed live against
# qwen2.5-coder:14b. complete_vision's "ollama" routing needs its own model
# name so a local text-model choice never silently breaks every vision tool.
_DEFAULT_LOCAL_VISION_MODEL = "llava:7b"
_COMPLEXITY_REJECT = "REJECT: Task too complex for local model"

# Native Gemini generateContent (REST) — a DIFFERENT calling convention than
# the OpenAI-compatible "gemini_openai" provider above (its own request/
# response shape, not OpenAI-wire tool_calls). Extracted from the now-
# removed legacy dana.graph.cloud_planner module, which this is the only
# live-stack caller of: cloud_provider_name() defaults to "gemini", so this
# is the plain-text cloud-fallback path complete()/_complete_cloud actually
# hits by default, not a legacy-only remnant.
_GEMINI_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY")
# gemini-2.0-flash was retired by Google (confirmed via debug_gemini.py:
# HTTP 404 "This model models/gemini-2.0-flash is no longer available.
# Please update your code to use models/gemini-3.6-flash") — this default is
# what _ask_gemini_text_native actually calls whenever no DANA_GEMINI_MODEL/
# GEMINI_MODEL override is set, so a stale value here fails every native
# Gemini request with a 404, not an auth/region error.
_DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
_GEMINI_THROTTLE_MAX_RETRIES = 5
_GEMINI_THROTTLE_MAX_WAIT_S = 30.0


def ensure_dotenv_loaded() -> None:
    """Reload ``.env`` into ``os.environ`` — called at the top of every
    provider-resolving function (``cloud_provider_name``, ``local_model_name``,
    ``_resolve_openai_endpoint``, ...), not once at process startup, precisely
    so a hand-edit to ``.env`` takes effect on this process's very next call,
    no restart needed — the exact same "no backend restart needed" promise
    ``dana.api.system.save_system_env`` already makes for its own Settings-
    modal writes (it updates ``.env`` AND ``os.environ`` together in one
    request for that reason).

    ``override=True`` is required for that promise to actually hold:
    ``load_dotenv()`` defaults to ``override=False`` (never replacing a key
    already present in ``os.environ``), which silently breaks it — a value
    ``os.environ`` picked up ONCE (an earlier ``.env`` load from before an
    edit, or a prior ``save_system_env`` write from a since-reverted Settings
    change) then wins forever, no matter how many times ``.env`` is corrected
    afterward, until the process is restarted. This is exactly the reported
    bug: ``.env`` read ``DANA_CLOUD_PROVIDER=openai`` but the running
    process kept resolving "gemini" — a stale ``os.environ`` entry from
    earlier in that process's life, which the default ``override=False``
    reload could never dislodge.
    """
    try:
        from dotenv import load_dotenv

        from dana.paths import ENV_PATH

        load_dotenv(ENV_PATH, override=True)
        load_dotenv(override=True)
    except Exception:  # noqa: BLE001
        pass


def _gemini_api_key() -> str:
    ensure_dotenv_loaded()
    for name in _GEMINI_KEY_ENVS:
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return raw
    return ""


def _gemini_model_id() -> str:
    return (
        (os.environ.get("DANA_GEMINI_MODEL") or "").strip()
        or (os.environ.get("GEMINI_MODEL") or "").strip()
        or _DEFAULT_GEMINI_MODEL
    )


def _ask_gemini_text_native(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_output_tokens: int,
    response_mime_type: str | None,
) -> str:
    """Native Gemini ``generateContent`` call — used only by ``_complete_cloud``
    when ``cloud_provider_name()`` resolves to "gemini"/"google" (the
    default). Retries a 429/503 with capped exponential backoff, same as
    the module this was extracted from.
    """
    key = _gemini_api_key()
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    gen_cfg: dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    if response_mime_type:
        gen_cfg["responseMimeType"] = str(response_mime_type)
    payload: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
    if system_bits:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_bits)}]}

    throttle_retries = 0
    while True:
        resp = requests.post(url, params={"key": key}, json=payload, timeout=90)
        if resp.status_code in {429, 503}:
            if throttle_retries >= _GEMINI_THROTTLE_MAX_RETRIES:
                resp.raise_for_status()
            wait = min(_GEMINI_THROTTLE_MAX_WAIT_S, float(2**throttle_retries))
            print(f"[Gemini] Throttled — retrying in {wait:g}s...", flush=True)
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
        return out


def cloud_fallback_enabled() -> bool:
    ensure_dotenv_loaded()
    raw = (os.environ.get("DANA_ALLOW_CLOUD_FALLBACK") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def ollama_fallback_enabled() -> bool:
    """Whether ``ModelProvider.complete_with_tool_calls`` should retry
    against local Ollama when the primary CLOUD provider call raises —
    any exception (a 402 "Payment Required" from a rate-limited free
    OpenRouter tier is the incident this exists for, but a 429/5xx/network
    failure gets the same treatment, since the caller can't tell them
    apart from the plain ``RuntimeError`` ``dana.core.openai_tool_bridge``
    raises for any non-2xx response — see that module's own HTTPError
    handling).

    Deliberately a SEPARATE flag from ``cloud_fallback_enabled`` above,
    not a reuse of it — that one gates the OPPOSITE direction (a
    local-first plain-text ``complete()`` call falling through to cloud
    once local fails). Reusing it here would have silently inherited
    backwards semantics. Also deliberately default-ENABLED (opposite of
    ``cloud_fallback_enabled``'s default-off) — this is meant to be
    "seamless": an unavailable cloud provider with no fallback today just
    dead-ends the user's turn with a raw error, so the safer default is on,
    overridable for anyone (tests, CI, a deliberately cloud-only setup)
    who wants it off.

    Defaults to OFF specifically on ``dana.platform.factory.IS_HF_SPACE``
    (unless ``DANA_OLLAMA_FALLBACK`` is explicitly set): a live HF Space run
    hit an OpenRouter free-tier daily 429, then this fallback attempted
    local Ollama anyway and got ``[Errno 111] Connection refused`` — no HF
    Space container actually runs an Ollama daemon, so the attempt is pure
    wasted latency that only replaces one honest cloud error with a
    confusing "AND local Ollama fallback failed too" compound one. Still
    overridable (``DANA_OLLAMA_FALLBACK=1``) for a custom Space image that
    genuinely bundles Ollama.
    """
    ensure_dotenv_loaded()
    raw = (os.environ.get("DANA_OLLAMA_FALLBACK") or "").strip().lower()
    if raw:
        return raw not in {"0", "false", "no", "off"}
    from dana.platform.factory import IS_HF_SPACE

    return not IS_HF_SPACE


def force_local() -> bool:
    return (os.environ.get("DANA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def local_model_name() -> str:
    ensure_dotenv_loaded()
    return (
        (os.environ.get("DANA_LOCAL_MODEL") or "").strip()
        or (os.environ.get("OLLAMA_MODEL") or "").strip()
        or _DEFAULT_LOCAL_MODEL
    )


def local_vision_model_name() -> str:
    """The local Ollama model ``complete_vision``'s ``"ollama"`` branch
    resolves to — deliberately independent of ``local_model_name()``, whose
    ``DANA_LOCAL_MODEL`` is a text/tool-calling model with no multimodal
    support of its own. See ``_DEFAULT_LOCAL_VISION_MODEL``'s comment for
    why routing a vision call through that model 400s outright."""
    ensure_dotenv_loaded()
    return (
        (os.environ.get("DANA_LOCAL_VISION_MODEL") or "").strip()
        or (os.environ.get("OLLAMA_VISION_MODEL") or "").strip()
        or _DEFAULT_LOCAL_VISION_MODEL
    )


def cloud_provider_name() -> str:
    ensure_dotenv_loaded()
    return (
        (os.environ.get("DANA_CLOUD_PROVIDER") or "").strip().lower()
        or "gemini"
    )


def cloud_primary_enabled() -> bool:
    """Whether the ReAct loop's per-turn TOOL-CALLING hot path (dana.core.
    react_dispatch._call_llm_once) should route through a cloud OpenAI-
    compatible endpoint instead of the local Ollama daemon — the "shift the
    heavy lifting off local VRAM onto a free, larger cloud model" rescue-plan
    path. Distinct from ``cloud_fallback_enabled``: that one only kicks in
    AFTER a local call fails/rejects; this one skips local entirely for the
    tool-calling hot path, by default.
    """
    ensure_dotenv_loaded()
    return (os.environ.get("DANA_CLOUD_PRIMARY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_cloud_tool_provider() -> str:
    """Which cloud provider name ``tool_calling_provider()`` resolves to
    once cloud-primary routing is active for the OpenAI-tool-calling
    bridge — factored out so a caller latching a session onto cloud mid-
    conversation (see the Context Handoff check in
    ``dana.api.server._run_react_loop``) computes the EXACT SAME name a
    normal cloud-primary turn would use. Deliberately NOT
    ``cloud_provider_name()`` above: that one's own "gemini" default backs
    the plain-text-only local-vs-cloud complexity-fallback path, and
    Gemini's OpenAI-compat endpoint has a known thought_signature 400 bug
    mid-multi-turn tool-calling (see model docstrings elsewhere in this
    module) — exactly the wrong provider to silently hand a struggling
    long-running tool-calling session off to.

    Bug fix: this used to return ``DANA_CLOUD_PROVIDER`` verbatim whenever
    it was explicitly set, with no check against the OpenAI tool-calling
    bridge's own ``_NON_OPENAI_SCHEMA_PROVIDERS`` — the "avoid gemini"
    reasoning above only ever protected against ``cloud_provider_name()``'s
    bare, unset-env default, not an operator's own ``DANA_CLOUD_PROVIDER=
    gemini`` (set for some other, plain-text call path). Confirmed live
    (dana_runtime.log): the Context Handoff latched a session onto
    ``"gemini"``, and every subsequent turn's ``next_react_turn`` raised
    ``NotImplementedError`` before ever reaching the LLM — a permanent,
    silent dead end for that session, since ``session["active_provider"]``
    is never cleared once set. Falling back to ``"openrouter"`` here for
    any value this bridge genuinely can't serve keeps the actual
    configured preference for every OTHER call path untouched.
    """
    configured = (os.environ.get("DANA_CLOUD_PROVIDER") or "").strip().lower()
    if configured and configured not in _NON_OPENAI_SCHEMA_PROVIDERS:
        return configured
    return "openrouter"


def ollama_num_ctx() -> int:
    """Context-window size (tokens) requested from local Ollama's NATIVE
    ``/api/chat`` surface via its ``options.num_ctx`` — the OpenAI-compat
    ``/v1/chat/completions`` surface this bridge used exclusively before
    has no equivalent knob (``max_tokens`` only maps to output length,
    ``num_predict``), so a long local tool-calling chain silently
    truncated at the pulled model tag's own default (often 2048-4096) with
    no error surfaced anywhere. ``DANA_OLLAMA_NUM_CTX`` overrides.

    Default lowered from 32768 to 8192 (confirmed CUDA OOM on an RTX 2080
    running Qwen2.5-Coder:14B): Ollama/llama.cpp pre-allocates VRAM for the
    FULL requested ``num_ctx`` as a fixed-size KV cache at context creation
    time, regardless of how many tokens a given call actually uses — for
    this model's architecture that's roughly 192KiB/token of KV cache, so
    32768 alone commits ~6GB on top of the ~9GB the Q4 weights already
    need, well past an 8GB (and tight on an 11GB) card before a single real
    token is processed. 8192 covers this codebase's own per-turn budget
    (tool schemas capped at ``_TOOL_TOKEN_BUDGET`` in
    ``dana.core.react_dispatch``, trajectory pruned/compressed by
    ``dana.core.context_manager``) with headroom, while the Two-Layer
    Context Management handoff (``dana.api.server._run_react_loop``) still
    hands a session off to cloud well before it would ever need more.
    """
    raw = (os.environ.get("DANA_OLLAMA_NUM_CTX") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 8192


def unload_ollama_model(model_name: str, *, timeout: float = 5.0) -> bool:
    """Best-effort request asking local Ollama to immediately unload
    ``model_name`` from RAM/VRAM, rather than waiting out its own default
    5-minute ``keep_alive``. Ollama's documented signal for this is an
    ordinary generate request with an empty ``prompt`` and
    ``"keep_alive": 0`` (see Ollama's FAQ: "How do I keep a model loaded in
    memory or make it unload immediately?") — ``/api/generate`` rather than
    ``/api/chat`` since an empty prompt needs no ``messages`` shape at all,
    the smallest request that still carries ``keep_alive``.

    Same base-URL normalization ``dana.core.openai_tool_bridge.
    complete_ollama_native_with_tools`` already uses: accepts either the
    bare Ollama origin or the ``"/v1"``-suffixed form
    ``_resolve_openai_endpoint`` builds for the OpenAI-compat path, so a
    caller never needs a second, native-specific URL to track.

    Never raises: this is opportunistic cleanup a caller fires and forgets
    once a session (or the whole process) has genuinely gone idle, never a
    step any turn's own success depends on. Ollama not running, a network
    hiccup, or the model already unloaded all just mean nothing to reclaim
    right now — returns ``True``/``False`` only for logging/tests.
    """
    model_name = (model_name or "").strip()
    if not model_name:
        return False
    root = (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        requests.post(
            f"{root}/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=timeout,
        )
        return True
    except requests.RequestException:
        return False


def ollama_num_gpu() -> int | None:
    """Number of model layers to offload to GPU, requested from local
    Ollama's NATIVE ``/api/chat`` surface via its ``options.num_gpu`` — the
    same "no OpenAI-wire equivalent" gap ``ollama_num_ctx`` documents.
    Unset (``None``, the default) omits ``num_gpu`` from ``options``
    entirely, leaving Ollama's own automatic VRAM-fit heuristic in charge,
    exactly as before this existed. ``DANA_OLLAMA_NUM_GPU``, when set to a
    valid integer, caps the offload instead — lower than Ollama's own guess
    trades inference speed for headroom on a GPU shared with something else
    (the CAD viewport, another process), rather than Ollama silently
    filling VRAM and starving it.
    """
    raw = (os.environ.get("DANA_OLLAMA_NUM_GPU") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return None


def estimate_message_tokens(messages: list[dict[str, Any]], *, tool_schema_tokens: int = 0) -> int:
    """Cheap, dependency-free token estimate for a ``messages`` history —
    a character-count heuristic (~3.5 chars/token, a reasonable blend of
    English prose and JSON tool-call payloads), not a real tokenizer.
    Good enough for "are we approaching this session's num_ctx budget",
    not for billing — see ``dana.tools.schema_minify.estimate_tokens`` for
    the word-count-based sibling heuristic used for schema-size
    comparisons, a different unit for a different question.

    ``tool_schema_tokens`` (default 0, for callers that genuinely have none
    to report) adds a flat token count on top of ``messages`` for the
    ``tools=`` schema payload sent alongside it in the SAME request —
    without this, a caller measuring only ``messages`` silently
    undercounts the real prompt Ollama/the provider actually processes by
    up to ``dana.core.react_dispatch._TOOL_TOKEN_BUDGET`` (2000) tokens,
    since that schema is serialized into the same context window but was
    never part of this ``messages`` list to begin with. Pass the tool
    schema's own hard token ceiling (rather than re-deriving the exact
    narrowed schema, which would mean re-running the same
    embedding-ranked narrowing this estimate is trying to avoid paying for
    twice per turn) for a conservative, never-under, worst-case bound —
    the actual schema is already capped at exactly that ceiling by
    ``_cap_schemas_by_token_budget``, so this can never overstate the true
    figure.
    """
    return int(len(json.dumps(messages, default=str)) / 3.5) + max(0, tool_schema_tokens)


def tool_calling_provider() -> str:
    """Which ``ModelProvider.complete_with_tool_calls`` provider the ReAct
    loop's hot path should target this turn — the single source of truth
    ``dana.core.react_dispatch._call_llm_once`` defers to instead of a
    hardcoded ``"ollama"``.

    ``"ollama"`` (local, free-per-request but VRAM/context limited) unless
    ``cloud_primary_enabled()`` — then ``DANA_CLOUD_PROVIDER`` if explicitly
    set, else ``"openrouter"`` by default: a direct call to OpenRouter,
    whose own server-side ``models`` array (``DANA_OPENROUTER_MODEL`` as a
    comma-separated list — see ``_resolve_openai_endpoint``'s
    ``"openrouter"`` branch) already retries a 429/5xx against the next
    model upstream in milliseconds, with no local gateway process needed to
    hold provider keys or cascade across providers itself.

    ``"gemini_openai"`` (Gemini's OpenAI-compatible endpoint, 1,000,000 TPM
    versus Groq's free-tier 8,000) briefly WAS the default here, to kill
    that 8k TPM ceiling. Reverted: Google's OpenAI-compat endpoint 400s
    mid-multi-turn-ReAct-loop, requiring a proprietary ``thought_signature``
    field in the replayed ``tool_calls`` history that this bridge's plain
    OpenAI wire format has no way to carry. Reverting is also no longer a
    real regression now that ``search_codebase`` (context compression —
    dana.plugins.coder_plugin) means the ReAct loop doesn't actually need
    Gemini's 1M-token ceiling to stay under Groq's 8,000 TPM in practice —
    Aider still calls Gemini NATIVELY (its own API, not this bridge) for
    the actual heavy file-editing work, so this is a genuine hybrid
    architecture, not a full retreat. ``_resolve_openai_endpoint``'s
    ``gemini_openai`` branch (below) is left fully intact and still
    reachable via ``DANA_CLOUD_PROVIDER=gemini_openai`` for whenever
    Google fixes that endpoint or this bridge learns to carry
    ``thought_signature`` — just no longer the default.
    """
    if not cloud_primary_enabled():
        return "ollama"
    return resolve_cloud_tool_provider()


def _log_ttft(
    model: str, ttft_ms: float | None, *, tools_schema_bytes: int | None = None
) -> None:
    """Best-effort perf log for a streamed ``complete_openai_with_tools``
    call's real "time to first token" — the same ``dana_performance.log``
    signal ``dana.core.agent_loop.ask_ollama_messages`` already records for
    the native-Ollama path, now also covering the OpenAI-tool-calling bridge
    (the ReAct loop's actual hot path), which previously logged nothing
    until the entire blocking request finished.

    ``tools_schema_bytes`` (P1 of the local-agent rescue plan) rides on the
    SAME log line as ``ttft_ms`` — not a separate metric — specifically so
    the two are trivially correlatable in ``dana_performance.log`` without
    joining across records: as a session's ``agent_loaded_capabilities``
    grows the tool schema, this is the number that should visibly grow
    alongside a climbing TTFT, and shrink back down once P1's per-session
    capability decay (dana.api.server._effective_capabilities) drops an
    unused domain back out.
    """
    if ttft_ms is None:
        return
    try:
        from dana.perf import log_perf

        log_perf("llm_ttft", ttft_ms, model=model, tools_schema_bytes=tools_schema_bytes)
    except Exception:  # noqa: BLE001 — perf logging must never break a real completion
        pass


def complexity_reject_marker() -> str:
    return _COMPLEXITY_REJECT


def is_complexity_reject(text: str) -> bool:
    s = str(text or "").strip()
    if not s.upper().startswith("REJECT:"):
        return False
    low = s.lower()
    return (
        "too complex for local" in low
        or "task too complex" in low
        or s.startswith(_COMPLEXITY_REJECT)
    )


def _sanitize_header_value(value: str, *, fallback: str) -> str:
    """Strip a header value down to what HTTP can actually transmit.

    Real, observed crash: `OPENROUTER_APP_TITLE=Dānā CAD Agent` raised
    ``UnicodeEncodeError: 'latin-1' codec can't encode character '\\u0101'``
    — not at the point this dict gets built, but deep inside
    ``http.client``/``urllib`` when the request actually goes out, since
    header values are transmitted as latin-1 regardless of what a Python
    ``str`` can hold. An env var feeding straight into a header value (this
    provider's ``OPENROUTER_SITE_URL``/``OPENROUTER_APP_TITLE``) is user
    input from this module's point of view, so it gets sanitized here
    rather than trusted. Stripping non-ASCII bytes is lossy but never
    crashes; falls back to ``fallback`` if that stripping empties the
    string out entirely (e.g. a title that was ALL non-ASCII).
    """
    cleaned = value.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or fallback


class ModelProvider:
    """Unified chat completion for Spec Compiler / Meta-Broker planning."""

    def __init__(
        self,
        *,
        local_model: str | None = None,
        prefer: ProviderKind = "auto",
        api_keys: dict[str, str] | None = None,
    ) -> None:
        self.local_model = (local_model or local_model_name()).strip()
        self.prefer = prefer
        self.last_provider: str = "none"
        self.last_error: str = ""
        # BYOK — per-session keys (e.g. from the frontend's SecretsMenu,
        # threaded down via dana.api.server's session dict), keyed the same
        # way the frontend's ServiceId already is: "openai", "anthropic".
        # Session key wins; _resolve_openai_endpoint/_complete_openai_compatible
        # fall back to the environment variable only when a provider has no
        # entry here. Never logged — this dict is never passed to print/log.
        self._api_keys: dict[str, str] = dict(api_keys) if api_keys else {}

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int = 512,
        temperature: float = 0.1,
        allow_cloud: bool | None = None,
        response_mime_type: str | None = "text/plain",
    ) -> str:
        """Run a chat completion; optionally fall back to cloud."""
        allow = cloud_fallback_enabled() if allow_cloud is None else bool(allow_cloud)
        if force_local():
            allow = False

        use_cloud_first = self.prefer == "cloud" and allow
        if use_cloud_first:
            try:
                return self._complete_cloud(
                    messages,
                    num_predict=num_predict,
                    temperature=temperature,
                    response_mime_type=response_mime_type,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"cloud: {exc}"
                # Fall through to local.

        try:
            text = self._complete_local(
                messages,
                num_predict=num_predict,
                temperature=temperature,
            )
            self.last_provider = "local"
            return text
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"local: {exc}"
            if not allow:
                raise
            return self._complete_cloud(
                messages,
                num_predict=num_predict,
                temperature=temperature,
                response_mime_type=response_mime_type,
            )

    def complete_with_complexity_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Local first; if response is a complexity REJECT and cloud fallback
        is enabled, re-issue the request on the cloud provider.
        """
        local_out = self.complete(
            messages,
            num_predict=num_predict,
            temperature=temperature,
            allow_cloud=False,
        )
        if not is_complexity_reject(local_out):
            return local_out
        if not cloud_fallback_enabled() or force_local():
            return local_out
        print(
            "[ModelProvider] local complexity REJECT → cloud fallback",
            flush=True,
        )
        cloud_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "The local model rejected this as too complex. "
                    "Produce a precise /broker multi-epic specification now. "
                    "Do not REJECT unless truly impossible with stdlib + MCP tools."
                ),
            }
        ]
        try:
            out = self._complete_cloud(
                cloud_messages,
                num_predict=max(num_predict, 768),
                temperature=temperature,
                response_mime_type="text/plain",
            )
            return out
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"cloud_fallback: {exc}"
            return local_out

    def _complete_local(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
    ) -> str:
        """Plain-text local completion via Ollama's own OpenAI-compatible
        ``/v1/chat/completions`` surface — the SAME ``complete_openai_with_tools``
        bridge the tool-calling hot path already uses (just with no
        ``tools=``), so this has no dependency on the legacy
        ``dana.core.agent_loop`` stack's native ``/api/chat`` caller.
        """
        key, base, _, _, _ = self._resolve_openai_endpoint("ollama")
        with llm_lock:
            raw = complete_openai_with_tools(
                messages,
                api_key=key,
                base_url=base,
                model=self.local_model,
                num_predict=num_predict,
                temperature=temperature,
            )
        self.last_provider = "local"
        return str(raw.get("content") or "").strip()

    def _complete_cloud(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        response_mime_type: str | None,
    ) -> str:
        provider = cloud_provider_name()
        if provider in {"gemini", "google"}:
            text = _ask_gemini_text_native(
                messages,
                temperature=temperature,
                max_output_tokens=int(num_predict),
                response_mime_type=response_mime_type,
            )
            self.last_provider = "cloud:gemini"
            return str(text or "").strip()

        # OpenAI-compatible path (OpenAI / Groq / Gemini's OpenAI-compat endpoint).
        return self._complete_openai_compatible(
            messages,
            num_predict=num_predict,
            temperature=temperature,
            provider=provider,
        )

    def _resolve_openai_endpoint(
        self, provider: str
    ) -> tuple[str, str, str, dict[str, str], list[str]]:
        """Return ``(api_key, base_url, model, extra_headers, fallback_models)``
        for an OpenAI-wire-compatible provider.

        Shared by the plain-text ``_complete_openai_compatible`` path and the
        tool-calling / vision bridge below — one place that knows how each
        provider's env vars map to a key/base/model/headers/fallbacks tuple.
        ``"ollama"`` targets the local Ollama daemon's own OpenAI-compatible
        ``/v1/chat/completions`` surface (distinct from the native
        ``/api/chat`` path used by ``_complete_local``), which needs no real
        API key. ``extra_headers`` is ``{}`` for every provider except
        ``"openrouter"`` (its recommended, not required, HTTP-Referer/
        X-Title attribution headers) — kept out of ``openai_tool_bridge.py``
        on purpose, since that module has no per-provider knowledge at all.
        ``fallback_models`` is ``[]`` for every provider except
        ``"openrouter"``, which accepts a comma-separated ``DANA_OPENROUTER_MODEL``
        list and forwards everything after the first entry as OpenRouter's
        native server-side ``models`` fallback/cascade array, so a 429 on the
        primary model retries the next one upstream in milliseconds instead
        of round-tripping back to this process.
        """
        ensure_dotenv_loaded()
        if provider == "openrouter":
            # OPENROUTER_API_KEY first; LLM_API_KEY as a generic fallback so
            # a Space owner who already uses that generic naming convention
            # doesn't need a second, provider-specific secret.
            key = (
                self._api_keys.get("openrouter")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("LLM_API_KEY")
                or ""
            ).strip()
            base = (
                (os.environ.get("OPENROUTER_API_BASE") or "").strip()
                or "https://openrouter.ai/api/v1"
            )
            raw_models = (
                (os.environ.get("DANA_OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL") or "").strip()
            )
            model_list = [m.strip() for m in raw_models.split(",") if m.strip()]
            # meta-llama/llama-3.3-70b-instruct:free confirmed DEAD (live 404 from
            # OpenRouter itself: "This model is unavailable for free. The paid
            # version is available now -- use this slug instead: meta-llama/
            # llama-3.3-70b-instruct") -- verified-live, tool-calling-capable
            # replacement; see DANA_OPENROUTER_MODEL's own .env comment for the
            # verification method (openrouter.ai/api/v1/models, not a blog post).
            model = model_list[0] if model_list else "nvidia/nemotron-3.5-lightning:free"
            fallback_models = model_list[1:]
            if not key:
                raise RuntimeError("No API key configured for cloud provider='openrouter'")
            headers = {
                # OpenRouter's own docs ask for these two for attribution/
                # rankings on their dashboard — the request works without
                # them, this just identifies the app instead of showing up
                # as anonymous. Both overridable; sensible defaults either
                # way (HF_SPACE_URL and SPACE_ID are HF's own auto-set env
                # vars, so this needs no Dana-specific config to be correct
                # out of the box in a Space). Sanitized (see
                # _sanitize_header_value) since these two specifically come
                # straight from env vars — a non-ASCII value crashes deep in
                # http.client at request-send time, not here, which is
                # exactly what happened with an early "Dānā"-branded title.
                "HTTP-Referer": _sanitize_header_value(
                    (os.environ.get("OPENROUTER_SITE_URL") or "").strip()
                    or (os.environ.get("HF_SPACE_URL") or "").strip()
                    or "https://github.com/",
                    fallback="https://github.com/",
                ),
                "X-Title": _sanitize_header_value(
                    (os.environ.get("OPENROUTER_APP_TITLE") or "").strip() or "Dana CAD Agent",
                    fallback="Dana CAD Agent",
                ),
            }
            return key, base, model, headers, fallback_models
        if provider == "gemini_openai":
            # Google's OpenAI-compatible endpoint — distinct from the
            # "gemini"/"google" provider names in _NON_OPENAI_SCHEMA_PROVIDERS,
            # which target Gemini's own native API (dana.graph.cloud_planner.
            # ask_gemini_text, plain-text only, no tool-calling/vision). This
            # branch is what actually lets Gemini serve the OpenAI-wire tool-
            # calling/vision bridge below — a 1,000,000 TPM ceiling versus
            # Groq's free-tier 8,000 TPM (see tool_calling_provider's docstring).
            key = (self._api_keys.get("gemini") or os.environ.get("GEMINI_API_KEY") or "").strip()
            base = (
                (os.environ.get("GEMINI_API_BASE") or "").strip()
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model = (
                (os.environ.get("DANA_GEMINI_MODEL") or "").strip()
                or "gemini-3.6-flash"
            )
        elif provider == "groq":
            # CLOUD_API_KEY is a generic fallback for whichever cloud
            # provider DANA_CLOUD_PRIMARY/DANA_CLOUD_PROVIDER selects —
            # GROQ_API_KEY wins if both happen to be set.
            key = (os.environ.get("GROQ_API_KEY") or os.environ.get("CLOUD_API_KEY") or "").strip()
            base = (
                (os.environ.get("GROQ_API_BASE") or "").strip()
                or "https://api.groq.com/openai/v1"
            )
            model = (
                (os.environ.get("DANA_GROQ_MODEL") or "").strip()
                or "llama-3.3-70b-versatile"
            )
        elif provider == "ollama":
            key = (os.environ.get("OLLAMA_API_KEY") or "").strip() or "ollama"
            base = (
                (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
                + "/v1"
            )
            model = (os.environ.get("DANA_OPENAI_TOOLS_MODEL") or "").strip() or self.local_model
        elif provider == "deepseek":
            # DeepSeek's own API — OpenAI-wire compatible, so this bridge
            # needs no new transport, just a new key/base pair. Not free
            # (unlike OpenRouter's :free tier or Groq's free tier) — see
            # dana.core.llm_router's cost_per_1m fields for why a routing
            # config would rank this behind the free entries by default.
            key = (self._api_keys.get("deepseek") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
            base = (
                (os.environ.get("DEEPSEEK_API_BASE") or "").strip()
                or "https://api.deepseek.com/v1"
            )
            model = (
                (os.environ.get("DANA_DEEPSEEK_MODEL") or "").strip()
                or "deepseek-chat"
            )
        else:
            key = (self._api_keys.get("openai") or os.environ.get("OPENAI_API_KEY") or "").strip()
            base = (
                (os.environ.get("OPENAI_API_BASE") or "").strip()
                or "https://api.openai.com/v1"
            )
            model = (
                (os.environ.get("DANA_OPENAI_MODEL") or "").strip()
                or "gpt-4o-mini"
            )
        if not key:
            raise RuntimeError(f"No API key configured for cloud provider={provider}")
        return key, base, model, {}, []

    def _complete_openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        provider: str,
    ) -> str:
        ensure_dotenv_loaded()
        if provider == "anthropic":
            # Prefer Anthropic Messages API via env-compatible OpenAI proxy if set;
            # otherwise use raw Anthropic endpoint.
            key = (self._api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not configured")
            return self._complete_anthropic(
                messages,
                num_predict=num_predict,
                temperature=temperature,
                api_key=key,
            )

        key, base, model, extra_headers, fallback_models = self._resolve_openai_endpoint(provider)
        # Serializes with every other LOCAL Ollama generation in this
        # process (dana.system_health.llm_lock) — running two generations
        # concurrently against the same local daemon is what doubles VRAM
        # usage and fragments it (see llm_lock's own docstring). A genuine
        # cloud provider (groq/openai/anthropic) consumes no local VRAM at
        # all, so it deliberately bypasses this lock (nullcontext) — cloud
        # calls run fully in parallel with each other AND with a concurrent
        # local Ollama call, rather than being serialized for no reason.
        with llm_lock if provider == "ollama" else contextlib.nullcontext():
            raw = complete_openai_with_tools(
                messages,
                api_key=key,
                base_url=base,
                model=model,
                num_predict=num_predict,
                temperature=temperature,
                extra_headers=extra_headers,
                fallback_models=fallback_models,
            )
        _log_ttft(model, raw.get("ttft_ms"))
        self.last_provider = f"cloud:{provider}"
        return str(raw.get("content") or "").strip()

    def complete_with_tool_calls(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        provider: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        num_predict: int = LLM_MAX_OUTPUT_TOKENS,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """OpenAI-schema tool-calling turn against a cloud or local-Ollama endpoint.

        Returns ``{"content": str, "tool_calls": list[ToolCall], "provider": str,
        "model": str, "usage": {"prompt_tokens": int, "completion_tokens": int},
        "cost_usd": float | None, "finish_reason": str | None}`` — ``tool_calls``
        is already Dana's native IR
        (see ``dana.tools.schema.openai_tool_calls_to_ir``), so callers can hand
        results straight to the existing broker/dispatch path
        (``dana.core.agent_loop.execute_tool_call``) with no OpenAI-shape
        parsing of their own. Raises ``NotImplementedError`` for providers
        that don't speak the OpenAI tools schema (Gemini, Anthropic).

        Cost Tracking: ``usage`` is read off whichever endpoint actually
        answered (primary or the Ollama fallback) — ``{0, 0}`` if the
        endpoint never sent a usage chunk (see ``openai_tool_bridge``'s
        ``stream_options.include_usage``). ``cost_usd`` is ``None`` whenever
        ``model`` isn't in ``dana.core.pricing``'s table (every local Ollama
        model, by construction — OpenRouter never priced them), so a caller
        must treat ``None`` as "unknown", never as free.

        Automatic Ollama Fallback: if the primary CLOUD call raises (a 402
        "Payment Required" from a rate-limited free OpenRouter tier is the
        incident this was built for, but any exception gets the same
        treatment — see ``ollama_fallback_enabled``'s own docstring for
        why this can't distinguish 402/429/5xx from each other), this
        retries the SAME messages/tools/tool_choice against local Ollama
        before giving up. Skipped when the primary attempt was already
        Ollama itself (nothing to fall back to) or when
        ``ollama_fallback_enabled()`` is off. If the fallback attempt ALSO
        raises, a single ``RuntimeError`` propagates naming BOTH failures
        (cloud + local), so whoever's reading logs doesn't have to go
        hunting for the original cloud error separately.
        """
        # Dynamic LLM Router (dana.core.llm_router) — entirely opt-in via
        # routing_config.yaml's presence. ``provider`` being explicitly
        # forced by the caller always wins (a caller that names a specific
        # provider has already made its own routing decision); otherwise, a
        # valid fleet config takes over provider/model selection AND
        # multi-hop fallback for this turn instead of the single hardcoded
        # cloud->Ollama hop below. No routing_config.yaml (the default,
        # out-of-the-box state) -> resolve_chain returns None -> falls
        # through to the unchanged legacy path beneath this block.
        if provider is None:
            # Lazy import — dana.core.__init__ imports FROM this module, so
            # a top-level `from dana.core import llm_router` here would race
            # dana.core's own partial initialization; deferring the import
            # to call time (same convention used throughout this codebase
            # for exactly this reason) sidesteps it entirely.
            from dana.core import llm_router

            chain = llm_router.resolve_chain(messages, tools)
            if chain:
                return self._complete_with_tool_calls_via_router(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    num_predict=num_predict,
                    temperature=temperature,
                    chain=chain,
                )

        resolved_provider = (provider or cloud_provider_name()).strip().lower()
        if resolved_provider in _NON_OPENAI_SCHEMA_PROVIDERS:
            raise NotImplementedError(
                f"OpenAI tool-calling bridge does not support provider={resolved_provider!r} "
                "(uses a non-OpenAI tool schema)"
            )
        key, base, model, extra_headers, fallback_models = self._resolve_openai_endpoint(resolved_provider)
        # See the matching comment in _complete_openai_compatible above —
        # this is the ReAct loop's actual per-turn call site, so it's the
        # one that matters most both for VRAM-fragmentation-from-concurrent-
        # local-generations (when DANA_CLOUD_PRIMARY is off) AND for letting
        # cloud-routed turns (DANA_CLOUD_PRIMARY on — see
        # dana.core.model_provider.tool_calling_provider) run fully
        # unserialized, since a cloud call has no local VRAM to contend for.
        try:
            with llm_lock if resolved_provider == "ollama" else contextlib.nullcontext():
                if resolved_provider == "ollama":
                    # Native /api/chat, not the OpenAI-compat surface — see
                    # complete_ollama_native_with_tools's own docstring for
                    # why (num_ctx has no OpenAI-wire equivalent).
                    raw = complete_ollama_native_with_tools(
                        messages,
                        base_url=base,
                        model=model,
                        tools=tools,
                        num_predict=num_predict,
                        num_ctx=ollama_num_ctx(),
                        num_gpu=ollama_num_gpu(),
                        temperature=temperature,
                    )
                else:
                    raw = complete_openai_with_tools(
                        messages,
                        api_key=key,
                        base_url=base,
                        model=model,
                        tools=tools,
                        tool_choice=tool_choice,
                        num_predict=num_predict,
                        temperature=temperature,
                        extra_headers=extra_headers,
                        fallback_models=fallback_models,
                    )
            effective_provider = resolved_provider
        except Exception as exc:  # noqa: BLE001 — any cloud failure falls back to local Ollama below
            if resolved_provider == "ollama" or not ollama_fallback_enabled():
                raise
            print(
                "[Ollama Fallback] Cloud provider failed, routing request to local Ollama...",
                file=sys.stderr,
                flush=True,
            )
            print(f"[Ollama Fallback] original {resolved_provider!r} error: {exc}", file=sys.stderr, flush=True)
            fb_key, fb_base, fb_model, fb_headers, fb_fallback_models = self._resolve_openai_endpoint("ollama")
            # OLLAMA_FALLBACK_MODEL, when set, wins outright; otherwise this
            # reuses whatever local model the user already has configured
            # for the existing local-first path (_resolve_openai_endpoint's
            # own DANA_OPENAI_TOOLS_MODEL/self.local_model resolution) —
            # NOT a hardcoded "llama3" default, which would just as likely
            # 404 ("model not found") on a machine that never pulled it,
            # defeating the entire point of a "seamless" fallback.
            fb_model = (os.environ.get("OLLAMA_FALLBACK_MODEL") or "").strip() or fb_model
            try:
                with llm_lock:
                    raw = complete_ollama_native_with_tools(
                        messages,
                        base_url=fb_base,
                        model=fb_model,
                        tools=tools,
                        num_predict=num_predict,
                        num_ctx=ollama_num_ctx(),
                        num_gpu=ollama_num_gpu(),
                        temperature=temperature,
                    )
            except Exception as fallback_exc:  # noqa: BLE001 — see docstring: this replaces exc, deliberately
                raise RuntimeError(
                    f"Cloud provider {resolved_provider!r} failed ({exc}), AND local Ollama fallback "
                    f"failed too ({fallback_exc}). Please ensure Ollama is running and the model "
                    f"({fb_model!r}) is pulled."
                ) from fallback_exc
            model = fb_model
            effective_provider = "ollama"
        # P1 metric — logged on the SAME line as ttft_ms (see _log_ttft) so
        # the two are directly correlatable turn over turn.
        _log_ttft(model, raw.get("ttft_ms"), tools_schema_bytes=len(json.dumps(tools)) if tools else 0)
        self.last_provider = (
            f"cloud:{resolved_provider}"
            if effective_provider == resolved_provider
            else f"ollama-fallback (was {resolved_provider})"
        )
        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost_usd = (
            estimate_cost_usd(model, prompt_tokens, completion_tokens)
            if (prompt_tokens or completion_tokens)
            else None
        )
        return {
            "content": str(raw.get("content") or "").strip(),
            "tool_calls": openai_tool_calls_to_ir(
                raw.get("tool_calls"), raw_text=str(raw.get("content") or "")
            ),
            "provider": self.last_provider,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "cost_usd": cost_usd,
            # Token-Truncation Trap: "length"/"MAX_TOKENS" (provider-dependent
            # wording) means the completion was cut off by num_predict, not
            # finished naturally — see openai_tool_bridge's own capture of
            # this field. None when the endpoint didn't report one at all.
            "finish_reason": raw.get("finish_reason"),
        }

    @staticmethod
    def _looks_like_attempted_tool_call(content: str | None) -> bool:
        """True when ``content`` (a completion's plain-text ``message.content``,
        already run through ``complete_openai_with_tools``'s own
        ``_fallback_tool_calls_from_content`` recovery with no luck) still
        looks like the model was TRYING to emit a structured call rather
        than giving a genuine conversational answer, rather than ordinary
        prose. Deliberately loose/cheap (no full JSON parse — that already
        failed, or this wouldn't be reached) so a real final answer that
        merely happens to mention a brace in passing is the only realistic
        false positive, and a false positive here just means one extra
        fleet-entry hop instead of accepting a slower model's genuine
        answer, never a crash or a dropped turn.

        Two explicit shapes, confirmed live against a real Ollama
        production failure (a raw ``{"name": "create_plan", "arguments":
        ...}`` string landing in ``message.content`` instead of populating
        ``message.tool_calls``):
          - starts with ``{`` and contains both ``"name"`` and
            ``"arguments"`` — the full OpenAI-shape call envelope, the
            exact shape that incident produced.
          - starts with ``[`` and contains ``{`` — a list of call-shaped
            objects.
        Also still catches the plainer case a small model is at least as
        likely to produce for a tool like ``create_plan`` — the bare
        ``arguments`` object with NO ``{"name":..., "arguments":...}``
        envelope at all (e.g. ``{"objective": ..., "tasks": [...]}``) —
        via the same ``startswith("{")`` fallback this already had; the two
        explicit shapes above are checked FIRST purely so a true positive
        is traceable to a named, specific shape in the log line below
        rather than a generic "looked bracy" catch-all.
        """
        text = (content or "").strip()
        if not text:
            return False
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text[:4].lower() == "json":
                text = text[4:].strip()
        if text.startswith("{") and '"name"' in text and '"arguments"' in text:
            return True
        if text.startswith("[") and "{" in text:
            return True
        return text.startswith("{")

    @staticmethod
    def _tools_include(tools: list[dict[str, Any]] | None, tool_id: str) -> bool:
        """True when ``tool_id`` is one of the function names in ``tools``
        (the OpenAI-shape schema this turn actually offered the model —
        see ``dana.tools.schema.to_openai_function_schema``'s own output
        shape)."""
        for tool in tools or []:
            name = (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None
            if name == tool_id:
                return True
        return False

    def _complete_with_tool_calls_via_router(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
        num_predict: int,
        temperature: float,
        chain: "list[Any]",
    ) -> dict[str, Any]:
        """Walk ``chain`` (dana.core.llm_router.resolve_chain's output, a
        list of ``FleetEntry``) in order, advancing to the next entry on ANY
        exception — same "can't distinguish 402/429/5xx from a real outage"
        reasoning as ``complete_with_tool_calls``'s own hardcoded Ollama
        fallback (see that method's docstring) — until one succeeds or the
        chain is exhausted. Returns the same result shape as
        ``complete_with_tool_calls`` so callers can't tell which path
        answered.
        """
        from dana.core import llm_router

        errors: list[str] = []
        for entry in chain:
            try:
                fallback_key, base, _, extra_headers, fallback_models = self._resolve_openai_endpoint(
                    entry.provider
                )
            except Exception as exc:  # noqa: BLE001 — this entry has no usable key/base; try the next one
                errors.append(f"{entry.id} ({entry.provider}): {exc}")
                continue
            key = llm_router.api_key_for(entry, fallback_key=fallback_key)
            if not key:
                errors.append(f"{entry.id} ({entry.provider}): no API key configured")
                continue
            try:
                with llm_lock if entry.provider == "ollama" else contextlib.nullcontext():
                    raw = complete_openai_with_tools(
                        messages,
                        api_key=key,
                        base_url=base,
                        model=entry.model,
                        tools=tools,
                        tool_choice=tool_choice,
                        num_predict=num_predict,
                        temperature=temperature,
                        extra_headers=extra_headers,
                        fallback_models=fallback_models,
                    )
            except Exception as exc:  # noqa: BLE001 — advance to the next chain entry
                errors.append(f"{entry.id} ({entry.provider}/{entry.model}): {exc}")
                print(
                    f"[LLM Router] {entry.id} failed, advancing to next fleet entry: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                llm_router.report_fleet_entry_failure(entry, exc)
                continue

            # create_plan Schema-Adherence Gate: geometry tools are excluded
            # from this chain up front (llm_router._turn_needs_geometry_
            # precision) whenever they're offered, but create_plan can't be
            # — it's dana.core.react_dispatch._CORE_TOOL_IDS, offered on
            # literally every turn, so pre-excluding Ollama on its presence
            # would exclude Ollama from every turn ever. Instead, react to
            # the SAME failure mode reactively: complete_openai_with_tools
            # already tries to recover a tool call an Ollama model emitted
            # as plain JSON text in message.content instead of populating
            # message.tool_calls (_fallback_tool_calls_from_content) — if
            # THAT recovery also came up empty, and this turn offered
            # create_plan, and the content still looks like an attempted
            # structured call (JSON-ish, not ordinary prose) rather than a
            # genuine conversational final answer, treat this entry as
            # failed and advance — confirmed live (dana_runtime.log) that a
            # 7B local model can fail create_plan's own nested tasks[i].
            # expected_tools schema exactly this way.
            if (
                entry.provider == "ollama"
                and not raw.get("tool_calls")
                and self._tools_include(tools, "create_plan")
                and self._looks_like_attempted_tool_call(raw.get("content"))
            ):
                reason = "returned an unrecoverable raw-text payload instead of a structured create_plan call"
                errors.append(f"{entry.id} ({entry.provider}/{entry.model}): {reason}")
                print(
                    f"[LLM Router] {entry.id} {reason}, advancing to next fleet entry.",
                    file=sys.stderr,
                    flush=True,
                )
                # Explicit interception proof, into dana_runtime.log (NOT a
                # bare `logging.getLogger(...).warning(...)` — this module
                # never configures a handler for that, so a call like that
                # goes nowhere and would look like a silent no-op the next
                # time this exact failure needs diagnosing). ERROR is one of
                # dana.core.telemetry's own seven fixed INFO-tier event
                # kinds (see that module's docstring) — same stage=/detail=
                # shape as its own EXISTING 'empty_final_turn' ERROR event,
                # so this reads as one more instance of an already-
                # established convention, not a new ad hoc log format.
                from dana.core import telemetry

                telemetry.log_error(
                    stage="ollama_raw_json_tool_call",
                    tool_id="create_plan",
                    fleet_entry=entry.id,
                    model=entry.model,
                    detail="Caught raw JSON from Ollama, escalating to next provider",
                )
                llm_router.report_fleet_entry_failure(entry, RuntimeError(reason))
                continue

            _log_ttft(entry.model, raw.get("ttft_ms"), tools_schema_bytes=len(json.dumps(tools)) if tools else 0)
            self.last_provider = f"router:{entry.id}"
            usage = raw.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            cost_usd = (
                (prompt_tokens / 1_000_000) * entry.cost_per_1m_prompt
                + (completion_tokens / 1_000_000) * entry.cost_per_1m_completion
                if (prompt_tokens or completion_tokens)
                else None
            )
            return {
                "content": str(raw.get("content") or "").strip(),
                "tool_calls": openai_tool_calls_to_ir(
                    raw.get("tool_calls"), raw_text=str(raw.get("content") or "")
                ),
                "provider": self.last_provider,
                "model": entry.model,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                "cost_usd": cost_usd,
                "finish_reason": raw.get("finish_reason"),
            }

        raise RuntimeError(
            f"LLM Router: every fleet entry failed ({len(chain)} tried): " + "; ".join(errors)
        )

    def complete_vision(
        self,
        prompt: str,
        image_b64: str | list[str],
        *,
        mime_type: str | list[str] = "image/png",
        provider: str | None = None,
        num_predict: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Describe/analyze one or more images via an OpenAI-vision-compatible
        model. ``image_b64`` is either a single base64 string or a list of
        them (e.g. multiple orthographic views of the same part); ``mime_type``
        matches it 1:1 when it's a list, or applies to every image when it's
        a single string.

        ``provider="ollama"`` (the default when cloud fallback is off) hits
        the local Ollama VLM (e.g. Qwen2.5-VL) over its OpenAI-compatible
        surface at zero cost/egress; any other resolved provider goes to the
        matching cloud OpenAI-wire endpoint (GPT-4o-class on OpenAI/Groq).
        Raises ``NotImplementedError`` for Gemini/Anthropic, whose image
        payload shapes are not OpenAI-compatible.
        """
        resolved_provider = (provider or cloud_provider_name()).strip().lower()
        if resolved_provider in _NON_OPENAI_SCHEMA_PROVIDERS:
            raise NotImplementedError(
                f"complete_vision does not support provider={resolved_provider!r} "
                "(uses a non-OpenAI image payload schema)"
            )
        key, base, model, extra_headers, fallback_models = self._resolve_openai_endpoint(resolved_provider)
        if resolved_provider == "ollama":
            # _resolve_openai_endpoint's "ollama" branch resolves `model` to
            # DANA_OPENAI_TOOLS_MODEL/self.local_model — a text/tool-calling
            # model that Ollama's own OpenAI-compat surface rejects outright
            # for a multimodal request. Override with the dedicated vision
            # model instead; key/base/headers/fallback_models are unaffected.
            model = local_vision_model_name()
        images_b64 = image_b64 if isinstance(image_b64, list) else [image_b64]
        mime_types = mime_type if isinstance(mime_type, list) else [mime_type] * len(images_b64)
        if len(mime_types) != len(images_b64):
            raise ValueError("image_b64 and mime_type lists must be the same length")
        messages = build_multimodal_messages(prompt, images=list(zip(images_b64, mime_types)))
        with llm_lock if resolved_provider == "ollama" else contextlib.nullcontext():
            raw = complete_openai_with_tools(
                messages,
                api_key=key,
                base_url=base,
                model=model,
                num_predict=num_predict,
                temperature=temperature,
                extra_headers=extra_headers,
                fallback_models=fallback_models,
            )
        _log_ttft(model, raw.get("ttft_ms"))
        self.last_provider = f"cloud:{resolved_provider}"
        return str(raw.get("content") or "").strip()

    def _complete_anthropic(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        api_key: str,
    ) -> str:
        import json
        import urllib.request

        system = ""
        converted: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if role == "system":
                system = (system + "\n" + content).strip()
                continue
            converted.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content,
                }
            )
        model = (
            (os.environ.get("DANA_ANTHROPIC_MODEL") or "").strip()
            or "claude-3-5-haiku-latest"
        )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(num_predict),
            "temperature": float(temperature),
            "messages": converted or [{"role": "user", "content": "Hello"}],
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("content") or []
        texts = [
            str(p.get("text") or "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        self.last_provider = "cloud:anthropic"
        return "\n".join(texts).strip()


def get_default_provider() -> ModelProvider:
    return ModelProvider()


__all__ = (
    "ModelProvider",
    "cloud_fallback_enabled",
    "cloud_primary_enabled",
    "complexity_reject_marker",
    "force_local",
    "get_default_provider",
    "is_complexity_reject",
    "local_model_name",
    "local_vision_model_name",
    "tool_calling_provider",
)
