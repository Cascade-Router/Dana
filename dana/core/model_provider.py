"""Hybrid ModelProvider — local Ollama with optional encrypted cloud fallback."""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Literal

import requests

from dana.core.openai_tool_bridge import _USER_AGENT, build_multimodal_messages, complete_openai_with_tools
from dana.system_health import llm_lock
from dana.tools.schema import openai_tool_calls_to_ir

ProviderKind = Literal["local", "cloud", "auto"]

# Providers whose tool-calling / vision schema is not OpenAI-wire-compatible.
# Gemini and Anthropic each use their own function-calling and image payload
# shapes; bridging them is out of scope for the OpenAI tool-calling bridge.
_NON_OPENAI_SCHEMA_PROVIDERS = frozenset({"gemini", "google", "anthropic"})

_DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
_COMPLEXITY_REJECT = "REJECT: Task too complex for local model"

# Cascade-Router — the local C++ AI gateway (not this module's own
# dana.cascade_router, an unrelated local-Ollama MoA router). It centrally
# manages provider API keys and cascades groq -> gemini -> openai on
# HTTP 429/5xx upstream, in milliseconds, so this bridge no longer needs to
# hold its own provider keys or retry logic for the cloud tool-calling path.
_DEFAULT_GATEWAY_URL = "http://localhost:8080/v1"
_DEFAULT_GATEWAY_MODEL = "cascade-auto"


def gateway_base_url() -> str:
    ensure_dotenv_loaded()
    return (os.environ.get("LLM_GATEWAY_URL") or "").strip().rstrip("/") or _DEFAULT_GATEWAY_URL


def gateway_model_name() -> str:
    ensure_dotenv_loaded()
    return (os.environ.get("DANA_GATEWAY_MODEL") or "").strip() or _DEFAULT_GATEWAY_MODEL

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
    try:
        from dotenv import load_dotenv

        from dana.paths import ENV_PATH

        load_dotenv(ENV_PATH)
        load_dotenv()
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


def tool_calling_provider() -> str:
    """Which ``ModelProvider.complete_with_tool_calls`` provider the ReAct
    loop's hot path should target this turn — the single source of truth
    ``dana.core.react_dispatch._call_llm_once`` defers to instead of a
    hardcoded ``"ollama"``.

    ``"ollama"`` (local, free-per-request but VRAM/context limited) unless
    ``cloud_primary_enabled()`` — then ``DANA_CLOUD_PROVIDER`` if explicitly
    set, else ``"gateway"`` by default: the local Cascade-Router gateway
    (see ``gateway_base_url``) centrally holds provider keys and already
    cascades groq -> gemini -> openai upstream, so this bridge no longer
    needs to pick a single cloud provider itself.

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
    return (os.environ.get("DANA_CLOUD_PROVIDER") or "").strip().lower() or "gateway"


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
        key, base, _, _ = self._resolve_openai_endpoint("ollama")
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

    def _resolve_openai_endpoint(self, provider: str) -> tuple[str, str, str, dict[str, str]]:
        """Return ``(api_key, base_url, model, extra_headers)`` for an
        OpenAI-wire-compatible provider.

        Shared by the plain-text ``_complete_openai_compatible`` path and the
        tool-calling / vision bridge below — one place that knows how each
        provider's env vars map to a key/base/model/headers quadruple.
        ``"ollama"`` targets the local Ollama daemon's own OpenAI-compatible
        ``/v1/chat/completions`` surface (distinct from the native
        ``/api/chat`` path used by ``_complete_local``), which needs no real
        API key. ``extra_headers`` is ``{}`` for every provider except
        ``"openrouter"`` (its recommended, not required, HTTP-Referer/
        X-Title attribution headers) — kept out of ``openai_tool_bridge.py``
        on purpose, since that module has no per-provider knowledge at all.
        """
        ensure_dotenv_loaded()
        if provider == "openrouter":
            # OPENROUTER_API_KEY first; LLM_API_KEY as a generic fallback so
            # a Space owner who already set that name (e.g. copying the
            # style of LLM_GATEWAY_URL/LLM_GATEWAY_API_KEY above) doesn't
            # need a second, provider-specific secret.
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
            model = (
                (os.environ.get("DANA_OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL") or "").strip()
                or "meta-llama/llama-3.3-70b-instruct:free"
            )
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
            return key, base, model, headers
        if provider == "gateway":
            # Local Cascade-Router (C++ gateway) — holds every real provider
            # key itself and cascades groq -> gemini -> openai on 429/5xx
            # upstream, so no per-provider key/model selection happens here.
            # The gateway doesn't require a real bearer token locally, but
            # complete_openai_with_tools always sends one, so fall back to a
            # harmless placeholder when LLM_GATEWAY_API_KEY isn't set.
            key = (self._api_keys.get("gateway") or os.environ.get("LLM_GATEWAY_API_KEY") or "").strip() or "gateway-local"
            base = gateway_base_url()
            model = gateway_model_name()
            return key, base, model, {}
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
        return key, base, model, {}

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

        key, base, model, extra_headers = self._resolve_openai_endpoint(provider)
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
        num_predict: int = 1024,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """OpenAI-schema tool-calling turn against a cloud or local-Ollama endpoint.

        Returns ``{"content": str, "tool_calls": list[ToolCall], "provider": str}``
        — ``tool_calls`` is already Dana's native IR (see
        ``dana.tools.schema.openai_tool_calls_to_ir``), so callers can hand
        results straight to the existing broker/dispatch path
        (``dana.core.agent_loop.execute_tool_call``) with no OpenAI-shape
        parsing of their own. Raises ``NotImplementedError`` for providers
        that don't speak the OpenAI tools schema (Gemini, Anthropic).
        """
        resolved_provider = (provider or cloud_provider_name()).strip().lower()
        if resolved_provider in _NON_OPENAI_SCHEMA_PROVIDERS:
            raise NotImplementedError(
                f"OpenAI tool-calling bridge does not support provider={resolved_provider!r} "
                "(uses a non-OpenAI tool schema)"
            )
        key, base, model, extra_headers = self._resolve_openai_endpoint(resolved_provider)
        # See the matching comment in _complete_openai_compatible above —
        # this is the ReAct loop's actual per-turn call site, so it's the
        # one that matters most both for VRAM-fragmentation-from-concurrent-
        # local-generations (when DANA_CLOUD_PRIMARY is off) AND for letting
        # cloud-routed turns (DANA_CLOUD_PRIMARY on — see
        # dana.core.model_provider.tool_calling_provider) run fully
        # unserialized, since a cloud call has no local VRAM to contend for.
        with llm_lock if resolved_provider == "ollama" else contextlib.nullcontext():
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
            )
        # P1 metric — logged on the SAME line as ttft_ms (see _log_ttft) so
        # the two are directly correlatable turn over turn.
        _log_ttft(model, raw.get("ttft_ms"), tools_schema_bytes=len(json.dumps(tools)) if tools else 0)
        self.last_provider = f"cloud:{resolved_provider}"
        return {
            "content": str(raw.get("content") or "").strip(),
            "tool_calls": openai_tool_calls_to_ir(
                raw.get("tool_calls"), raw_text=str(raw.get("content") or "")
            ),
            "provider": self.last_provider,
        }

    def complete_vision(
        self,
        prompt: str,
        image_b64: str,
        *,
        mime_type: str = "image/png",
        provider: str | None = None,
        num_predict: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Describe/analyze ``image_b64`` via an OpenAI-vision-compatible model.

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
        key, base, model, extra_headers = self._resolve_openai_endpoint(resolved_provider)
        messages = build_multimodal_messages(prompt, image_b64=image_b64, mime_type=mime_type)
        with llm_lock if resolved_provider == "ollama" else contextlib.nullcontext():
            raw = complete_openai_with_tools(
                messages,
                api_key=key,
                base_url=base,
                model=model,
                num_predict=num_predict,
                temperature=temperature,
                extra_headers=extra_headers,
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
    "gateway_base_url",
    "gateway_model_name",
    "get_default_provider",
    "is_complexity_reject",
    "local_model_name",
    "tool_calling_provider",
)
