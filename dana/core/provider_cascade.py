"""Cascade LLM Router — in-process, fault-tolerant provider failover for the
ReAct loop's cloud tool-calling hot path.

Distinct from two other things that already share the "cascade" name in
this codebase:

- ``dana.cascade_router`` — an unrelated local-Ollama Mixture-of-Agents
  router (fast chat model vs. a two-stage vision+reasoner escalation). No
  cloud providers, no HTTP retries, nothing in common with this module
  besides the English word "cascade".
- The external "Cascade-Router" C++ gateway ``dana.core.model_provider``
  talks to via ``gateway_base_url()`` (default ``http://localhost:8080/v1``,
  provider name ``"gateway"``) — documented there as already cascading
  groq -> gemini -> openai on 429/5xx, but that service is not implemented
  anywhere in this repo; it's an external dependency an operator would have
  to run separately. This module is the first real, in-process
  implementation of that same idea, so a session gets working failover
  whether or not that gateway happens to be running.

Design: a fixed priority ladder of OpenAI-wire-compatible providers
(cheapest/fastest first), filtered per call to whichever are actually
configured (session BYOK or environment — the "dynamic provider registry"),
with a short-term cooldown (circuit breaker) on any provider that just
failed with a rate-limit/server-error/timeout, so a turn doesn't keep
re-hammering a provider it already knows is down.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dana.core.model_provider import ModelProvider, ensure_dotenv_loaded

# Cheapest/fastest first. Anthropic is OpenAI-tool-schema-incompatible (see
# dana.core.model_provider._NON_OPENAI_SCHEMA_PROVIDERS) — kept at the tail
# of the ladder anyway so a session where it's the ONLY configured provider
# still gets a clean, logged skip instead of the ladder silently omitting it
# from `available_providers`.
TOOL_CALL_LADDER: tuple[str, ...] = ("groq", "gemini_openai", "openai", "anthropic")

# Providers this router will cascade across at all — see _call_llm_once in
# react_dispatch.py. "gateway" is included so cloud-primary sessions with no
# explicit DANA_CLOUD_PROVIDER (tool_calling_provider()'s own default) get
# this ladder instead of trying to reach the not-implemented external
# gateway.
CASCADABLE_TARGETS: frozenset[str] = frozenset({"gateway", *TOOL_CALL_LADDER})

_DISPLAY_NAMES: dict[str, str] = {
    "groq": "Groq",
    "gemini_openai": "Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gateway": "Gateway",
}

_COOLDOWN_SECONDS = 60.0
_RETRYABLE_STATUSES = frozenset({429, 503})
_HTTP_STATUS_RE = re.compile(r"cloud HTTP (\d+):")

# provider -> time.time() the cooldown ends. Process-wide by design: a rate
# limit is a property of the provider's account/key, not of one chat
# session, so concurrent sessions should all see the same breaker state
# instead of each independently re-discovering the same limit.
_cooldowns: dict[str, float] = {}


def display_name(provider: str) -> str:
    return _DISPLAY_NAMES.get(provider, provider)


def is_cooling_down(provider: str) -> bool:
    until = _cooldowns.get(provider)
    return until is not None and time.time() < until


def cooldown_remaining(provider: str) -> float:
    until = _cooldowns.get(provider)
    return max(0.0, until - time.time()) if until is not None else 0.0


def _start_cooldown(provider: str, *, seconds: float = _COOLDOWN_SECONDS) -> None:
    _cooldowns[provider] = time.time() + seconds


def reset_cooldowns() -> None:
    """Test/ops hook — clears every provider's circuit-breaker state."""
    _cooldowns.clear()


def provider_has_key(provider: str, api_keys: dict[str, str] | None = None) -> bool:
    """Whether ``provider`` has a usable key right now.

    Mirrors ``ModelProvider._resolve_openai_endpoint``'s own key-resolution
    order for each provider exactly (session BYOK first where that function
    checks it, then the matching environment variable) — this is a
    read-only availability check, never a second copy of the actual
    resolution logic that could drift from it.
    """
    ensure_dotenv_loaded()
    keys = api_keys or {}
    if provider == "groq":
        return bool((os.environ.get("GROQ_API_KEY") or os.environ.get("CLOUD_API_KEY") or "").strip())
    if provider == "gemini_openai":
        return bool((keys.get("gemini") or os.environ.get("GEMINI_API_KEY") or "").strip())
    if provider == "openai":
        return bool((keys.get("openai") or os.environ.get("OPENAI_API_KEY") or "").strip())
    if provider == "anthropic":
        return bool((keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    return True  # ollama/gateway need no real key to attempt


def available_providers(
    api_keys: dict[str, str] | None = None,
    *,
    ladder: tuple[str, ...] = TOOL_CALL_LADDER,
) -> list[str]:
    """The ladder, filtered to configured providers, in priority order.

    A provider currently in cooldown is deprioritized but not dropped
    outright: if every configured provider happens to be cooling down at
    once, this returns them anyway (ignoring cooldown) rather than refusing
    to try anything — a cooldown is a heuristic, not a guarantee the
    provider is still down. If NOTHING is configured at all, returns the
    ladder's first entry so the normal "no API key configured" error still
    surfaces from the real call site, instead of this function inventing a
    new error shape.
    """
    keyed = [p for p in ladder if provider_has_key(p, api_keys)]
    if not keyed:
        return list(ladder[:1])
    fresh = [p for p in keyed if not is_cooling_down(p)]
    return fresh or keyed


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, RuntimeError):
        match = _HTTP_STATUS_RE.search(str(exc))
        if match and int(match.group(1)) in _RETRYABLE_STATUSES:
            return True
    return False


def complete_with_tool_calls_cascading(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None = None,
    num_predict: int = 1024,
    temperature: float = 0.1,
    ladder: tuple[str, ...] = TOOL_CALL_LADDER,
) -> dict[str, Any]:
    """``ModelProvider.complete_with_tool_calls``, retried down ``ladder`` on
    a 429/503/connection-timeout instead of failing the whole ReAct turn on
    the first provider's outage.

    A provider with no configured key is never attempted (see
    ``available_providers``). A provider that can't speak the OpenAI
    tool-calling schema at all (``NotImplementedError`` — Anthropic today)
    is skipped without a cooldown, since retrying it would just raise the
    identical error again; it isn't "down," it's structurally unusable for
    this call shape. Re-raises the last real error once every candidate is
    exhausted, so a caller with no working provider still sees a genuine
    failure rather than a silently swallowed one.

    Not responsible for the OUTER per-turn timeout (``asyncio.wait_for`` in
    ``dana.core.react_dispatch._call_llm_once``) — a provider that stalls
    long enough to blow that whole budget on its own still ends the turn via
    the existing timeout-fallback path exactly as before; this router only
    improves the common case (a fast 429/503 rejection), which is what
    "cascading on rate limit" actually means in practice.
    """
    # getattr, not a direct attribute access: some callers (test doubles
    # standing in for ModelProvider) duck-type only complete_with_tool_calls
    # itself and have no _api_keys at all — treat that the same as "no BYOK
    # keys configured" rather than crashing.
    candidates = available_providers(getattr(provider, "_api_keys", None), ladder=ladder)
    last_exc: Exception | None = None
    for i, name in enumerate(candidates):
        try:
            return provider.complete_with_tool_calls(
                messages,
                tools=tools,
                provider=name,
                tool_choice=tool_choice,
                num_predict=num_predict,
                temperature=temperature,
            )
        except NotImplementedError as exc:
            print(
                f"[CascadeRouter] {display_name(name)} does not support tool-calling — skipping.",
                flush=True,
            )
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001 — classified below, re-raised if not retryable
            if not _is_retryable(exc):
                raise
            _start_cooldown(name)
            last_exc = exc
            nxt = candidates[i + 1] if i + 1 < len(candidates) else None
            if nxt:
                print(
                    f"[CascadeRouter] {display_name(name)} rate-limited -> cascading to "
                    f"{display_name(nxt)}. ({exc})",
                    flush=True,
                )
            else:
                print(
                    f"[CascadeRouter] {display_name(name)} rate-limited/unavailable, "
                    f"no further providers configured. ({exc})",
                    flush=True,
                )
            continue
    assert last_exc is not None
    raise last_exc


__all__ = (
    "CASCADABLE_TARGETS",
    "TOOL_CALL_LADDER",
    "available_providers",
    "complete_with_tool_calls_cascading",
    "cooldown_remaining",
    "display_name",
    "is_cooling_down",
    "provider_has_key",
    "reset_cooldowns",
)
