"""ModelProvider local / cloud fallback tests (hermetic)."""

from __future__ import annotations

import pytest

from dana.core import model_provider as model_provider_module
from dana.core.model_provider import (
    ModelProvider,
    _sanitize_header_value,
    cloud_fallback_enabled,
    complexity_reject_marker,
    is_complexity_reject,
    tool_calling_provider,
)


def test_is_complexity_reject_marker() -> None:
    assert is_complexity_reject(complexity_reject_marker())
    assert is_complexity_reject("REJECT: Task too complex for local model")
    assert not is_complexity_reject("REJECT: Intent is too vague")
    assert not is_complexity_reject("/broker Epic 1: hi")


def test_complete_with_complexity_fallback_routes_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_ALLOW_CLOUD_FALLBACK", "1")
    monkeypatch.delenv("DANA_FORCE_LOCAL", raising=False)
    assert cloud_fallback_enabled() is True

    provider = ModelProvider()
    calls: list[str] = []

    def fake_local(messages, *, num_predict=512, temperature=0.1):
        calls.append("local")
        return complexity_reject_marker()

    def fake_cloud(
        messages,
        *,
        num_predict=512,
        temperature=0.1,
        response_mime_type=None,
    ):
        calls.append("cloud")
        return (
            "/broker Epic 1: Write cloud_mod.py with class CloudMod. "
            "Epic 2: Write tests/test_cloud_mod.py with pytest."
        )

    monkeypatch.setattr(provider, "_complete_local", fake_local)
    monkeypatch.setattr(provider, "_complete_cloud", fake_cloud)

    out = provider.complete_with_complexity_fallback(
        [{"role": "user", "content": "build a huge multi-system controller"}]
    )
    assert calls == ["local", "cloud"]
    assert out.startswith("/broker")
    assert provider.last_provider.startswith("cloud") or "cloud" in calls


def test_resolve_openai_endpoint_prefers_session_key_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = ModelProvider(api_keys={"openai": "session-key"})
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("openai")
    assert key == "session-key"


def test_resolve_openai_endpoint_falls_back_to_env_when_no_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = ModelProvider()
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("openai")
    assert key == "env-key"


def test_resolve_openai_endpoint_session_key_does_not_leak_to_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session-provided "openai" key must never satisfy a different
    provider's endpoint resolution just because a session dict exists."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-env-key")
    provider = ModelProvider(api_keys={"openai": "session-key"})
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("groq")
    assert key == "groq-env-key"


def test_resolve_openai_endpoint_raises_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # This repo's .env has a real OPENAI_API_KEY for local dev — silence
    # ensure_dotenv_loaded() too, or it would just reload the var right
    # back into os.environ before the read below.
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = ModelProvider()
    with pytest.raises(RuntimeError):
        provider._resolve_openai_endpoint("openai")


def test_resolve_openai_endpoint_gemini_openai_uses_expected_base_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killing the 8k TPM Groq ceiling: Gemini's OpenAI-compatible endpoint
    (1,000,000 TPM) is now the default cloud tool-calling target — this
    locks in its exact base URL/model so a future refactor can't silently
    drift back to the old Groq-shaped defaults."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    monkeypatch.delenv("DANA_GEMINI_MODEL", raising=False)
    provider = ModelProvider()
    key, base, model, _headers, _fallback_models = provider._resolve_openai_endpoint("gemini_openai")
    assert key == "test-gemini-key"
    assert base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model == "gemini-3.6-flash"


def test_resolve_openai_endpoint_gemini_openai_prefers_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    provider = ModelProvider(api_keys={"gemini": "session-gemini-key"})
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("gemini_openai")
    assert key == "session-gemini-key"


def test_resolve_openai_endpoint_gemini_openai_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_API_BASE", "https://custom.example/v1beta/openai/")
    monkeypatch.setenv("DANA_GEMINI_MODEL", "gemini-9.9-ultra")
    provider = ModelProvider()
    _key, base, model, _headers, _fallback_models = provider._resolve_openai_endpoint("gemini_openai")
    assert base == "https://custom.example/v1beta/openai/"
    assert model == "gemini-9.9-ultra"


def test_tool_calling_provider_defaults_to_openrouter_when_cloud_primary_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud tool-calling defaults directly to OpenRouter (no local gateway
    process) — OpenRouter's own server-side ``models`` fallback array
    (DANA_OPENROUTER_MODEL, comma-separated) is what retries a 429/5xx
    against the next model upstream. Any single provider remains explicitly
    selectable via DANA_CLOUD_PROVIDER."""
    # This repo's .env now sets DANA_CLOUD_PROVIDER=openrouter for local
    # dev — silence ensure_dotenv_loaded() too, or it reloads that value
    # right back into os.environ before the read below undoes the delenv.
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.delenv("DANA_CLOUD_PROVIDER", raising=False)
    assert tool_calling_provider() == "openrouter"


def test_tool_calling_provider_still_supports_explicit_gemini_openai_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemini_openai must remain fully selectable via the same
    DANA_CLOUD_PROVIDER override every other provider uses — its
    _resolve_openai_endpoint branch is deliberately left intact for
    whenever Google fixes the thought_signature incompatibility."""
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.setenv("DANA_CLOUD_PROVIDER", "gemini_openai")
    assert tool_calling_provider() == "gemini_openai"


def test_tool_calling_provider_is_ollama_when_cloud_primary_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # This repo's .env sets DANA_CLOUD_PRIMARY=true for local dev — an
    # explicit falsy setenv (not delenv) is what actually disables it,
    # since ensure_dotenv_loaded() would just reload the .env value back
    # into an unset os.environ entry.
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "0")
    assert tool_calling_provider() == "ollama"


def test_complete_openai_compatible_anthropic_prefers_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    provider = ModelProvider(api_keys={"anthropic": "session-anthropic-key"})
    captured: dict[str, str] = {}

    def fake_complete_anthropic(messages, *, num_predict, temperature, api_key):
        captured["api_key"] = api_key
        return "ok"

    monkeypatch.setattr(provider, "_complete_anthropic", fake_complete_anthropic)
    out = provider._complete_openai_compatible(
        [{"role": "user", "content": "hi"}], num_predict=10, temperature=0.1, provider="anthropic"
    )
    assert out == "ok"
    assert captured["api_key"] == "session-anthropic-key"


def test_complete_openai_compatible_anthropic_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    provider = ModelProvider()
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        provider,
        "_complete_anthropic",
        lambda messages, *, num_predict, temperature, api_key: captured.setdefault("api_key", api_key) or "ok",
    )
    provider._complete_openai_compatible(
        [{"role": "user", "content": "hi"}], num_predict=10, temperature=0.1, provider="anthropic"
    )
    assert captured["api_key"] == "env-anthropic-key"


def test_complete_with_complexity_fallback_stays_local_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DANA_ALLOW_CLOUD_FALLBACK", "0")
    provider = ModelProvider()

    monkeypatch.setattr(
        provider,
        "_complete_local",
        lambda *a, **k: complexity_reject_marker(),
    )

    def boom(*a, **k):
        raise AssertionError("cloud must not be called")

    monkeypatch.setattr(provider, "_complete_cloud", boom)
    out = provider.complete_with_complexity_fallback(
        [{"role": "user", "content": "huge task"}]
    )
    assert is_complexity_reject(out)


# --------------------------------------------------------------------------
# OpenRouter provider
# --------------------------------------------------------------------------


def test_resolve_openai_endpoint_openrouter_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # This repo's .env now sets DANA_OPENROUTER_MODEL/OPENROUTER_SITE_URL/
    # OPENROUTER_APP_TITLE for local dev — silence ensure_dotenv_loaded()
    # too, or it reloads those values right back into os.environ before
    # the delenv calls below take effect.
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.delenv("DANA_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("HF_SPACE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
    provider = ModelProvider()
    key, base, model, headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert key == "test-openrouter-key"
    assert base == "https://openrouter.ai/api/v1"
    assert model == "meta-llama/llama-3.3-70b-instruct:free"
    assert headers["HTTP-Referer"] == "https://github.com/"
    assert headers["X-Title"] == "Dana CAD Agent"


def test_resolve_openai_endpoint_openrouter_prefers_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")
    provider = ModelProvider(api_keys={"openrouter": "session-openrouter-key"})
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert key == "session-openrouter-key"


def test_resolve_openai_endpoint_openrouter_falls_back_to_generic_llm_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_API_KEY is the generic fallback name for anyone already using
    that convention — OPENROUTER_API_KEY still wins if both happen to be
    set."""
    # This repo's .env now has a real OPENROUTER_API_KEY for local dev —
    # silence ensure_dotenv_loaded() too, or it reloads that value right
    # back into os.environ before the delenv below takes effect.
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "generic-llm-key")
    provider = ModelProvider()
    key, _base, _model, _headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert key == "generic-llm-key"


def test_resolve_openai_endpoint_openrouter_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://custom.example/api/v1")
    monkeypatch.setenv("DANA_OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://my-space.hf.space")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Custom Title")
    provider = ModelProvider()
    _key, base, model, headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert base == "https://custom.example/api/v1"
    assert model == "google/gemini-2.0-flash-001"
    assert headers["HTTP-Referer"] == "https://my-space.hf.space"
    assert headers["X-Title"] == "Custom Title"


def test_resolve_openai_endpoint_openrouter_single_model_has_no_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("DANA_OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    provider = ModelProvider()
    _key, _base, model, _headers, fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert model == "google/gemini-2.0-flash-001"
    assert fallback_models == []


def test_resolve_openai_endpoint_openrouter_parses_comma_separated_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DANA_OPENROUTER_MODEL accepts a comma-separated list for OpenRouter's
    native server-side model cascade — the first entry is the primary
    ``model``, the rest ride along as ``fallback_models``."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv(
        "DANA_OPENROUTER_MODEL",
        "google/gemma-4-26b-a4b-it:free, openai/gpt-oss-120b:free ,openrouter/free",
    )
    provider = ModelProvider()
    _key, _base, model, _headers, fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert model == "google/gemma-4-26b-a4b-it:free"
    assert fallback_models == ["openai/gpt-oss-120b:free", "openrouter/free"]


def test_resolve_openai_endpoint_openrouter_raises_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = ModelProvider()
    with pytest.raises(RuntimeError):
        provider._resolve_openai_endpoint("openrouter")


def test_resolve_openai_endpoint_non_openrouter_providers_have_no_extra_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter's attribution headers must never leak onto an unrelated
    provider's request just because _resolve_openai_endpoint's return shape
    grew a 4th element."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = ModelProvider()
    _key, _base, _model, headers, _fallback_models = provider._resolve_openai_endpoint("openai")
    assert headers == {}


def test_tool_calling_provider_supports_openrouter_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.setenv("DANA_CLOUD_PROVIDER", "openrouter")
    assert tool_calling_provider() == "openrouter"


# --------------------------------------------------------------------------
# Header sanitization — regression coverage for a real crash:
# OPENROUTER_APP_TITLE="Dānā CAD Agent" raised UnicodeEncodeError deep in
# http.client at request-send time (headers are transmitted as latin-1,
# a stricter subset of what a Python str can hold), not at the point these
# dicts get built — so nothing in this module's own code path ever saw an
# exception, making it the kind of bug a plain "does this raise" test at
# construction time wouldn't have caught either. These tests go through
# the same _resolve_openai_endpoint("openrouter") call site a real request
# does, not just the helper in isolation.
# --------------------------------------------------------------------------


def test_sanitize_header_value_strips_non_ascii() -> None:
    assert _sanitize_header_value("Dānā CAD Agent", fallback="x") == "Dn CAD Agent"


def test_sanitize_header_value_falls_back_when_stripping_empties_the_string() -> None:
    assert _sanitize_header_value("日本語", fallback="Dana CAD Agent") == "Dana CAD Agent"


def test_sanitize_header_value_passes_through_plain_ascii_unchanged() -> None:
    assert _sanitize_header_value("Dana CAD Agent", fallback="x") == "Dana CAD Agent"


def test_resolve_openai_endpoint_openrouter_sanitizes_non_ascii_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact real-world trigger: a non-ASCII OPENROUTER_APP_TITLE must
    never reach the returned headers dict un-sanitized, and resolving the
    endpoint must not raise."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Dānā CAD Agent")
    provider = ModelProvider()
    _key, _base, _model, headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert headers["X-Title"] == "Dn CAD Agent"
    headers["X-Title"].encode("latin-1")  # must not raise UnicodeEncodeError


def test_resolve_openai_endpoint_openrouter_sanitizes_non_ascii_site_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://my-space.hf.space/日本語")
    provider = ModelProvider()
    _key, _base, _model, headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    headers["HTTP-Referer"].encode("latin-1")  # must not raise UnicodeEncodeError


def test_resolve_openai_endpoint_openrouter_all_non_ascii_title_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A title that's ENTIRELY non-ASCII strips down to nothing — must fall
    back to the plain-ASCII default instead of sending an empty header."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "日本語")
    provider = ModelProvider()
    _key, _base, _model, headers, _fallback_models = provider._resolve_openai_endpoint("openrouter")
    assert headers["X-Title"] == "Dana CAD Agent"
