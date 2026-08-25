"""ModelProvider local / cloud fallback tests (hermetic)."""

from __future__ import annotations

import pytest

from dana.core import model_provider as model_provider_module
from dana.core.model_provider import (
    ModelProvider,
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
    key, _base, _model, _headers = provider._resolve_openai_endpoint("openai")
    assert key == "session-key"


def test_resolve_openai_endpoint_falls_back_to_env_when_no_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = ModelProvider()
    key, _base, _model, _headers = provider._resolve_openai_endpoint("openai")
    assert key == "env-key"


def test_resolve_openai_endpoint_session_key_does_not_leak_to_other_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session-provided "openai" key must never satisfy a different
    provider's endpoint resolution just because a session dict exists."""
    monkeypatch.setenv("GROQ_API_KEY", "groq-env-key")
    provider = ModelProvider(api_keys={"openai": "session-key"})
    key, _base, _model, _headers = provider._resolve_openai_endpoint("groq")
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
    key, base, model, _headers = provider._resolve_openai_endpoint("gemini_openai")
    assert key == "test-gemini-key"
    assert base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert model == "gemini-3.6-flash"


def test_resolve_openai_endpoint_gemini_openai_prefers_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    provider = ModelProvider(api_keys={"gemini": "session-gemini-key"})
    key, _base, _model, _headers = provider._resolve_openai_endpoint("gemini_openai")
    assert key == "session-gemini-key"


def test_resolve_openai_endpoint_gemini_openai_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_API_BASE", "https://custom.example/v1beta/openai/")
    monkeypatch.setenv("DANA_GEMINI_MODEL", "gemini-9.9-ultra")
    provider = ModelProvider()
    _key, base, model, _headers = provider._resolve_openai_endpoint("gemini_openai")
    assert base == "https://custom.example/v1beta/openai/"
    assert model == "gemini-9.9-ultra"


def test_tool_calling_provider_defaults_to_gateway_when_cloud_primary_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud tool-calling now defaults to the local Cascade-Router gateway
    (see model_provider.gateway_base_url) rather than picking Groq/Gemini/
    OpenAI directly — the gateway centrally holds every provider key and
    cascades groq -> gemini -> openai on 429/5xx upstream itself. Any single
    provider remains explicitly selectable via DANA_CLOUD_PROVIDER."""
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.delenv("DANA_CLOUD_PROVIDER", raising=False)
    assert tool_calling_provider() == "gateway"


def test_resolve_openai_endpoint_gateway_defaults_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("DANA_GATEWAY_MODEL", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    provider = ModelProvider()
    key, base, model, _headers = provider._resolve_openai_endpoint("gateway")
    assert base == "http://localhost:8000/v1"
    assert model == "cascade-auto"
    assert key  # placeholder key when none configured — never empty

    monkeypatch.setenv("LLM_GATEWAY_URL", "http://localhost:9000/v1/")
    monkeypatch.setenv("DANA_GATEWAY_MODEL", "cascade-fast")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "real-gateway-key")
    key, base, model, _headers = provider._resolve_openai_endpoint("gateway")
    assert base == "http://localhost:9000/v1"
    assert model == "cascade-fast"
    assert key == "real-gateway-key"


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
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.delenv("DANA_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("HF_SPACE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)
    provider = ModelProvider()
    key, base, model, headers = provider._resolve_openai_endpoint("openrouter")
    assert key == "test-openrouter-key"
    assert base == "https://openrouter.ai/api/v1"
    assert model == "meta-llama/llama-3.3-70b-instruct:free"
    assert headers["HTTP-Referer"] == "https://github.com/"
    assert headers["X-Title"] == "Dana CAD Agent"


def test_resolve_openai_endpoint_openrouter_prefers_session_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-openrouter-key")
    provider = ModelProvider(api_keys={"openrouter": "session-openrouter-key"})
    key, _base, _model, _headers = provider._resolve_openai_endpoint("openrouter")
    assert key == "session-openrouter-key"


def test_resolve_openai_endpoint_openrouter_falls_back_to_generic_llm_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_API_KEY is the generic fallback name for anyone already using
    that convention (matching LLM_GATEWAY_URL/LLM_GATEWAY_API_KEY's own
    naming) — OPENROUTER_API_KEY still wins if both happen to be set."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "generic-llm-key")
    provider = ModelProvider()
    key, _base, _model, _headers = provider._resolve_openai_endpoint("openrouter")
    assert key == "generic-llm-key"


def test_resolve_openai_endpoint_openrouter_respects_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://custom.example/api/v1")
    monkeypatch.setenv("DANA_OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://my-space.hf.space")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Custom Title")
    provider = ModelProvider()
    _key, base, model, headers = provider._resolve_openai_endpoint("openrouter")
    assert base == "https://custom.example/api/v1"
    assert model == "google/gemini-2.0-flash-001"
    assert headers["HTTP-Referer"] == "https://my-space.hf.space"
    assert headers["X-Title"] == "Custom Title"


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
    _key, _base, _model, headers = provider._resolve_openai_endpoint("openai")
    assert headers == {}


def test_tool_calling_provider_supports_openrouter_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.setenv("DANA_CLOUD_PROVIDER", "openrouter")
    assert tool_calling_provider() == "openrouter"
