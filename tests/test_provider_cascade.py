"""Cascade LLM Router — provider failover / circuit-breaker tests (hermetic)."""

from __future__ import annotations

import pytest

from dana.core import model_provider as model_provider_module
from dana.core import provider_cascade as cascade_module
from dana.core.model_provider import ModelProvider
from dana.core.provider_cascade import (
    TOOL_CALL_LADDER,
    available_providers,
    complete_with_tool_calls_cascading,
    cooldown_remaining,
    is_cooling_down,
    provider_has_key,
    reset_cooldowns,
)

_ALL_KEY_ENVS = (
    "GROQ_API_KEY",
    "CLOUD_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    # Real .env values in this dev repo would otherwise leak through
    # ensure_dotenv_loaded() and make "no keys configured" tests flaky.
    # Patched in BOTH modules: provider_cascade.provider_has_key calls its
    # own imported reference, but ModelProvider() itself (constructed in
    # several tests below) independently calls model_provider's copy via
    # local_model_name() at __init__ time — missing either one re-loads the
    # real .env mid-test.
    monkeypatch.setattr(cascade_module, "ensure_dotenv_loaded", lambda: None)
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)
    for name in _ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    reset_cooldowns()
    yield
    reset_cooldowns()


def _ok_result(provider: str) -> dict:
    return {"content": "hi", "tool_calls": [], "provider": f"cloud:{provider}"}


def test_provider_has_key_checks_env_and_byok(monkeypatch: pytest.MonkeyPatch) -> None:
    assert provider_has_key("groq") is False
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    assert provider_has_key("groq") is True

    assert provider_has_key("openai") is False
    assert provider_has_key("openai", {"openai": "sk-byok"}) is True


def test_available_providers_filters_by_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert available_providers() == ["groq", "openai"]


def test_available_providers_respects_ladder_order_regardless_of_env_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    # Ladder order (groq, gemini_openai, openai, anthropic) wins regardless
    # of the order keys were set in.
    assert available_providers() == ["groq", "openai"]


def test_available_providers_falls_back_to_ladder_head_when_nothing_configured() -> None:
    assert available_providers() == [TOOL_CALL_LADDER[0]]


def test_available_providers_ignores_cooldown_when_everything_configured_is_cooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    cascade_module._start_cooldown("groq", seconds=60.0)
    assert is_cooling_down("groq") is True
    # groq is the ONLY configured provider — cooldown must not remove every
    # candidate, or the turn would have literally nothing left to try.
    assert available_providers() == ["groq"]


def test_available_providers_prefers_non_cooling_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    cascade_module._start_cooldown("groq", seconds=60.0)
    assert available_providers() == ["openai"]


def test_cooldown_remaining_reports_positive_then_zero_after_reset() -> None:
    cascade_module._start_cooldown("groq", seconds=30.0)
    assert 0.0 < cooldown_remaining("groq") <= 30.0
    reset_cooldowns()
    assert cooldown_remaining("groq") == 0.0
    assert is_cooling_down("groq") is False


def test_cascade_fails_over_from_groq_429_to_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact scenario from the spec: Groq rate-limited -> Gemini."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("GEMINI_API_KEY", "gm_real")

    calls: list[str] = []

    def fake_complete(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        calls.append(provider)
        if provider == "groq":
            raise RuntimeError("cloud HTTP 429: Too Many Requests -- {\"error\": \"rate limited\"}")
        return _ok_result(provider)

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete)

    provider = ModelProvider()
    result = complete_with_tool_calls_cascading(provider, [{"role": "user", "content": "hi"}], tools=[])

    assert calls == ["groq", "gemini_openai"]
    assert result["provider"] == "cloud:gemini_openai"
    assert is_cooling_down("groq") is True
    assert is_cooling_down("gemini_openai") is False


def test_cascade_fails_over_on_503_and_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

    def fake_complete(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        if provider == "groq":
            raise RuntimeError("cloud HTTP 503: Service Unavailable -- {}")
        return _ok_result(provider)

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete)
    result = complete_with_tool_calls_cascading(
        ModelProvider(), [{"role": "user", "content": "hi"}], tools=[]
    )
    assert result["provider"] == "cloud:openai"

    reset_cooldowns()

    def fake_complete_timeout(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        if provider == "groq":
            raise TimeoutError("model endpoint unreachable or stalled: timed out")
        return _ok_result(provider)

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete_timeout)
    result = complete_with_tool_calls_cascading(
        ModelProvider(), [{"role": "user", "content": "hi"}], tools=[]
    )
    assert result["provider"] == "cloud:openai"


def test_cascade_does_not_retry_non_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 (bad key) is a real, permanent failure — must propagate
    immediately rather than burning a cooldown/cascade cycle on it, and the
    second provider must never even be attempted."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_bad")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

    calls: list[str] = []

    def fake_complete(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        calls.append(provider)
        raise RuntimeError("cloud HTTP 401: Unauthorized -- {}")

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete)

    with pytest.raises(RuntimeError, match="401"):
        complete_with_tool_calls_cascading(ModelProvider(), [{"role": "user", "content": "hi"}], tools=[])

    assert calls == ["groq"]
    assert is_cooling_down("groq") is False


def test_cascade_skips_provider_that_cannot_speak_tool_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic is last in the ladder and can't do OpenAI-schema tool
    calls at all — configuring ONLY it should surface that NotImplementedError
    cleanly rather than a confusing cooldown/HTTP-shaped error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")

    def fake_complete(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        raise NotImplementedError(f"OpenAI tool-calling bridge does not support provider={provider!r}")

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete)

    with pytest.raises(NotImplementedError):
        complete_with_tool_calls_cascading(ModelProvider(), [{"role": "user", "content": "hi"}], tools=[])
    assert is_cooling_down("anthropic") is False


def test_cascade_raises_original_error_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No keys at all: the ladder's head is attempted anyway (so the normal
    "no API key configured" error surfaces), not a synthetic new error."""

    def fake_complete(self, messages, *, tools, provider=None, tool_choice=None, num_predict=1024, temperature=0.1):
        raise RuntimeError(f"No API key configured for cloud provider={provider}")

    monkeypatch.setattr(ModelProvider, "complete_with_tool_calls", fake_complete)
    with pytest.raises(RuntimeError, match="No API key configured"):
        complete_with_tool_calls_cascading(ModelProvider(), [{"role": "user", "content": "hi"}], tools=[])
