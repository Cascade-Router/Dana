"""Dynamic LLM Router tests (hermetic, no network): fleet ordering by
priority/cost, context-window skipping for oversized prompts, and
multi-hop fallback on a 429/503-shaped failure.

Integration tests point ``dana.core.routing_config.ROUTING_CONFIG_PATH``
at a real temp YAML file (same "patch the path, exercise the real reader"
convention ``tests/api/test_system_env.py`` already uses for ``ENV_PATH``)
rather than mocking the loader function itself, so the real YAML parse +
validation path is exercised too, not just the selection logic in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dana.core import model_provider as model_provider_module
from dana.core import model_registry as model_registry_module
from dana.core import routing_config as routing_config_module
from dana.core.llm_router import FleetEntry, estimate_tokens, report_fleet_entry_failure, select_fleet_chain
from dana.core.model_provider import ModelProvider


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same rationale as tests/test_model_provider.py's fixture of the same
    name: _resolve_openai_endpoint (reached via complete_with_tool_calls'
    router path here) calls ensure_dotenv_loaded() internally, which would
    otherwise reload this repo's real .env over whatever these tests set."""
    monkeypatch.setattr(model_provider_module, "ensure_dotenv_loaded", lambda: None)


@pytest.fixture(autouse=True)
def _reset_shared_registry_runtime_errors():
    """report_fleet_entry_failure reports into the process-wide default
    ModelRegistryService (dana.core.model_registry.get_registry_service()),
    not a per-test instance — reset it around every test in this file so one
    test's simulated 429 can't leak into the next test's assertions. Captures
    the service reference ONCE rather than calling get_registry_service()
    again at teardown — a test that monkeypatches get_registry_service
    itself (to simulate the registry being unavailable) would otherwise make
    this fixture's own teardown call blow up too."""
    service = model_registry_module.get_registry_service()
    service.runtime_errors.clear()
    yield
    service.runtime_errors.clear()


def _entry(
    id: str,
    *,
    provider: str = "openrouter",
    model: str = "m",
    context_window: int = 100_000,
    priority: int = 1,
    cost: float = 0.0,
) -> FleetEntry:
    return FleetEntry(
        id=id, provider=provider, model=model, context_window=context_window,
        priority=priority, cost_per_1m_prompt=cost,
    )


# --------------------------------------------------------------------------
# Pure selection logic — no I/O
# --------------------------------------------------------------------------


def test_select_fleet_chain_orders_by_priority_then_cost() -> None:
    entries = [_entry("b", priority=1, cost=0.5), _entry("a", priority=1, cost=0.1), _entry("c", priority=2, cost=0.0)]
    chain = select_fleet_chain(entries, {}, estimated_tokens=10)
    assert [e.id for e in chain] == ["a", "b", "c"]


def test_select_fleet_chain_skips_entries_too_small_for_the_prompt() -> None:
    small = _entry("small", context_window=1_000, priority=1)
    big = _entry("big", context_window=1_000_000, priority=2)
    chain = select_fleet_chain([small, big], {}, estimated_tokens=500_000)
    assert chain[0].id == "big"  # only entry that actually fits, tried first
    assert chain[-1].id == "small"  # still attempted as a last resort, not dropped


def test_select_fleet_chain_always_ends_with_terminal_fallback() -> None:
    a = _entry("a", context_window=1_000_000, priority=1)
    terminal = _entry("term", context_window=500, priority=2)  # would sort before "a" doesn't fit anyway; priority irrelevant here
    chain = select_fleet_chain([a, terminal], {"terminal_fallback": "term"}, estimated_tokens=10)
    assert chain[-1].id == "term"


def test_select_fleet_chain_respects_custom_safety_margin() -> None:
    entry = _entry("tight", context_window=1_000, priority=1)
    # 1000 * 0.5 = 500 < 600 estimated tokens -> doesn't fit under a strict margin.
    chain = select_fleet_chain([entry], {"context_window_safety_margin": 0.5}, estimated_tokens=600)
    assert chain == [entry]  # only entry, still attempted, just not preferred over anything


def test_estimate_tokens_scales_with_content_length() -> None:
    small = estimate_tokens([{"role": "user", "content": "hi"}])
    big = estimate_tokens([{"role": "user", "content": "x" * 4000}])
    assert big > small
    assert big == pytest.approx(1000, rel=0.05)


def test_estimate_tokens_includes_tool_schema_size() -> None:
    messages = [{"role": "user", "content": "hi"}]
    without_tools = estimate_tokens(messages, None)
    with_tools = estimate_tokens(messages, [{"type": "function", "function": {"name": "x" * 2000}}])
    assert with_tools > without_tools


# --------------------------------------------------------------------------
# resolve_chain — the None-if-absent contract
# --------------------------------------------------------------------------


def test_resolve_chain_returns_none_when_routing_config_path_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dana.core.llm_router import resolve_chain

    monkeypatch.setattr(routing_config_module, "ROUTING_CONFIG_PATH", tmp_path / "does_not_exist.yaml")
    assert resolve_chain([{"role": "user", "content": "hi"}]) is None


def test_resolve_chain_returns_none_for_malformed_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.core.llm_router import resolve_chain

    bad = tmp_path / "routing_config.yaml"
    bad.write_text("fleet: not_a_list\n", encoding="utf-8")
    monkeypatch.setattr(routing_config_module, "ROUTING_CONFIG_PATH", bad)
    assert resolve_chain([{"role": "user", "content": "hi"}]) is None


# --------------------------------------------------------------------------
# Integration — ModelProvider.complete_with_tool_calls delegating to the router
# --------------------------------------------------------------------------

_FLEET_RAW = {
    "version": 1,
    "defaults": {"context_window_safety_margin": 0.85, "terminal_fallback": "local"},
    "fleet": [
        {
            "id": "cheap", "provider": "openrouter", "model": "free/model",
            "api_key_env": "OPENROUTER_API_KEY", "context_window": 100000,
            "cost_per_1m_prompt": 0.0, "cost_per_1m_completion": 0.0, "priority": 1,
        },
        {
            "id": "big", "provider": "gemini_openai", "model": "gemini-x",
            "api_key_env": "GEMINI_API_KEY", "context_window": 1000000,
            "cost_per_1m_prompt": 0.1, "cost_per_1m_completion": 0.1, "priority": 2,
        },
        {"id": "local", "provider": "ollama", "model": "qwen", "context_window": 8192, "priority": 99},
    ],
}


@pytest.fixture()
def fake_fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "routing_config.yaml"
    config_path.write_text(yaml.safe_dump(_FLEET_RAW), encoding="utf-8")
    monkeypatch.setattr(routing_config_module, "ROUTING_CONFIG_PATH", config_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


def _canned_ok(**_: object) -> dict:
    return {
        "content": "ok", "tool_calls": [], "ttft_ms": 1.0,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "finish_reason": "stop",
    }


def test_complete_with_tool_calls_picks_cheapest_fitting_entry(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    calls: list[str] = []

    def fake_complete(messages, *, api_key, base_url, model, **kw):
        calls.append(model)
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    result = ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])
    assert calls == ["free/model"]
    assert result["provider"] == "router:cheap"


def test_complete_with_tool_calls_skips_to_larger_window_for_oversized_prompt(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    calls: list[str] = []

    def fake_complete(messages, *, api_key, base_url, model, **kw):
        calls.append(model)
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    huge = [{"role": "user", "content": "x" * 500_000}]  # ~125k estimated tokens > cheap/local's windows
    result = ModelProvider().complete_with_tool_calls(huge, tools=[])
    assert calls == ["gemini-x"]
    assert result["provider"] == "router:big"


def test_complete_with_tool_calls_advances_chain_on_429(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    calls: list[str] = []

    def fake_complete(messages, *, api_key, base_url, model, **kw):
        calls.append(model)
        if model == "free/model":
            raise RuntimeError("cloud HTTP 429: rate limited (simulated)")
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    result = ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])
    assert calls == ["free/model", "gemini-x"]
    assert result["provider"] == "router:big"


def test_complete_with_tool_calls_raises_with_all_errors_when_every_entry_fails(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    def fake_complete(*a, **kw):
        raise RuntimeError("cloud HTTP 503: simulated outage")

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    with pytest.raises(RuntimeError, match="every fleet entry failed"):
        ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])


def test_complete_with_tool_calls_explicit_provider_bypasses_the_router(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    """A caller that names a specific provider has already made its own
    routing decision — the router must not override it."""
    calls: list[str] = []

    def fake_complete(messages, *, base_url, model, **kw):
        calls.append(model)
        return _canned_ok()

    # provider="ollama" goes through Ollama's native /api/chat bridge, not the
    # OpenAI-compatible one the router tests above patch.
    monkeypatch.setattr(model_provider_module, "complete_ollama_native_with_tools", fake_complete)
    monkeypatch.setenv("DANA_OPENAI_TOOLS_MODEL", "forced-ollama-model")
    result = ModelProvider().complete_with_tool_calls(
        [{"role": "user", "content": "hi"}], tools=[], provider="ollama"
    )
    assert calls == ["forced-ollama-model"]
    assert not result["provider"].startswith("router:")


@pytest.fixture()
def no_fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No routing_config.yaml and no provider env: the out-of-the-box state."""
    monkeypatch.setattr(routing_config_module, "ROUTING_CONFIG_PATH", tmp_path / "does_not_exist.yaml")
    monkeypatch.delenv("DANA_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("DANA_CLOUD_PRIMARY", raising=False)


def test_complete_with_tool_calls_without_fleet_or_provider_env_uses_local_ollama(
    monkeypatch: pytest.MonkeyPatch, no_fleet: None
) -> None:
    """Regression: this used to resolve to cloud_provider_name()'s bare
    "gemini" default and raise NotImplementedError on every ReAct turn."""
    calls: list[str] = []

    def fake_complete(messages, *, base_url, model, **kw):
        calls.append(model)
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_ollama_native_with_tools", fake_complete)
    result = ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])
    assert len(calls) == 1
    assert result["provider"] == "cloud:ollama"


def test_complete_with_tool_calls_without_fleet_but_cloud_primary_uses_openrouter(
    monkeypatch: pytest.MonkeyPatch, no_fleet: None
) -> None:
    monkeypatch.setenv("DANA_CLOUD_PRIMARY", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    calls: list[str] = []

    def fake_complete(messages, *, api_key, base_url, model, **kw):
        calls.append(base_url)
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    result = ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])
    assert result["provider"] == "cloud:openrouter"
    assert calls and "openrouter" in calls[0]


# --------------------------------------------------------------------------
# report_fleet_entry_failure — the circuit breaker wired into the Model Registry
# --------------------------------------------------------------------------


def test_report_fleet_entry_failure_parses_the_http_status_code() -> None:
    report_fleet_entry_failure(_entry("x", model="some-model"), RuntimeError("cloud HTTP 404: Not Found -- body"))
    error = model_registry_module.get_registry_service().runtime_errors["some-model"]
    assert error["status_code"] == 404
    assert "Not Found" in error["message"]


def test_report_fleet_entry_failure_defaults_status_code_to_zero_without_one() -> None:
    report_fleet_entry_failure(
        _entry("y", model="other-model"), TimeoutError("model endpoint unreachable or stalled: timed out")
    )
    error = model_registry_module.get_registry_service().runtime_errors["other-model"]
    assert error["status_code"] == 0


def test_report_fleet_entry_failure_never_raises_even_if_the_registry_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("registry unavailable (simulated)")

    monkeypatch.setattr(model_registry_module, "get_registry_service", _boom)
    report_fleet_entry_failure(_entry("z"), RuntimeError("cloud HTTP 500: simulated"))  # must not raise


def test_complete_with_tool_calls_advancing_on_429_reports_into_the_model_registry(
    monkeypatch: pytest.MonkeyPatch, fake_fleet: None
) -> None:
    """Same 429 scenario as test_complete_with_tool_calls_advances_chain_on_429
    above, but asserting the circuit-breaker side effect: the failing entry's
    MODEL id (not its fleet entry id) ends up in the shared registry's
    runtime_errors, which is what the frontend's Model Registry panel reads."""

    def fake_complete(messages, *, api_key, base_url, model, **kw):
        if model == "free/model":
            raise RuntimeError("cloud HTTP 429: rate limited (simulated)")
        return _canned_ok()

    monkeypatch.setattr(model_provider_module, "complete_openai_with_tools", fake_complete)
    ModelProvider().complete_with_tool_calls([{"role": "user", "content": "hi"}], tools=[])

    error = model_registry_module.get_registry_service().runtime_errors.get("free/model")
    assert error is not None
    assert error["status_code"] == 429
    assert "rate limited" in error["message"]
