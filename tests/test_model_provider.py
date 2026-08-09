"""ModelProvider local / cloud fallback tests (hermetic)."""

from __future__ import annotations

import pytest

from dana.core.model_provider import (
    ModelProvider,
    cloud_fallback_enabled,
    complexity_reject_marker,
    is_complexity_reject,
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
