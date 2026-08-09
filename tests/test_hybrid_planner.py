"""Hybrid Broker (Cloud Planner) settings + routing (offline-safe)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dana.graph.cloud_planner import (
    hybrid_cloud_planner_active,
    planner_mode_label,
)
from dana.llm_client import ask_planner_structured
from dana.llm_schemas import DAGPlan
from dana.settings import (
    is_hybrid_planner_enabled,
    load_dana_settings,
    set_hybrid_planner_enabled,
)


@pytest.fixture()
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("dana.settings.SETTINGS_PATH", str(path))
    import dana.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_CACHE", None)
    yield path
    monkeypatch.setattr(settings_mod, "_CACHE", None)


def test_hybrid_planner_default_false(isolated_settings: Path) -> None:
    assert is_hybrid_planner_enabled() is False
    assert planner_mode_label() == "LOCAL"


def test_hybrid_planner_persists_to_settings_json(isolated_settings: Path) -> None:
    set_hybrid_planner_enabled(True)
    assert is_hybrid_planner_enabled() is True
    raw = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert raw.get("hybrid_planner_enabled") is True
    set_hybrid_planner_enabled(False)
    raw2 = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert raw2.get("hybrid_planner_enabled") is False
    assert load_dana_settings(force_reload=True)["hybrid_planner_enabled"] is False


def test_hybrid_active_requires_api_key(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ensure_dotenv_loaded() re-reads the repo-root .env on every call; on a
    # machine with a real GOOGLE_API_KEY/etc. configured there, that silently
    # resurrects the var right after delenv() below. No-op it so the deleted
    # env actually stays deleted for this assertion.
    monkeypatch.setattr(
        "dana.graph.cloud_planner.ensure_dotenv_loaded", lambda: None
    )
    set_hybrid_planner_enabled(True)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert hybrid_cloud_planner_active() is False
    assert planner_mode_label() == "LOCAL"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    assert hybrid_cloud_planner_active() is True
    assert planner_mode_label() == "HYBRID CLOUD"


def test_ask_planner_structured_uses_cloud_when_hybrid(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_hybrid_planner_enabled(True)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls: list[str] = []

    def fake_cloud(messages, model, max_retries=3, temperature=0.0):  # noqa: ANN001
        calls.append("cloud")
        return DAGPlan.model_validate(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": "Write test to x/test_a.py",
                        "tool_name": "file_editor",
                        "dependencies": [],
                    }
                ]
            }
        )

    def boom_ollama(*_a, **_k):  # noqa: ANN001
        calls.append("ollama")
        raise AssertionError("local ollama must not be called when hybrid cloud active")

    monkeypatch.setattr(
        "dana.graph.cloud_planner.ask_cloud_structured", fake_cloud
    )
    monkeypatch.setattr("dana.llm_client.ask_ollama_structured", boom_ollama)

    plan = ask_planner_structured(
        [{"role": "user", "content": "TDD plan"}],
        DAGPlan,
        max_retries=1,
    )
    assert calls == ["cloud"]
    assert plan.tasks[0].tool_name == "file_editor"


def test_ask_planner_structured_stays_local_by_default(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_hybrid_planner_enabled(False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls: list[str] = []

    def fake_ollama(messages, response_model, **_k):  # noqa: ANN001
        calls.append("ollama")
        return DAGPlan.model_validate(
            {
                "tasks": [
                    {
                        "task_id": 1,
                        "action": "Write impl to x/a.py",
                        "tool_name": "file_editor",
                        "dependencies": [],
                    }
                ]
            }
        )

    def boom_cloud(*_a, **_k):  # noqa: ANN001
        calls.append("cloud")
        raise AssertionError("cloud must not be called when hybrid is off")

    monkeypatch.setattr("dana.llm_client.ask_ollama_structured", fake_ollama)
    monkeypatch.setattr(
        "dana.graph.cloud_planner.ask_cloud_structured", boom_cloud
    )

    plan = ask_planner_structured(
        [{"role": "user", "content": "plan"}],
        DAGPlan,
    )
    assert calls == ["ollama"]
    assert plan.tasks[0].tool_name == "file_editor"
