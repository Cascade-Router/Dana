"""Stage 9.2 — smoke checks for HF Gradio wrapper (no network / no GPU)."""

from __future__ import annotations

import os


def test_predict_signature_and_cloud_env(monkeypatch) -> None:
    monkeypatch.setenv("DONNA_CLOUD", "1")
    monkeypatch.setenv("DONNA_HITL_AUTO_APPROVE", "1")
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_VAULT_KEY", "test-key")

    from deploy import cloud_bridge as cb

    cb.apply_cloud_mode()
    assert cb.is_cloud_mode() is True
    assert os.environ.get("DONNA_HITL_AUTO_APPROVE") == "1"

    # Empty prompt short-circuits without LangGraph.
    assert "Type a command" in cb.run_text_command("   ")


def test_cloud_execute_mocks_vision() -> None:
    from donna.tools.schema import ToolCall
    from deploy.cloud_bridge import _cloud_execute

    obs = _cloud_execute(
        ToolCall(tool_id="analyze_visual_context", arguments={"source": "screen"})
    )
    assert "[CLOUD]" in obs
    assert "mocked" in obs.lower()


def test_hf_app_predict_is_str_to_str(monkeypatch) -> None:
    """Gradio Interface schema requires predict(str) -> str for {"data":[prompt]}."""
    monkeypatch.setenv("DONNA_CLOUD", "1")
    monkeypatch.setenv("DONNA_VAULT_KEY", "test-key")

    import deploy.hf_app as app

    assert callable(app.predict)
    # Monkeypatch the bridge so we don't boot Ollama / cloud LLMs in CI.
    monkeypatch.setattr(
        "deploy.hf_app.run_text_command",
        lambda message, history=None: f"echo:{message}",
    )
    assert app.predict("hello") == "echo:hello"
