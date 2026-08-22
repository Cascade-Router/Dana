"""Tests for dana.plugins.vision.image_analysis — the real "vision_tools"
capability domain (dana.core.react_dispatch's _VISION_TOOLS_TOOL_IDS):
analyze_workspace_image. The LLM/VLM provider is always mocked — these
tests never touch a real model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dana.plugins.os import file_system
from dana.plugins.vision import image_analysis


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    # exist_ok=True: tests/conftest.py's global _isolate_os_tools_sandbox
    # autouse fixture already creates this same tmp_path/agent_workspace
    # directory first — tolerate it already existing rather than raising.
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


class _FakeModelProvider:
    """Stands in for dana.core.model_provider.ModelProvider — records its
    constructor kwargs (to verify BYOK api_keys threading) and returns a
    canned description or raises per-candidate, exactly like the real
    complete_vision would on an unreachable/unsupported provider."""

    instances: list["_FakeModelProvider"] = []

    def __init__(self, description: str = "A bar chart with labeled axes.", fail_providers: tuple[str, ...] = (), **kwargs: Any) -> None:
        self.constructor_kwargs = kwargs
        self._description = description
        self._fail_providers = fail_providers
        self.calls: list[dict[str, Any]] = []
        _FakeModelProvider.instances.append(self)

    def complete_vision(self, prompt: str, image_b64: str, *, mime_type: str, provider: str) -> str:
        self.calls.append({"prompt": prompt, "mime_type": mime_type, "provider": provider})
        if provider in self._fail_providers:
            raise RuntimeError(f"{provider} unavailable")
        return self._description


def _mock_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    description: str = "A bar chart with labeled axes.",
    fail_providers: tuple[str, ...] = (),
) -> None:
    """Patches image_analysis.ModelProvider so the REAL constructor call
    inside analyze_workspace_image is what populates
    _FakeModelProvider.instances — no throwaway instance created here that
    would shadow it at index 0.
    """
    _FakeModelProvider.instances = []
    monkeypatch.setattr(
        image_analysis,
        "ModelProvider",
        lambda **ctor_kwargs: _FakeModelProvider(description=description, fail_providers=fail_providers, **ctor_kwargs),
    )


def _write_png(sandbox: Path, name: str) -> None:
    (sandbox / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


# --------------------------------------------------------------------------
# Path traversal rejection
# --------------------------------------------------------------------------


def test_rejects_parent_traversal(_sandbox: Path) -> None:
    result = image_analysis.analyze_workspace_image("../outside.png", "describe it")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"]


def test_rejects_absolute_path(_sandbox: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.png"
    result = image_analysis.analyze_workspace_image(str(outside), "describe it")
    assert result["ok"] is False
    assert "outside the sandbox" in result["error"] or "absolute paths are not allowed" in result["error"]


# --------------------------------------------------------------------------
# Non-image / missing file rejection
# --------------------------------------------------------------------------


def test_rejects_non_image_extension(_sandbox: Path) -> None:
    (_sandbox / "notes.txt").write_text("not an image")
    result = image_analysis.analyze_workspace_image("notes.txt", "describe it")
    assert result["ok"] is False
    assert "images are supported" in result["error"]


def test_rejects_unsupported_image_extension(_sandbox: Path) -> None:
    (_sandbox / "anim.gif").write_bytes(b"GIF89a")
    result = image_analysis.analyze_workspace_image("anim.gif", "describe it")
    assert result["ok"] is False
    assert "images are supported" in result["error"]


def test_missing_file_reports_clean_error_not_crash(_sandbox: Path) -> None:
    result = image_analysis.analyze_workspace_image("missing.png", "describe it")
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_directory_target_reports_clean_error(_sandbox: Path) -> None:
    (_sandbox / "adir.png").mkdir()
    result = image_analysis.analyze_workspace_image("adir.png", "describe it")
    assert result["ok"] is False
    assert "not a file" in result["error"]


# --------------------------------------------------------------------------
# Successful VLM handoff
# --------------------------------------------------------------------------


def test_successful_vlm_handoff_returns_description(_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_png(_sandbox, "chart.png")
    _mock_provider(monkeypatch, description="A line chart showing revenue over time, axes labeled.")

    result = image_analysis.analyze_workspace_image("chart.png", "Are the axes labeled?")

    assert result["ok"] is True
    assert result["path"] == "chart.png"
    assert result["query"] == "Are the axes labeled?"
    assert result["description"] == "A line chart showing revenue over time, axes labeled."


def test_empty_query_defaults_to_generic_description_request(_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_png(_sandbox, "chart.png")
    _mock_provider(monkeypatch)

    result = image_analysis.analyze_workspace_image("chart.png", "")

    assert result["ok"] is True
    assert result["query"] == "Describe what is shown in this image."


def test_correct_mime_type_is_passed_for_jpeg(_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (_sandbox / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    _mock_provider(monkeypatch)

    image_analysis.analyze_workspace_image("photo.jpg", "describe it")

    assert _FakeModelProvider.instances[0].calls[0]["mime_type"] == "image/jpeg"


def test_api_keys_are_threaded_into_model_provider_constructor(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "chart.png")
    _mock_provider(monkeypatch)

    image_analysis.analyze_workspace_image("chart.png", "describe it", api_keys={"openai": "sk-session-key"})

    assert _FakeModelProvider.instances[0].constructor_kwargs == {"api_keys": {"openai": "sk-session-key"}}


def test_all_providers_failing_reports_clean_error_with_attempts(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "chart.png")
    _mock_provider(monkeypatch, fail_providers=("ollama",))

    result = image_analysis.analyze_workspace_image("chart.png", "describe it")

    assert result["ok"] is False
    assert result["error"] == "all VLM providers failed"
    assert any("ollama" in attempt for attempt in result["attempts"])


def test_candidate_providers_defaults_to_ollama_only_when_cloud_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_analysis, "cloud_fallback_enabled", lambda: False)
    assert image_analysis._candidate_providers() == ["ollama"]


# --------------------------------------------------------------------------
# Registry / routing wiring
# --------------------------------------------------------------------------


def test_analyze_workspace_image_is_not_mutating() -> None:
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("analyze_workspace_image") is False


def test_analyze_workspace_image_registered_in_vision_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    assert "analyze_workspace_image" in rd.TOOL_HANDLERS
    assert "analyze_workspace_image" in rd._VISION_TOOLS_TOOL_IDS
    assert rd._CAPABILITY_TOOL_IDS["vision_tools"] == rd._VISION_TOOLS_TOOL_IDS


def test_dispatch_tool_call_threads_api_keys_end_to_end(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one end-to-end proof that dana.core.react_dispatch.dispatch_tool_call
    actually threads api_keys through to this tool's handler (not just the
    module function tested directly above)."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _write_png(_sandbox, "chart.png")
    _mock_provider(monkeypatch)

    call = ToolCall(tool_id="analyze_workspace_image", arguments={"file_path": "chart.png", "query": "describe it"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, api_keys={"openai": "sk-dispatch-key"})

    assert result.ok is True
    assert _FakeModelProvider.instances[0].constructor_kwargs == {"api_keys": {"openai": "sk-dispatch-key"}}


def test_dispatch_tool_call_traversal_is_digested_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(
        tool_id="analyze_workspace_image", arguments={"file_path": "../escape.png", "query": "describe it"}
    )
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)

    assert result.ok is False
    assert "outside the sandbox" in result.payload.get("raw_error", "")


def test_other_tools_are_unaffected_by_api_keys_threading(monkeypatch: pytest.MonkeyPatch) -> None:
    """_TOOLS_NEEDING_API_KEYS must be a narrow allowlist — an ordinary
    3-argument handler (system_state) must keep working with no api_keys
    kwarg passed to it at all."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="system_state", arguments={})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, api_keys={"openai": "sk-unused"})
    assert result.ok is True
