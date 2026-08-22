"""Tests for dana.plugins.os.desktop_vision — Desktop Omni-Vision's
analyze_desktop_screen, the real "os_tools" capability domain tool that
captures the user's ACTUAL primary monitor (dana.core.react_dispatch's
_OS_TOOLS_TOOL_IDS). Both the screen capture (mss) and the VLM call
(ModelProvider) are always mocked here — these tests must never take a
real screenshot or make a real network call.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest

from dana.plugins.os import desktop_vision


class _FakeScreenShot:
    """Stands in for mss's ScreenShot — .size + raw .bgra bytes are the
    only two attributes desktop_vision._capture_primary_monitor_jpeg_b64
    actually reads off it."""

    def __init__(self, width: int = 8, height: int = 6) -> None:
        self.size = (width, height)
        self.bgra = bytes((10, 20, 30, 255)) * (width * height)


class _FakeSct:
    """Stands in for mss.mss()'s context-manager instance."""

    def __init__(self, num_monitors: int = 2, width: int = 8, height: int = 6) -> None:
        # index 0 is mss's own "combined virtual desktop" pseudo-monitor;
        # index 1+ are real monitors — matches production's own
        # `monitors[1] if len(monitors) > 1 else monitors[0]` selection.
        self.monitors = [{"left": 0, "top": 0, "width": width, "height": height} for _ in range(num_monitors)]
        self._width = width
        self._height = height

    def grab(self, monitor: dict[str, int]) -> _FakeScreenShot:
        return _FakeScreenShot(self._width, self._height)

    def __enter__(self) -> "_FakeSct":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _mock_mss(monkeypatch: pytest.MonkeyPatch, *, num_monitors: int = 2, width: int = 8, height: int = 6) -> None:
    monkeypatch.setattr("mss.mss", lambda: _FakeSct(num_monitors, width, height))


def _mock_mss_capture_failure(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def _boom() -> None:
        raise error

    monkeypatch.setattr("mss.mss", _boom)


class _FakeModelProvider:
    """Stands in for dana.core.model_provider.ModelProvider — records its
    constructor kwargs (to verify BYOK api_keys threading) and every
    complete_vision call (including the actual image_b64 it was given, so
    tests can verify the downscale/JPEG behavior), and returns a canned
    description or raises per-candidate."""

    instances: list["_FakeModelProvider"] = []

    def __init__(
        self, description: str = "A stack trace showing a KeyError.", fail_providers: tuple[str, ...] = (), **kwargs: Any
    ) -> None:
        self.constructor_kwargs = kwargs
        self._description = description
        self._fail_providers = fail_providers
        self.calls: list[dict[str, Any]] = []
        _FakeModelProvider.instances.append(self)

    def complete_vision(self, prompt: str, image_b64: str, *, mime_type: str, provider: str) -> str:
        self.calls.append({"prompt": prompt, "image_b64": image_b64, "mime_type": mime_type, "provider": provider})
        if provider in self._fail_providers:
            raise RuntimeError(f"{provider} unavailable")
        return self._description


def _mock_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    description: str = "A stack trace showing a KeyError.",
    fail_providers: tuple[str, ...] = (),
) -> None:
    """Patches desktop_vision.ModelProvider so the REAL constructor call
    inside analyze_desktop_screen is what populates
    _FakeModelProvider.instances — no throwaway instance shadowing index 0.
    """
    _FakeModelProvider.instances = []
    monkeypatch.setattr(
        desktop_vision,
        "ModelProvider",
        lambda **ctor_kwargs: _FakeModelProvider(description=description, fail_providers=fail_providers, **ctor_kwargs),
    )


# --------------------------------------------------------------------------
# Capture + VLM handoff
# --------------------------------------------------------------------------


def test_successful_capture_and_analysis_returns_description(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch, description="An IDE showing a KeyError on line 42.")

    result = desktop_vision.analyze_desktop_screen("What error is shown?")

    assert result["ok"] is True
    assert result["query"] == "What error is shown?"
    assert result["description"] == "An IDE showing a KeyError on line 42."


def test_empty_query_defaults_to_generic_description_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch)

    result = desktop_vision.analyze_desktop_screen("")

    assert result["ok"] is True
    assert result["query"] == "Describe what is shown on the screen."


def test_captured_image_is_sent_as_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch)

    desktop_vision.analyze_desktop_screen("describe it")

    assert _FakeModelProvider.instances[0].calls[0]["mime_type"] == "image/jpeg"


def test_capture_wider_than_max_is_downscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture wider than _MAX_WIDTH must be shrunk before it's ever
    base64-encoded/sent — verified here by actually decoding the JPEG
    bytes handed to complete_vision and checking its real pixel width."""
    from PIL import Image

    _mock_mss(monkeypatch, width=desktop_vision._MAX_WIDTH + 400, height=10)
    _mock_provider(monkeypatch)

    desktop_vision.analyze_desktop_screen("describe it")

    sent_b64 = _FakeModelProvider.instances[0].calls[0]["image_b64"]
    decoded = Image.open(io.BytesIO(base64.b64decode(sent_b64)))
    assert decoded.width <= desktop_vision._MAX_WIDTH
    assert decoded.format == "JPEG"


def test_capture_narrower_than_max_is_not_upscaled(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    _mock_mss(monkeypatch, width=8, height=6)
    _mock_provider(monkeypatch)

    desktop_vision.analyze_desktop_screen("describe it")

    sent_b64 = _FakeModelProvider.instances[0].calls[0]["image_b64"]
    decoded = Image.open(io.BytesIO(base64.b64decode(sent_b64)))
    assert decoded.width == 8


def test_api_keys_are_threaded_into_model_provider_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch)

    desktop_vision.analyze_desktop_screen("describe it", api_keys={"openai": "sk-session-key"})

    assert _FakeModelProvider.instances[0].constructor_kwargs == {"api_keys": {"openai": "sk-session-key"}}


def test_all_providers_failing_reports_clean_error_with_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch, fail_providers=("ollama",))

    result = desktop_vision.analyze_desktop_screen("describe it")

    assert result["ok"] is False
    assert result["error"] == "all VLM providers failed"
    assert any("ollama" in attempt for attempt in result["attempts"])


def test_candidate_providers_defaults_to_ollama_only_when_cloud_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(desktop_vision, "cloud_fallback_enabled", lambda: False)
    assert desktop_vision._candidate_providers() == ["ollama"]


# --------------------------------------------------------------------------
# Capture-failure resilience
# --------------------------------------------------------------------------


def test_capture_failure_reports_clean_error_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss_capture_failure(monkeypatch, RuntimeError("no display attached"))

    result = desktop_vision.analyze_desktop_screen("describe it")

    assert result["ok"] is False
    assert "screen capture failed" in result["error"]
    assert "no display attached" in result["error"]


def test_capture_failure_never_calls_the_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mss_capture_failure(monkeypatch, RuntimeError("no display attached"))
    _mock_provider(monkeypatch)

    desktop_vision.analyze_desktop_screen("describe it")

    assert _FakeModelProvider.instances == []


# --------------------------------------------------------------------------
# Registry / routing wiring — the "Crucial Privacy Gate" requirement
# --------------------------------------------------------------------------


def test_analyze_desktop_screen_is_mutating() -> None:
    """The key HITL classification check: unlike analyze_workspace_image
    (a sandboxed-artifact read, never gated), analyze_desktop_screen
    captures the user's REAL desktop and must ALWAYS require explicit
    human approval before it ever dispatches."""
    import dana.core.react_dispatch as rd

    assert rd.is_mutating_tool("analyze_desktop_screen") is True


def test_analyze_desktop_screen_registered_in_os_tools_domain() -> None:
    import dana.core.react_dispatch as rd

    assert "analyze_desktop_screen" in rd.TOOL_HANDLERS
    assert "analyze_desktop_screen" in rd._OS_TOOLS_TOOL_IDS
    assert rd._CAPABILITY_TOOL_IDS["os_tools"] == rd._OS_TOOLS_TOOL_IDS


def test_analyze_desktop_screen_needs_api_keys() -> None:
    import dana.core.react_dispatch as rd

    assert "analyze_desktop_screen" in rd._TOOLS_NEEDING_API_KEYS


def test_hitl_description_names_the_desktop_and_the_query() -> None:
    """describe_tool_call is what the user actually reads on the HITL
    approval card — must make clear this captures the REAL desktop, not
    a sandboxed file, and show what will be asked about it."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    call = ToolCall(tool_id="analyze_desktop_screen", arguments={"query": "What error is on screen?"})
    description = rd.describe_tool_call(call)

    assert "desktop" in description.lower() or "screen" in description.lower()
    assert "What error is on screen?" in description


def test_dispatch_tool_call_threads_api_keys_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one end-to-end proof that dana.core.react_dispatch.dispatch_tool_call
    actually threads api_keys through to this tool's handler (not just the
    module function tested directly above) — mirrors
    tests/plugins/vision/test_image_analysis.py's own such test."""
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _mock_mss(monkeypatch)
    _mock_provider(monkeypatch)

    call = ToolCall(tool_id="analyze_desktop_screen", arguments={"query": "describe it"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None, api_keys={"openai": "sk-dispatch-key"})

    assert result.ok is True
    assert _FakeModelProvider.instances[0].constructor_kwargs == {"api_keys": {"openai": "sk-dispatch-key"}}


def test_dispatch_tool_call_capture_failure_is_digested_not_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    import dana.core.react_dispatch as rd
    from dana.tools.schema import ToolCall

    _mock_mss_capture_failure(monkeypatch, RuntimeError("no display attached"))

    call = ToolCall(tool_id="analyze_desktop_screen", arguments={"query": "describe it"})
    result = rd.dispatch_tool_call(call, engine=None, control_plane=None)

    assert result.ok is False
    assert "raw_error" in result.payload
