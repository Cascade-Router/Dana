"""Tests for dana.plugins.vision.image_analysis — the real "vision_tools"
capability domain (dana.core.react_dispatch's _VISION_TOOLS_TOOL_IDS):
analyze_workspace_image and analyze_reference_design. The LLM/VLM provider
is always mocked — these tests never touch a real model.
"""

from __future__ import annotations

import json
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


# A clean, unambiguous 10x20x30 box, spread across the standard 3-view
# triple with every number distinct -- analyze_reference_design's bounding
# box is now computed FROM this data (_compute_deterministic_bbox), not
# from the mocked VLM response, so any test that expects ok: True needs a
# real, non-degenerate OCR mock reaching it, same as production would.
#   FRONT ∩ TOP   = {10.0} -> x=10.0
#   FRONT ∩ RIGHT = {30.0} -> z=30.0
#   TOP   ∩ RIGHT = {20.0} -> y=20.0
_DEFAULT_OCR_MOCK = {"FRONT": ["10.0", "30.0"], "TOP": ["10.0", "20.0"], "RIGHT": ["20.0", "30.0"]}
_DEFAULT_OCR_BBOX = {"x": 10.0, "y": 20.0, "z": 30.0}


@pytest.fixture(autouse=True)
def _mock_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let analyze_reference_design's tests hit a real
    easyocr.Reader() -- constructing one loads/downloads real detection +
    recognition models (confirmed live: slow, and network-dependent on a
    fresh machine/CI runner), which every VLM provider call in this file is
    already mocked specifically to avoid.

    Defaults to _DEFAULT_OCR_MOCK (real per-view dimensions) rather than {}
    -- the bounding box now comes entirely from this data, so a test that
    doesn't care about OCR specifically still needs SOME valid triple to
    reach ok: True, same as a real drawing with legible dimensions would.
    Tests that specifically exercise OCR behavior (threading, fallback,
    empty-view handling) override this again via their own
    monkeypatch.setattr(...).
    """
    monkeypatch.setattr(image_analysis, "extract_blueprint_dimensions", lambda file_paths: dict(_DEFAULT_OCR_MOCK))


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


class _FakeSequentialProvider:
    """Like _FakeModelProvider, but returns one canned response per call IN
    ORDER rather than the same description every time. analyze_reference_design
    makes two DIFFERENT complete_vision calls (pass 1's primitives prompt,
    then pass 2's relationships/joints prompt) within a single invocation,
    each expecting its own distinct JSON schema back — a single fixed
    description can't stand in for both.
    """

    instances: list["_FakeSequentialProvider"] = []

    def __init__(self, responses: list[str], **kwargs: Any) -> None:
        self.constructor_kwargs = kwargs
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        _FakeSequentialProvider.instances.append(self)

    def complete_vision(self, prompt: str, image_b64: str, *, mime_type: str, provider: str) -> str:
        self.calls.append({"prompt": prompt, "mime_type": mime_type, "provider": provider})
        if not self._responses:
            raise RuntimeError("_FakeSequentialProvider ran out of canned responses")
        return self._responses.pop(0)


def _mock_sequential_provider(monkeypatch: pytest.MonkeyPatch, *, responses: list[str]) -> None:
    """Patches image_analysis.ModelProvider (and forces cloud fallback off,
    so _candidate_providers() is deterministically just ["ollama"] and each
    pass makes exactly one complete_vision call) the same way _mock_provider
    does for the single-response case above.
    """
    _FakeSequentialProvider.instances = []
    monkeypatch.setattr(image_analysis, "cloud_fallback_enabled", lambda: False)
    monkeypatch.setattr(
        image_analysis,
        "ModelProvider",
        lambda **ctor_kwargs: _FakeSequentialProvider(responses=list(responses), **ctor_kwargs),
    )


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
# analyze_reference_design — two-pass CSG extraction + numerical-integrity gate
# --------------------------------------------------------------------------

_PASS_1_OK = json.dumps(
    {"primitives": [{"id": "p1", "type": "box", "dimensions": {"length": 100.0, "width": 50.0, "height": 25.0}}]}
)
_PASS_2_OK = json.dumps({"relationships": [], "joints": []})


def test_analyze_reference_design_two_pass_success(_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is True
    # Bounding box comes from OCR (_DEFAULT_OCR_MOCK via _mock_ocr), not from
    # the mocked pass 1 response -- see _compute_deterministic_bbox.
    assert result["blueprint"]["bounding_box"] == _DEFAULT_OCR_BBOX
    assert len(result["blueprint"]["primitives"]) == 1
    assert result["blueprint"]["relationships"] == []
    assert result["blueprint"]["joints"] == []
    # Two distinct calls (one per pass), same single-image payload both times.
    assert len(_FakeSequentialProvider.instances[0].calls) == 2


def test_analyze_reference_design_degenerate_bounding_box_rejected(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounding box is computed from OCR, not asked of the VLM at all
    (see _CSG_PASS1_PROMPT's own comment on why) -- so what makes it
    degenerate now is OCR finding nothing usable, not a hallucinated
    pass-1 field. Overrides the autouse _mock_ocr default specifically to
    exercise that path.
    """
    monkeypatch.setattr(image_analysis, "extract_blueprint_dimensions", lambda file_paths: {})
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "Degenerate bounding box" in result["error"]
    # The (rejected) blueprint is still surfaced for debugging, not swallowed.
    assert result["blueprint"]["bounding_box"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_analyze_reference_design_degenerate_primitive_dimensions_rejected(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pass_1_degenerate = json.dumps(
        {"primitives": [{"id": "p1", "type": "box", "dimensions": {"length": 0.0, "width": 0.0, "height": 0.0}}]}
    )
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[pass_1_degenerate, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "Degenerate primitive dimensions" in result["error"]
    assert "p1" in result["error"]


def test_analyze_reference_design_no_primitives_rejected(_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pass_1_empty = json.dumps({"primitives": []})
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[pass_1_empty, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert result["error"] == "No primitives extracted"


def test_analyze_reference_design_dangling_relationship_reference_rejected(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pass_2_dangling = json.dumps(
        {"relationships": [{"parent_id": "p1", "child_id": "p99", "attachment_type": "face_to_face"}], "joints": []}
    )
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, pass_2_dangling])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "Dangling relationship reference" in result["error"]


def test_analyze_reference_design_dangling_joint_reference_rejected(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pass_2_dangling = json.dumps(
        {"relationships": [], "joints": [{"parent_id": "p1", "child_id": "p99", "type": "fixed"}]}
    )
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, pass_2_dangling])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "Dangling joint reference" in result["error"]


def test_analyze_reference_design_pass_1_json_decode_failure(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=["this is not json"])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "pass 1" in result["error"]
    assert any("JSON decode failed" in attempt for attempt in result["attempts"])


def test_analyze_reference_design_pass_2_json_decode_failure(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, "still not json"])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "pass 2" in result["error"]
    assert result["primitives"] == json.loads(_PASS_1_OK)


def test_analyze_reference_design_strips_markdown_code_fence(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fenced_pass_1 = "```json\n" + _PASS_1_OK + "\n```"
    fenced_pass_2 = "```\n" + _PASS_2_OK + "\n```"
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[fenced_pass_1, fenced_pass_2])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is True
    assert result["blueprint"]["bounding_box"] == _DEFAULT_OCR_BBOX


def test_analyze_reference_design_threads_ocr_dimensions_into_pass_1_prompt(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "front.png")
    monkeypatch.setattr(
        image_analysis, "extract_blueprint_dimensions", lambda file_paths: {"FRONT": ["60.0", "50.0"]}
    )
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is True
    pass_1_prompt = _FakeSequentialProvider.instances[0].calls[0]["prompt"]
    assert "HARD DIMENSIONAL CONSTRAINTS" in pass_1_prompt
    assert "FRONT VIEW" in pass_1_prompt
    assert "60.0" in pass_1_prompt
    assert "50.0" in pass_1_prompt


def test_analyze_reference_design_falls_back_when_no_ocr_dimensions_found(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no OCR data at all, pass 1's prompt correctly falls back to the
    original guidance text -- but the run itself now correctly ends in
    ok: False too, not True: the bounding box is computed FROM that same
    OCR data (_compute_deterministic_bbox), so no OCR data unavoidably
    means a degenerate {0,0,0} box. This is a deliberate consequence of no
    longer trusting the VLM's own bounding-box guess at all, not a bug --
    a drawing with no legible printed dimensions genuinely can't produce a
    numerically-grounded blueprint under this design.
    """
    monkeypatch.setattr(image_analysis, "extract_blueprint_dimensions", lambda file_paths: {})
    _write_png(_sandbox, "front.png")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png"])

    assert result["ok"] is False
    assert "Degenerate bounding box" in result["error"]
    pass_1_prompt = _FakeSequentialProvider.instances[0].calls[0]["prompt"]
    assert "HARD DIMENSIONAL CONSTRAINTS" not in pass_1_prompt
    assert "do not attempt to guess exact millimeter tolerances" in pass_1_prompt.lower()


def test_analyze_reference_design_ignores_view_with_no_dimensions_found(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A view whose OCR list came back empty (e.g. no legible text on that
    one file) must not produce an empty "- VIEW dimensions: " line -- that
    would tell the VLM a view has zero-length dimensions, which is exactly
    the kind of degenerate-looking hint _validate_numerical_integrity
    exists to catch further downstream, not something worth injecting into
    the prompt in the first place.
    """
    _write_png(_sandbox, "front.png")
    monkeypatch.setattr(
        image_analysis,
        "extract_blueprint_dimensions",
        lambda file_paths: {"FRONT": ["60.0"], "TOP": []},
    )
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    image_analysis.analyze_reference_design(["front.png"])

    pass_1_prompt = _FakeSequentialProvider.instances[0].calls[0]["prompt"]
    assert "FRONT VIEW" in pass_1_prompt
    assert "TOP VIEW" not in pass_1_prompt


def test_analyze_reference_design_multi_view_uses_stitched_composite(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 file_paths must go through _stitch_images (real Pillow decoding is
    exercised separately/not here — this only proves analyze_reference_design
    dispatches to it and reuses its single composite for BOTH passes,
    instead of passing 3 separate image_url blocks the way the pre-fix code
    did (see this module's own _stitch_images docstring for why that broke).
    """
    _write_png(_sandbox, "front.png")
    _write_png(_sandbox, "top.png")
    _write_png(_sandbox, "right.png")
    monkeypatch.setattr(image_analysis, "_stitch_images", lambda paths: "STITCHED_COMPOSITE_B64")
    _mock_sequential_provider(monkeypatch, responses=[_PASS_1_OK, _PASS_2_OK])

    result = image_analysis.analyze_reference_design(["front.png", "top.png", "right.png"])

    assert result["ok"] is True
    calls = _FakeSequentialProvider.instances[0].calls
    assert len(calls) == 2
    assert all(c["mime_type"] == "image/png" for c in calls)


def test_analyze_reference_design_rejects_missing_file() -> None:
    result = image_analysis.analyze_reference_design(["missing.png"])
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_analyze_reference_design_rejects_empty_file_paths() -> None:
    result = image_analysis.analyze_reference_design([])
    assert result["ok"] is False
    assert "at least one file_path is required" in result["error"]


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
