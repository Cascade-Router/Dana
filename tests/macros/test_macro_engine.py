"""Offline tests for Dānā desktop macro recorder / replay (no Florence, no clicks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dana.graph.nodes.execute_macro import (
    execute_macro_node,
    parse_macro_command,
)
from dana.macros.engine import MacroEngine, bbox_center, sanitize_macro_id
from dana.macros.schema import MacroSequence, MacroStep


def test_sanitize_macro_id() -> None:
    assert sanitize_macro_id("build_and_test") == "build_and_test"
    assert sanitize_macro_id("../evil/name!!") == "evil_name"
    assert sanitize_macro_id("") == "macro"


def test_bbox_center_translation() -> None:
    assert bbox_center((10.0, 20.0, 30.0, 40.0)) == (20, 30)
    assert bbox_center((100, 200, 200, 400)) == (150, 300)


def test_save_load_macro_json(tmp_path: Path) -> None:
    engine = MacroEngine(macros_dir=tmp_path)
    seq = MacroSequence(
        macro_id="build_and_test",
        description="Build then run tests",
        steps=[
            MacroStep(
                target_label="Run",
                action_type="click",
                action_value=None,
                visual_context_prompt="Run",
            ),
            MacroStep(
                target_label="Terminal",
                action_type="type_text",
                action_value="pytest -q",
                visual_context_prompt="Terminal",
            ),
        ],
    )
    path = engine.save_macro(seq)
    assert path.is_file()
    assert path.name == "build_and_test.json"
    loaded = engine.load_macro("build_and_test")
    assert loaded.macro_id == "build_and_test"
    assert len(loaded.steps) == 2
    assert loaded.steps[0].action_type == "click"
    assert loaded.steps[1].action_value == "pytest -q"


def test_record_step_uses_grounding_prompt(tmp_path: Path) -> None:
    def fake_ground(screenshot: Any, prompt: str) -> dict[str, Any]:
        assert prompt == "Save"
        return {
            "ok": True,
            "labels": ["Save"],
            "boxes_xyxy": [(10.0, 10.0, 50.0, 40.0)],
            "picked_xyxy": (10.0, 10.0, 50.0, 40.0),
            "matched_label": "Save",
            "image_wh": (100, 100),
        }

    engine = MacroEngine(macros_dir=tmp_path, grounding_fn=fake_ground)
    engine.begin_recording("save_flow", description="click save")
    step = engine.record_step("Save", "click", None, screenshot=object())
    assert step.visual_context_prompt == "Save"
    assert step.action_type == "click"
    seq = engine.finish_recording()
    assert seq.macro_id == "save_flow"
    assert len(seq.steps) == 1


def test_replay_macro_translates_bbox_to_coords(tmp_path: Path) -> None:
    clicks: list[tuple[int, int]] = []
    types: list[str] = []
    hotkeys: list[str] = []
    # Florence mock: prompt → known ROI in screen pixels.
    rois = {
        "Run": (100.0, 200.0, 140.0, 240.0),  # center 120, 220
        "Editor": (0.0, 0.0, 200.0, 100.0),  # center 100, 50
    }

    def fake_ground(screenshot: Any, prompt: str) -> dict[str, Any]:
        box = rois[prompt]
        return {
            "ok": True,
            "labels": [prompt],
            "boxes_xyxy": [box],
            "picked_xyxy": box,
            "matched_label": prompt,
            "image_wh": (800, 600),
        }

    engine = MacroEngine(
        macros_dir=tmp_path,
        grounding_fn=fake_ground,
        screenshot_fn=lambda: "fake-frame",
        click_fn=lambda x, y: clicks.append((x, y)),
        type_fn=lambda t: types.append(t),
        hotkey_fn=lambda h: hotkeys.append(h),
        double_click_fn=lambda x, y: clicks.append((x, y)),
    )
    seq = MacroSequence(
        macro_id="build_and_test",
        description="demo",
        steps=[
            MacroStep(
                target_label="Run",
                action_type="click",
                action_value=None,
                visual_context_prompt="Run",
            ),
            MacroStep(
                target_label="Editor",
                action_type="type_text",
                action_value="hello",
                visual_context_prompt="Editor",
            ),
            MacroStep(
                target_label="Editor",
                action_type="key_combination",
                action_value="ctrl+s",
                visual_context_prompt="Editor",
            ),
        ],
    )
    engine.save_macro(seq)
    result = engine.replay_macro("build_and_test")
    assert result["ok"] is True
    assert result["steps"] == 3
    # click Run → center of (100,200,140,240)
    assert clicks[0] == (120, 220)
    # type_text focuses Editor first → center (100, 50)
    assert clicks[1] == (100, 50)
    assert types == ["hello"]
    assert hotkeys == ["ctrl+s"]
    # Explicit bbox → action coordinate check via executed payload.
    assert result["executed"][0]["coords"] == [120, 220]
    assert result["executed"][0]["bbox"] == [100.0, 200.0, 140.0, 240.0]


def test_replay_fails_closed_when_roi_missing(tmp_path: Path) -> None:
    engine = MacroEngine(
        macros_dir=tmp_path,
        grounding_fn=lambda *_a, **_k: {
            "ok": False,
            "error": "no match",
            "picked_xyxy": None,
            "boxes_xyxy": [],
            "labels": [],
        },
        screenshot_fn=lambda: None,
        click_fn=lambda *_a, **_k: pytest.fail("must not click"),
    )
    engine.save_macro(
        MacroSequence(
            macro_id="missing",
            description="",
            steps=[
                MacroStep(
                    target_label="Nope",
                    action_type="click",
                    action_value=None,
                    visual_context_prompt="Nope",
                )
            ],
        )
    )
    result = engine.replay_macro("missing")
    assert result["ok"] is False
    assert "Nope" in result["error"]


def test_parse_macro_command() -> None:
    assert parse_macro_command("Run macro build_and_test") == "build_and_test"
    assert parse_macro_command("please execute macro login_flow now") == "login_flow"
    assert parse_macro_command("replay macro smoke-check") == "smoke-check"
    assert parse_macro_command("just chatting") is None


def test_execute_macro_node_with_injectable_engine(tmp_path: Path) -> None:
    clicks: list[tuple[int, int]] = []
    box = (10.0, 20.0, 30.0, 40.0)

    engine = MacroEngine(
        macros_dir=tmp_path,
        grounding_fn=lambda *_a, **_k: {
            "ok": True,
            "picked_xyxy": box,
            "boxes_xyxy": [box],
            "labels": ["Go"],
            "matched_label": "Go",
        },
        screenshot_fn=lambda: "shot",
        click_fn=lambda x, y: clicks.append((x, y)),
    )
    engine.save_macro(
        MacroSequence(
            macro_id="go",
            description="",
            steps=[
                MacroStep(
                    target_label="Go",
                    action_type="click",
                    action_value=None,
                    visual_context_prompt="Go",
                )
            ],
        )
    )
    patch = execute_macro_node(
        {"macro_id": "go", "messages": []},
        engine=engine,
    )
    assert patch["last_obs"].startswith("[macro] ok")
    assert clicks == [bbox_center(box)]
