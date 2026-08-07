"""click_ui_element / type_text_in_element / scroll_screen / drag_ui_element — tool wiring.

Every hardware-adjacent test forces ``DONNA_OS_DRY_RUN=1`` so the full
validation/rate-limit/failsafe path runs but no real SendInput call fires,
and monkeypatches the vision lookup at its import sites so no real screen
capture, Florence model, or UIA backend is touched. A couple of tests turn
dry-run back off to prove the full click-then-type plumbing fires, but even
those mock the os_control SendInput entry points so no real hardware input
happens.
"""
from __future__ import annotations

import dana.tools.drag_actuator as drag_actuator
import dana.tools.keyboard_actuator as keyboard_actuator
import dana.tools.mouse_actuator as mouse_actuator
import dana.tools.scroll_actuator as scroll_actuator
import dana.tools.vision as vision_tool
import pytest


@pytest.fixture(autouse=True)
def _dry_run_and_reset_rate_limiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    mouse_actuator._last_actuation_ts = 0.0
    keyboard_actuator._last_actuation_ts = 0.0
    scroll_actuator._last_actuation_ts = 0.0
    drag_actuator._last_actuation_ts = 0.0
    yield
    mouse_actuator._last_actuation_ts = 0.0
    keyboard_actuator._last_actuation_ts = 0.0
    scroll_actuator._last_actuation_ts = 0.0
    drag_actuator._last_actuation_ts = 0.0


def _patch_vision_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bbox_1000,
    screen_size=(2000, 2000),
    image=object(),
):
    """Stub the three inline imports click_ui_element makes: screen capture,
    hybrid grounding, and screen-size lookup."""
    import dana.graph.nodes.vision as vision_node
    import dana.tools.os_control as os_control
    import dana.vision_tools as vision_tools

    monkeypatch.setattr(vision_tools, "capture_screen_frame", lambda: image)
    monkeypatch.setattr(
        vision_node, "locate_ui_element", lambda img, label: bbox_1000
    )
    monkeypatch.setattr(os_control, "get_screen_size", lambda: screen_size)


def _patch_drag_vision_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bboxes_by_label: dict,
    screen_size=(2000, 2000),
    image=object(),
):
    """Stub the drag_ui_element inline imports: screen capture, hybrid
    grounding (dispatched per-label so source/destination resolve to
    different boxes), and screen-size lookup."""
    import dana.graph.nodes.vision as vision_node
    import dana.tools.os_control as os_control
    import dana.vision_tools as vision_tools

    monkeypatch.setattr(vision_tools, "capture_screen_frame", lambda: image)
    monkeypatch.setattr(
        vision_node, "locate_ui_element", lambda img, label: bboxes_by_label.get(label)
    )
    monkeypatch.setattr(os_control, "get_screen_size", lambda: screen_size)


def test_click_ui_element_success_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Centered box in Florence [0,1000] space; screen is 2000x2000 -> centroid
    # (550, 550) scales to (1100, 1100) regardless of default inset padding.
    _patch_vision_pipeline(monkeypatch, bbox_1000=[500, 500, 600, 600])

    out = vision_tool.click_ui_element("Submit button")

    assert out == "SUCCESS: Clicked 'Submit button' at (1100, 1100) (dry_run)"


def test_click_ui_element_element_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=None)

    out = vision_tool.click_ui_element("Nonexistent Widget")

    assert out == "ERROR: Could not locate 'Nonexistent Widget' on screen"


def test_click_ui_element_empty_description_short_circuits() -> None:
    out = vision_tool.click_ui_element("   ")
    assert out.startswith("ERROR: click_ui_element requires a non-empty")


def test_click_ui_element_screen_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=[0, 0, 100, 100], image=None)

    out = vision_tool.click_ui_element("Submit button")

    assert out == "ERROR: click_ui_element could not capture a screen frame"


def test_click_ui_element_failsafe_blocks_out_of_bounds_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A malformed grounding result (values beyond [0,1000]) scales past the
    # live screen bounds; the actuator's failsafe must abort with no motion.
    _patch_vision_pipeline(
        monkeypatch,
        bbox_1000=[1500, 1500, 1600, 1600],
        screen_size=(2000, 2000),
    )

    out = vision_tool.click_ui_element("Submit button")

    assert out.startswith("ERROR: click_ui_element failed to click 'Submit button'")
    assert "failsafe" in out


def test_click_ui_element_rate_limits_rapid_successive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=[500, 500, 600, 600])

    first = vision_tool.click_ui_element("Submit button")
    second = vision_tool.click_ui_element("Submit button")

    assert first.startswith("SUCCESS:")
    assert second.startswith("ERROR: click_ui_element failed to click")
    assert "rate_limited" in second


def test_click_ui_element_registered_in_tool_registry_with_required_param() -> None:
    from dana.tools.registry import get_tool_registry

    entry = get_tool_registry(reload=True).get("click_ui_element")
    assert entry is not None
    param_names = {(p.name, p.required) for p in entry.spec.parameters}
    assert ("target_description", True) in param_names


def test_default_args_for_forced_click_ui_element() -> None:
    from dana.agentic_react_graph import _default_args_for_forced_tool

    args = _default_args_for_forced_tool("click_ui_element", "click the submit button")
    assert args == {"target_description": "click the submit button"}


# --------------------------------------------------------------------------
# type_text_in_element
# --------------------------------------------------------------------------


def test_type_text_in_element_success_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=[500, 500, 600, 600])

    out = vision_tool.type_text_in_element("Search bar", "hello world")

    assert out == "SUCCESS: Clicked 'Search bar' and typed text. (dry_run)"


def test_type_text_in_element_element_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=None)

    out = vision_tool.type_text_in_element("Nonexistent Widget", "hello")

    assert out == "ERROR: Could not locate 'Nonexistent Widget' on screen"


def test_type_text_in_element_empty_target_description_short_circuits() -> None:
    out = vision_tool.type_text_in_element("   ", "hello")
    assert out.startswith("ERROR: type_text_in_element requires a non-empty target")


def test_type_text_in_element_empty_text_short_circuits() -> None:
    out = vision_tool.type_text_in_element("Search bar", "   ")
    assert out == "ERROR: type_text_in_element requires non-empty text"


def test_type_text_in_element_click_failure_never_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Malformed grounding box scales past the live screen bounds -> the
    # mouse actuator's failsafe aborts before any typing is attempted.
    _patch_vision_pipeline(
        monkeypatch,
        bbox_1000=[1500, 1500, 1600, 1600],
        screen_size=(2000, 2000),
    )

    out = vision_tool.type_text_in_element("Search bar", "hello world")

    assert out.startswith("ERROR: type_text_in_element failed to focus 'Search bar'")
    assert "failsafe" in out


def test_type_text_in_element_rate_limits_rapid_successive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vision_pipeline(monkeypatch, bbox_1000=[500, 500, 600, 600])

    first = vision_tool.type_text_in_element("Search bar", "hello")
    second = vision_tool.type_text_in_element("Search bar", "world")

    assert first.startswith("SUCCESS:")
    assert second.startswith("ERROR: type_text_in_element failed to focus")
    assert "rate_limited" in second


def test_type_text_in_element_registered_in_tool_registry_with_required_params() -> None:
    from dana.tools.registry import get_tool_registry

    entry = get_tool_registry(reload=True).get("type_text_in_element")
    assert entry is not None
    param_names = {(p.name, p.required) for p in entry.spec.parameters}
    assert ("target_description", True) in param_names
    assert ("text", True) in param_names


def test_type_text_in_element_triggers_click_and_type_when_not_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end with dry-run OFF: mocks the os_control SendInput entry
    points (never real hardware) and asserts both the mouse actuator and the
    keyboard actuator actually fire, in order, with the right data."""
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "0")
    _patch_vision_pipeline(monkeypatch, bbox_1000=[500, 500, 600, 600])

    import dana.middleware.kill_switch as kill_switch
    import dana.tools.os_control as os_control

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: False)

    move_calls: list[tuple[int, int]] = []
    click_calls: list[bool] = []
    type_calls: list[str] = []

    monkeypatch.setattr(
        os_control, "move_cursor_absolute", lambda x, y: move_calls.append((x, y))
    )
    monkeypatch.setattr(
        os_control, "click_left_sendinput", lambda: click_calls.append(True)
    )
    monkeypatch.setattr(
        os_control,
        "type_text_sendinput",
        lambda text: (type_calls.append(text) or {"ok": True, "chars_typed": len(text)}),
    )
    monkeypatch.setattr(vision_tool.time, "sleep", lambda _s: None)

    out = vision_tool.type_text_in_element("Search bar", "hello world")

    assert out == "SUCCESS: Clicked 'Search bar' and typed text."
    assert move_calls == [(1100, 1100)]
    assert click_calls == [True]
    assert type_calls == ["hello world"]


# --------------------------------------------------------------------------
# scroll_screen
# --------------------------------------------------------------------------


def test_scroll_screen_success_dry_run() -> None:
    out = vision_tool.scroll_screen("down")
    assert out == "SUCCESS: Scrolled down by a medium amount. (dry_run)"


def test_scroll_screen_unknown_direction() -> None:
    out = vision_tool.scroll_screen("sideways")
    assert out.startswith("ERROR: scroll_screen failed")
    assert "unknown direction" in out


def test_scroll_screen_unknown_amount() -> None:
    out = vision_tool.scroll_screen("down", "gigantic")
    assert out == (
        "ERROR: scroll_screen unknown amount 'gigantic'; expected one of "
        "['large', 'medium', 'small']"
    )


def test_scroll_screen_rate_limits_rapid_successive_calls() -> None:
    first = vision_tool.scroll_screen("down")
    second = vision_tool.scroll_screen("up")
    assert first.startswith("SUCCESS:")
    assert second.startswith("ERROR: scroll_screen failed")
    assert "rate_limited" in second


def test_scroll_screen_registered_in_tool_registry_with_enum_params() -> None:
    from dana.tools.registry import get_tool_registry

    entry = get_tool_registry(reload=True).get("scroll_screen")
    assert entry is not None
    params = {p.name: p for p in entry.spec.parameters}
    assert params["direction"].required is True
    assert set(params["direction"].enum) == {"up", "down", "left", "right"}
    assert params["amount"].required is False
    assert set(params["amount"].enum) == {"small", "medium", "large"}


def test_default_args_for_forced_scroll_screen() -> None:
    from dana.agentic_react_graph import _default_args_for_forced_tool

    args = _default_args_for_forced_tool("scroll_screen", "please scroll up a bit")
    assert args == {"direction": "up", "amount": "medium"}

    fallback = _default_args_for_forced_tool("scroll_screen", "reveal more content")
    assert fallback == {"direction": "down", "amount": "medium"}


@pytest.mark.parametrize(
    "amount,expected_ticks",
    [("small", 2), ("medium", 5), ("large", 12)],
)
def test_scroll_screen_translates_semantic_amount_to_backend_tick_count(
    monkeypatch: pytest.MonkeyPatch,
    amount: str,
    expected_ticks: int,
) -> None:
    """Dry-run OFF: mocks the os_control SendInput entry point (never real
    hardware) and counts exactly how many wheel ticks each semantic amount
    produces."""
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "0")

    import dana.middleware.kill_switch as kill_switch
    import dana.tools.os_control as os_control

    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: False)

    scroll_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        os_control,
        "scroll_wheel_sendinput",
        lambda *, dx=0, dy=0: scroll_calls.append((dx, dy)),
    )
    monkeypatch.setattr(scroll_actuator.random, "uniform", lambda a, b: 0.0)

    out = vision_tool.scroll_screen("down", amount)

    assert out == f"SUCCESS: Scrolled down by a {amount} amount."
    assert len(scroll_calls) == expected_ticks
    assert all(dx == 0 and dy < 0 for dx, dy in scroll_calls)


# --------------------------------------------------------------------------
# drag_ui_element
# --------------------------------------------------------------------------


def test_drag_ui_element_success_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Source centroid (550,550) and destination centroid (850,850) in Florence
    # [0,1000] space; screen is 2000x2000 -> scaled to (1100,1100)/(1700,1700).
    _patch_drag_vision_pipeline(
        monkeypatch,
        bboxes_by_label={
            "File icon": [500, 500, 600, 600],
            "Archive folder": [800, 800, 900, 900],
        },
    )

    out = vision_tool.drag_ui_element("File icon", "Archive folder")

    assert out == (
        "SUCCESS: Dragged 'File icon' from (1100, 1100) to "
        "'Archive folder' at (1700, 1700) (dry_run)"
    )


def test_drag_ui_element_source_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_drag_vision_pipeline(
        monkeypatch, bboxes_by_label={"Archive folder": [800, 800, 900, 900]}
    )

    out = vision_tool.drag_ui_element("Nonexistent Icon", "Archive folder")

    assert out == "ERROR: Could not locate 'Nonexistent Icon' on screen"


def test_drag_ui_element_destination_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_drag_vision_pipeline(
        monkeypatch, bboxes_by_label={"File icon": [500, 500, 600, 600]}
    )

    out = vision_tool.drag_ui_element("File icon", "Nonexistent Folder")

    assert out == "ERROR: Could not locate 'Nonexistent Folder' on screen"


def test_drag_ui_element_empty_source_description_short_circuits() -> None:
    out = vision_tool.drag_ui_element("   ", "Archive folder")
    assert out.startswith("ERROR: drag_ui_element requires a non-empty source")


def test_drag_ui_element_empty_destination_description_short_circuits() -> None:
    out = vision_tool.drag_ui_element("File icon", "   ")
    assert out.startswith("ERROR: drag_ui_element requires a non-empty destination")


def test_drag_ui_element_failsafe_blocks_out_of_bounds_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A malformed grounding result for the destination scales past the live
    # screen bounds; the actuator's failsafe must abort with no motion.
    _patch_drag_vision_pipeline(
        monkeypatch,
        bboxes_by_label={
            "File icon": [500, 500, 600, 600],
            "Archive folder": [1500, 1500, 1600, 1600],
        },
        screen_size=(2000, 2000),
    )

    out = vision_tool.drag_ui_element("File icon", "Archive folder")

    assert out.startswith(
        "ERROR: drag_ui_element failed to drag 'File icon' to 'Archive folder'"
    )
    assert "failsafe" in out


def test_drag_ui_element_rate_limits_rapid_successive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_drag_vision_pipeline(
        monkeypatch,
        bboxes_by_label={
            "File icon": [500, 500, 600, 600],
            "Archive folder": [800, 800, 900, 900],
        },
    )

    first = vision_tool.drag_ui_element("File icon", "Archive folder")
    second = vision_tool.drag_ui_element("File icon", "Archive folder")

    assert first.startswith("SUCCESS:")
    assert second.startswith("ERROR: drag_ui_element failed to drag")
    assert "rate_limited" in second


def test_drag_ui_element_registered_in_tool_registry_with_required_params() -> None:
    from dana.tools.registry import get_tool_registry

    entry = get_tool_registry(reload=True).get("drag_ui_element")
    assert entry is not None
    param_names = {(p.name, p.required) for p in entry.spec.parameters}
    assert ("source_description", True) in param_names
    assert ("destination_description", True) in param_names


def test_default_args_for_forced_drag_ui_element() -> None:
    from dana.agentic_react_graph import _default_args_for_forced_tool

    args = _default_args_for_forced_tool(
        "drag_ui_element", "drag the invoice icon to the archive folder"
    )
    assert args == {
        "source_description": "the invoice icon",
        "destination_description": "the archive folder",
    }

    fallback = _default_args_for_forced_tool("drag_ui_element", "drag the report file")
    assert fallback == {
        "source_description": "the report file",
        "destination_description": "",
    }


def test_drag_ui_element_triggers_full_move_down_move_up_sequence_when_not_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end with dry-run OFF: mocks the os_control SendInput entry
    points (never real hardware) and asserts the compound tool triggers both
    vision lookups plus the exact move -> down -> move(s) -> up sequence."""
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "0")

    lookup_calls: list[str] = []
    bboxes_by_label = {
        "File icon": [500, 500, 600, 600],
        "Archive folder": [800, 800, 900, 900],
    }

    import dana.graph.nodes.vision as vision_node
    import dana.middleware.kill_switch as kill_switch
    import dana.tools.os_control as os_control
    import dana.vision_tools as vision_tools

    monkeypatch.setattr(vision_tools, "capture_screen_frame", lambda: object())
    monkeypatch.setattr(
        vision_node,
        "locate_ui_element",
        lambda img, label: (lookup_calls.append(label), bboxes_by_label.get(label))[1],
    )
    monkeypatch.setattr(os_control, "get_screen_size", lambda: (2000, 2000))
    monkeypatch.setattr(kill_switch, "halt_if_requested", lambda: False)

    sequence: list[tuple[str, tuple[int, int] | None]] = []
    monkeypatch.setattr(
        os_control,
        "move_cursor_absolute",
        lambda x, y: sequence.append(("move", (x, y))),
    )
    monkeypatch.setattr(
        os_control, "mouse_down_sendinput", lambda: sequence.append(("down", None))
    )
    monkeypatch.setattr(
        os_control, "mouse_up_sendinput", lambda: sequence.append(("up", None))
    )
    monkeypatch.setattr(drag_actuator.time, "sleep", lambda _s: None)
    monkeypatch.setattr(drag_actuator.random, "uniform", lambda a, b: 0.0)

    out = vision_tool.drag_ui_element("File icon", "Archive folder")

    assert out == (
        "SUCCESS: Dragged 'File icon' from (1100, 1100) to "
        "'Archive folder' at (1700, 1700)"
    )
    # Both elements were located via the hybrid grounding lookup.
    assert lookup_calls == ["File icon", "Archive folder"]

    # Full physical sequence: move to source, press down, one or more
    # human-cadenced moves toward the destination, release at the end.
    assert sequence[0] == ("move", (1100, 1100))
    assert sequence[1] == ("down", None)
    assert sequence[-1] == ("up", None)
    move_events = [s for s in sequence if s[0] == "move"]
    assert len(move_events) >= 2
    assert move_events[-1] == ("move", (1700, 1700))
    down_events = [s for s in sequence if s[0] == "down"]
    up_events = [s for s in sequence if s[0] == "up"]
    assert len(down_events) == 1
    assert len(up_events) == 1
