"""Shared ReAct tool-dispatch core — UI-agnostic.

Originally extracted from the now-deleted Gradio ``dana.ui.unified_app``
(replaced by the headless ``dana.api.server`` FastAPI/WebSocket server + the
Tauri/React frontend) so the dispatch pipeline (intent parsing, tool
registry, driver/plugin introspection) lives in exactly one place, reusable
by any future frontend. Every tool dispatch still goes through
``dana.platform.get_control_plane()`` / ``get_cad_engine()`` — never Win32 or
FreeCADCmd directly.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from dana.platform import factory as platform_factory
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.plugin_manager import discover_plugin_dirs, load_all_plugins
from dana.security.dry_run import is_dry_run_enabled
from dana.tools.cad_vision import analyze_cad_blueprint, capture_cad_viewport
from dana.tools.schema import ToolCall


class ToolResult:
    """Dispatch outcome for one ``ToolCall`` — mirrors the broker's shape
    without importing ``dana.tools.broker`` (heavier than callers need)."""

    __slots__ = ("tool_id", "ok", "payload", "message", "duration_ms")

    def __init__(self, tool_id: str, ok: bool, payload: dict[str, Any], message: str, duration_ms: int):
        self.tool_id = tool_id
        self.ok = ok
        self.payload = payload
        self.message = message
        self.duration_ms = duration_ms


# ---------------------------------------------------------------------------
# Tool registry — every handler receives (arguments, cad_engine, control_plane)
# and returns a plain result dict. This is where "dispatch through the
# abstract drivers, not Win32/FreeCADCmd directly" is actually enforced.
# ---------------------------------------------------------------------------


def _tool_create_box(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_box(
        float(args.get("length", 40)),
        float(args.get("width", 25)),
        float(args.get("height", 15)),
        name=str(args.get("name") or "Box"),
    )


def _tool_create_cylinder(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_cylinder(
        float(args.get("radius", 10)),
        float(args.get("height", 30)),
        name=str(args.get("name") or "Cylinder"),
    )


def _tool_resync_workspace(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.resync_workspace()


def _tool_get_active_display(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.get_active_display()


def _tool_prevent_focus_steal(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.prevent_focus_steal()


def _tool_system_state(_args: dict[str, Any], engine: Any, cp: Any) -> dict[str, Any]:
    return {"ok": True, **driver_state(engine, cp)}


def _tool_check_plugin_registry(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return {"ok": True, **plugin_registry_view()}


def _tool_execute_vision_analysis(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    capture = capture_cad_viewport(save_copy=True)
    if not capture.get("ok"):
        return {"ok": False, "error": capture.get("error") or "capture failed"}
    if not capture.get("window_found"):
        return {"ok": False, "error": "no AutoCAD/FreeCAD window found on screen"}

    try:
        analysis = json.loads(analyze_cad_blueprint(capture["path"]))
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "error": "vision analysis returned non-JSON"}
    if not analysis.get("ok"):
        return {"ok": False, "error": analysis.get("error") or "vision analysis failed"}

    return {**analysis, "image_url": "/api/vision/last_cad_viewport.png"}


def _tool_manipulate_camera(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    position = args.get("position")
    target = args.get("target")
    if not (isinstance(position, (list, tuple)) and len(position) == 3):
        return {"ok": False, "error": "manipulate_camera requires a 3-element 'position'"}
    if not (isinstance(target, (list, tuple)) and len(target) == 3):
        return {"ok": False, "error": "manipulate_camera requires a 3-element 'target'"}
    return {"ok": True, "position": [float(v) for v in position], "target": [float(v) for v in target]}


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "create_freecad_cylinder": _tool_create_cylinder,
    "resync_workspace": _tool_resync_workspace,
    "get_active_display": _tool_get_active_display,
    "prevent_focus_steal": _tool_prevent_focus_steal,
    "system_state": _tool_system_state,
    "check_plugin_registry": _tool_check_plugin_registry,
    "execute_vision_analysis": _tool_execute_vision_analysis,
    "manipulate_camera": _tool_manipulate_camera,
}

# Tools that mutate on-disk geometry or actuate the OS/CAD host pause for
# human-in-the-loop approval before dispatch; everything else (status reads,
# vision, camera framing) is safe to run immediately.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {"create_freecad_box", "create_freecad_cylinder", "resync_workspace", "prevent_focus_steal"}
)


def is_mutating_tool(tool_id: str) -> bool:
    return tool_id in MUTATING_TOOLS


def describe_tool_call(call: ToolCall) -> str:
    if call.tool_id == "create_freecad_box":
        length = call.arguments.get("length", 40)
        width = call.arguments.get("width", 25)
        height = call.arguments.get("height", 15)
        return f"Create a {length}x{width}x{height}mm box in FreeCAD."
    if call.tool_id == "create_freecad_cylinder":
        radius = call.arguments.get("radius", 10)
        height = call.arguments.get("height", 30)
        return f"Create a cylinder (radius {radius}mm, height {height}mm) in FreeCAD."
    if call.tool_id == "resync_workspace":
        return "Reposition managed FreeCAD windows onto their target monitor."
    if call.tool_id == "prevent_focus_steal":
        return "Read the foreground window without changing OS focus."
    return f"Run `{call.tool_id}`."


# (regex, tool_id, arg_names) — first match wins, same shape as the
# simplified broker sibling this replaces in hf_space/hf_sandbox/agent_bridge.py.
_NUM = r"(\d+(?:\.\d+)?)"
# A unit suffix is optional and non-capturing, so a group like "50mm" still
# captures the clean numeric string "50" — no separate stripping step needed
# before float() at the tool-handler level.
_UNIT = r"(?:\s*(?:mm|cm|in|inch(?:es)?))?"
_DIM_SEP = r"\s*(?:x|by|,)\s*"
_DIM = rf"{_NUM}{_UNIT}"
_DIMS3 = rf"{_DIM}{_DIM_SEP}{_DIM}{_DIM_SEP}{_DIM}"
_DIMS2 = rf"{_DIM}{_DIM_SEP}{_DIM}"
_BOX_WORD = r"\b(?:box|mounting plate|cube|cuboid|block)\b"
_CYLINDER_WORD = r"\bcylinder\b"

INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str, tuple[str, ...]], ...] = (
    # Dimensions may land either after the object noun ("box 60x40x20") or
    # before it ("60x40x20mm box") — matched as two separate patterns (each
    # with its own 3 capturing groups) rather than one alternation, so the
    # arg_names zip below always lines up with exactly 3 groups either way.
    (
        re.compile(rf"{_BOX_WORD}.*?{_DIMS3}", re.I),
        "create_freecad_box",
        ("length", "width", "height"),
    ),
    (
        re.compile(rf"{_DIMS3}.*?{_BOX_WORD}", re.I),
        "create_freecad_box",
        ("length", "width", "height"),
    ),
    (re.compile(_BOX_WORD, re.I), "create_freecad_box", ()),
    (
        re.compile(rf"{_CYLINDER_WORD}.*?radius\s*{_NUM}{_UNIT}.*?height\s*{_NUM}{_UNIT}", re.I),
        "create_freecad_cylinder",
        ("radius", "height"),
    ),
    (
        re.compile(rf"{_CYLINDER_WORD}.*?{_DIMS2}", re.I),
        "create_freecad_cylinder",
        ("radius", "height"),
    ),
    (
        re.compile(rf"{_DIMS2}.*?{_CYLINDER_WORD}", re.I),
        "create_freecad_cylinder",
        ("radius", "height"),
    ),
    (re.compile(_CYLINDER_WORD, re.I), "create_freecad_cylinder", ()),
    (
        re.compile(r"\b(look at|see|check|analyze|what'?s on)\b.*\b(screen|window|freecad|blueprint|cad)\b", re.I),
        "execute_vision_analysis",
        (),
    ),
    (re.compile(r"\bresync\b.*\bworkspace\b", re.I), "resync_workspace", ()),
    (re.compile(r"\b(active|open)\s+(windows?|display)\b", re.I), "get_active_display", ()),
    (re.compile(r"\bfocus\b", re.I), "prevent_focus_steal", ()),
    (re.compile(r"\b(system state|status|driver)\b", re.I), "system_state", ()),
    (re.compile(r"\bplugins?\b", re.I), "check_plugin_registry", ()),
)

# Preset camera poses for "look at it from the top/front/side" phrasing —
# distances are tuned for the ~40-100mm primitives create_freecad_* produces.
_CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
    "top": (0, 200, 0.001),
    "front": (0, 0, 200),
    "side": (200, 0, 0),
    "iso": (120, 120, 120),
}
_CAMERA_PRESET_PATTERN = re.compile(
    r"\b(?:look at|view|orbit to|camera)\b.*?\b(top|front|side|iso(?:metric)?)\b", re.I
)
_SELECTION_REFERENCE_PATTERN = re.compile(r"\b(this|here|that spot|selected)\b", re.I)


def parse_utterance(text: str, active_selection: dict[str, Any] | None = None) -> ToolCall | None:
    camera_match = _CAMERA_PRESET_PATTERN.search(text)
    if camera_match:
        preset = camera_match.group(1).lower()
        preset = "iso" if preset.startswith("iso") else preset
        target = active_selection.get("centroid") if active_selection else None
        target = target if isinstance(target, list) and len(target) == 3 else [0.0, 0.0, 0.0]
        position = list(_CAMERA_PRESETS[preset])
        return ToolCall(
            tool_id="manipulate_camera",
            arguments={"position": position, "target": target},
            raw_text=text,
        )

    for pattern, tool_id, arg_names in INTENT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        arguments: dict[str, Any] = {}
        if arg_names and m.groups():
            for name, value in zip(arg_names, m.groups(), strict=True):
                if value is not None:
                    arguments[name] = value
        if (
            active_selection
            and tool_id in ("create_freecad_box", "create_freecad_cylinder")
            and _SELECTION_REFERENCE_PATTERN.search(text)
        ):
            arguments["target_position"] = active_selection.get("centroid")
            arguments["target_normal"] = active_selection.get("normal")
        return ToolCall(tool_id=tool_id, arguments=arguments, raw_text=text)
    return None


def dispatch_tool_call(call: ToolCall, engine: Any, control_plane: Any) -> ToolResult:
    handler = TOOL_HANDLERS.get(call.tool_id)
    if handler is None:
        return ToolResult(call.tool_id, False, {}, f"unknown tool_id '{call.tool_id}'", 0)
    start = time.perf_counter()
    try:
        payload = handler(call.arguments, engine, control_plane)
        ok = bool(payload.get("ok", True))
        message = "ok" if ok else str(payload.get("error") or "tool reported failure")
    except Exception as exc:  # noqa: BLE001 — surface as a failed ToolResult, not a crashed caller
        payload, ok, message = {}, False, str(exc)
    duration_ms = int((time.perf_counter() - start) * 1000)
    return ToolResult(call.tool_id, ok, payload, message, duration_ms)


def summarize_result(call: ToolCall, result: ToolResult) -> str:
    if not result.ok:
        return f"`{call.tool_id}` failed: {result.message}"
    payload = result.payload
    if call.tool_id in ("create_freecad_box", "create_freecad_cylinder"):
        driver = payload.get("driver", "win32/freecad")
        return (
            f"Created `{payload.get('type')}` named `{payload.get('name')}` via the "
            f"**{driver}** driver -> `{payload.get('path')}`."
        )
    if call.tool_id == "resync_workspace":
        moved = payload.get("moved", [])
        return (
            f"Resynced workspace — {len(moved)} window(s) repositioned (zero-focus)."
            if moved
            else "Resynced workspace — nothing to move."
        )
    if call.tool_id == "check_plugin_registry":
        return "Active plugins: " + ", ".join(payload.get("plugins", [])) + "."
    if call.tool_id == "execute_vision_analysis":
        return str(payload.get("summary") or "Analyzed the CAD viewport.")
    if call.tool_id == "manipulate_camera":
        return f"Moved the camera to {payload.get('position')}, looking at {payload.get('target')}."
    if call.tool_id == "system_state":
        return (
            f"control_plane={payload.get('control_plane')}, cad_engine={payload.get('cad_engine')}, "
            f"is_hf_space={payload.get('is_hf_space')}, dry_run={payload.get('dry_run')}."
        )
    return f"`{call.tool_id}` completed: {payload}"


def driver_state(engine: Any | None = None, control_plane: Any | None = None) -> dict[str, Any]:
    engine = engine or get_cad_engine()
    control_plane = control_plane or get_control_plane()
    return {
        "control_plane": type(control_plane).__name__,
        "cad_engine": type(engine).__name__,
        "is_hf_space": platform_factory.IS_HF_SPACE,
        "is_windows": platform_factory.IS_WINDOWS,
        "is_mac": platform_factory.IS_MAC,
        "dry_run": is_dry_run_enabled(),
    }


def plugin_registry_view() -> dict[str, Any]:
    plugins = [p.name for p in discover_plugin_dirs()]
    tools = [{"id": spec.id, "description": spec.description_en} for spec, _fn in load_all_plugins()]
    return {"plugins": plugins, "tools": tools}


__all__ = (
    "INTENT_PATTERNS",
    "MUTATING_TOOLS",
    "TOOL_HANDLERS",
    "ToolResult",
    "describe_tool_call",
    "dispatch_tool_call",
    "driver_state",
    "is_mutating_tool",
    "parse_utterance",
    "plugin_registry_view",
    "summarize_result",
)
