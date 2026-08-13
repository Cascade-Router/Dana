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


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "create_freecad_cylinder": _tool_create_cylinder,
    "resync_workspace": _tool_resync_workspace,
    "get_active_display": _tool_get_active_display,
    "prevent_focus_steal": _tool_prevent_focus_steal,
    "system_state": _tool_system_state,
    "check_plugin_registry": _tool_check_plugin_registry,
    "execute_vision_analysis": _tool_execute_vision_analysis,
}

# (regex, tool_id, arg_names) — first match wins, same shape as the
# simplified broker sibling this replaces in hf_space/hf_sandbox/agent_bridge.py.
_NUM = r"(\d+(?:\.\d+)?)"
INTENT_PATTERNS: tuple[tuple[re.Pattern[str], str, tuple[str, ...]], ...] = (
    (
        re.compile(rf"\bbox\b.*?{_NUM}\s*(?:x|by|,)\s*{_NUM}\s*(?:x|by|,)\s*{_NUM}", re.I),
        "create_freecad_box",
        ("length", "width", "height"),
    ),
    (re.compile(r"\b(box|mounting plate|cube|cuboid)\b", re.I), "create_freecad_box", ()),
    (
        re.compile(rf"\bcylinder\b.*?radius\s*{_NUM}.*?height\s*{_NUM}", re.I),
        "create_freecad_cylinder",
        ("radius", "height"),
    ),
    (re.compile(r"\bcylinder\b", re.I), "create_freecad_cylinder", ()),
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


def parse_utterance(text: str) -> ToolCall | None:
    for pattern, tool_id, arg_names in INTENT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        arguments: dict[str, Any] = {}
        if arg_names and m.groups():
            for name, value in zip(arg_names, m.groups(), strict=True):
                if value is not None:
                    arguments[name] = value
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
    "TOOL_HANDLERS",
    "ToolResult",
    "dispatch_tool_call",
    "driver_state",
    "parse_utterance",
    "plugin_registry_view",
    "summarize_result",
)
