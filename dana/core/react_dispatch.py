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

import asyncio
import json
import re
import time
from collections.abc import Callable
from typing import Any

from dana.core.model_provider import ModelProvider
from dana.platform import factory as platform_factory
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.plugin_manager import discover_plugin_dirs, load_all_plugins
from dana.security.dry_run import is_dry_run_enabled
from dana.tools.cad_vision import analyze_cad_blueprint, capture_cad_viewport
from dana.tools.schema import ToolCall, ToolSpec, load_tool_registry, openai_tools_schema


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


def _extract_placement(args: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(args.get("placement_x", 0.0) or 0.0),
        float(args.get("placement_y", 0.0) or 0.0),
        float(args.get("placement_z", 0.0) or 0.0),
    )


def _tool_create_box(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_box(
        float(args.get("length", 40)),
        float(args.get("width", 25)),
        float(args.get("height", 15)),
        name=str(args.get("name") or "Box"),
        placement=_extract_placement(args),
    )


def _tool_create_cylinder(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_cylinder(
        float(args.get("radius", 10)),
        float(args.get("height", 30)),
        name=str(args.get("name") or "Cylinder"),
        placement=_extract_placement(args),
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


# Half-width (mm) of the square footprint synthesized when "extrude this/
# here" gives us only a clicked point+normal, not real profile geometry —
# a raycast hit alone can never recover a face's true boundary.
_EXTRUSION_DEFAULT_HALF_WIDTH = 10.0


def _tool_create_freecad_extrusion(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    height = float(args.get("height", args.get("distance", 25)))
    profile_points = args.get("profile_points")

    if not profile_points:
        target_position = args.get("target_position")
        target_normal = args.get("target_normal")
        if not target_position or not target_normal:
            return {
                "ok": False,
                "error": (
                    "create_freecad_extrusion needs either explicit 2D profile points or a "
                    "selected face — click a face in the viewer and say 'extrude this' so its "
                    "position can anchor the profile"
                ),
            }
        # The underlying engine only extrudes straight up along Z (see
        # dana.plugins.freecad.engine.create_extruded_polyline) — it has no
        # arbitrary extrusion-axis support, so a face whose normal isn't
        # close to Z can't be extruded meaningfully here yet.
        if abs(float(target_normal[2])) < 0.5:
            return {
                "ok": False,
                "error": (
                    "the selected face's normal isn't close enough to the Z axis for a "
                    "straight-up extrusion — FreeCAD extrusion here only extrudes along Z, "
                    "so a steep side face can't be extruded correctly yet"
                ),
            }
        x, y = float(target_position[0]), float(target_position[1])
        half = _EXTRUSION_DEFAULT_HALF_WIDTH
        profile_points = [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]

    return engine.create_extrusion(profile_points, height, name=str(args.get("name") or "Extrusion"))


def _tool_create_freecad_pyramid(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_pyramid(
        float(args.get("length", 40)),
        float(args.get("width", 40)),
        float(args.get("height", 60)),
        name=str(args.get("name") or "Pyramid"),
        placement=_extract_placement(args),
    )


def _tool_create_freecad_star_prism(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    points = int(args.get("points", 5))
    if points < 3:
        return {"ok": False, "error": "create_freecad_star_prism requires at least 3 points"}
    return engine.create_star_prism(
        points,
        float(args.get("outer_radius", 50)),
        float(args.get("inner_radius", 20)),
        float(args.get("height", 10)),
        name=str(args.get("name") or "StarPrism"),
        placement=_extract_placement(args),
    )


# Maps a FreeCAD object's LLM-visible label (e.g. "Box") to the on-disk
# .FCStd/.stl path it was last saved to — every create_*/perform_freecad_boolean
# call spawns a brand-new document/subprocess (see dana.plugins.freecad.engine's
# module docstring), so there's no persistent ActiveDocument to fetch objects
# from by name across calls; this is that continuity, entirely dispatch-side so
# neither CAD engine driver needs to know about LLM-facing object names.
_OBJECT_PATH_REGISTRY: dict[str, str] = {}

_BOOLEAN_OPERATIONS = frozenset({"cut", "union", "intersect"})


def _tool_perform_freecad_boolean(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    operation = str(args.get("operation") or "").strip().lower()
    if operation not in _BOOLEAN_OPERATIONS:
        return {
            "ok": False,
            "error": "perform_freecad_boolean requires operation to be one of cut, union, intersect",
        }
    base_name = str(args.get("base_object") or "").strip()
    tool_name = str(args.get("tool_object") or "").strip()
    if not base_name or not tool_name:
        return {"ok": False, "error": "perform_freecad_boolean requires base_object and tool_object"}
    base_path = _OBJECT_PATH_REGISTRY.get(base_name)
    tool_path = _OBJECT_PATH_REGISTRY.get(tool_name)
    if not base_path:
        return {
            "ok": False,
            "error": f"unknown base_object '{base_name}' — create it first with a create_freecad_* tool",
        }
    if not tool_path:
        return {
            "ok": False,
            "error": f"unknown tool_object '{tool_name}' — create it first with a create_freecad_* tool",
        }
    return engine.apply_boolean(operation, base_path, tool_path, name=str(args.get("name") or "").strip() or None)


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "create_freecad_cylinder": _tool_create_cylinder,
    "create_freecad_pyramid": _tool_create_freecad_pyramid,
    "create_freecad_star_prism": _tool_create_freecad_star_prism,
    "create_freecad_extrusion": _tool_create_freecad_extrusion,
    "perform_freecad_boolean": _tool_perform_freecad_boolean,
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
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "perform_freecad_boolean",
        "resync_workspace",
        "prevent_focus_steal",
    }
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
    if call.tool_id == "create_freecad_extrusion":
        height = call.arguments.get("height", 25)
        if call.arguments.get("profile_points"):
            return f"Extrude the given 2D profile by {height}mm in FreeCAD."
        return (
            f"Extrude a default {2 * _EXTRUSION_DEFAULT_HALF_WIDTH:g}x"
            f"{2 * _EXTRUSION_DEFAULT_HALF_WIDTH:g}mm footprint at the selected point by {height}mm "
            "(approximate — exact face bounds aren't available from a single click)."
        )
    if call.tool_id == "create_freecad_pyramid":
        length = call.arguments.get("length", 40)
        width = call.arguments.get("width", 40)
        height = call.arguments.get("height", 60)
        return f"Create a sharp-edged {length}x{width}mm base pyramid, height {height}mm, in FreeCAD."
    if call.tool_id == "create_freecad_star_prism":
        points = call.arguments.get("points", 5)
        outer = call.arguments.get("outer_radius", 50)
        inner = call.arguments.get("inner_radius", 20)
        height = call.arguments.get("height", 10)
        return (
            f"Create a sharp-edged {points}-point star prism (outer radius {outer}mm, "
            f"inner radius {inner}mm, thickness {height}mm) in FreeCAD."
        )
    if call.tool_id == "perform_freecad_boolean":
        operation = str(call.arguments.get("operation") or "?")
        base = call.arguments.get("base_object", "?")
        tool = call.arguments.get("tool_object", "?")
        verbs = {"cut": f"Cut `{tool}` out of `{base}`", "union": f"Fuse `{base}` and `{tool}` together",
                 "intersect": f"Keep only the overlap of `{base}` and `{tool}`"}
        return f"{verbs.get(operation, f'Combine {base} and {tool}')} in FreeCAD."
    if call.tool_id == "resync_workspace":
        return "Reposition managed FreeCAD windows onto their target monitor."
    if call.tool_id == "prevent_focus_steal":
        return "Read the foreground window without changing OS focus."
    return f"Run `{call.tool_id}`."


# Preset camera poses for "look at it from the top/front/side" phrasing —
# distances are tuned for the ~40-100mm primitives create_freecad_* produces.
# The LLM proposes a preset NAME (see manipulate_camera's tools.json schema);
# this is where that name still gets turned into an actual position/target,
# exactly as the old regex parser did.
_CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
    "top": (0, 200, 0.001),
    "front": (0, 0, 200),
    "side": (200, 0, 0),
    "iso": (120, 120, 120),
}

# Cheap deterministic backstop, not a parser: local 7B tool-calling models
# don't reliably copy multi-decimal floats verbatim into JSON arguments, so
# if the LLM omits target_position/target_normal but the user clearly meant
# the active selection ("this"/"here"/...), inject it ourselves rather than
# silently losing the anchor point.
_SELECTION_REFERENCE_PATTERN = re.compile(r"\b(this|here|that spot|selected)\b", re.I)

# Tools exposed to the LLM for function-calling — a subset of tools.json's
# full registry (which also serves dana.tools.broker's regex/alias router
# for the legacy desktop agent; that router is untouched by this module).
_LLM_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "perform_freecad_boolean",
        "manipulate_camera",
        "resync_workspace",
        "get_active_display",
        "prevent_focus_steal",
        "system_state",
        "check_plugin_registry",
    }
)

_llm_tool_registry_cache: dict[str, ToolSpec] | None = None


def _llm_tools_schema() -> list[dict[str, Any]]:
    global _llm_tool_registry_cache
    if _llm_tool_registry_cache is None:
        _llm_tool_registry_cache = load_tool_registry()
    return openai_tools_schema(_llm_tool_registry_cache, tool_ids=_LLM_TOOL_IDS)


def build_system_prompt(active_selection: dict[str, Any] | None) -> str:
    """The dynamic context the LLM reasons over each turn — this is where
    the React 3D viewer's canvas-selection state enters the ReAct loop."""
    lines = [
        "You are Dana, a CAD co-pilot for FreeCAD. Call at most one tool per message, "
        "only when the user is clearly asking for an action. If they're just chatting "
        "or asking a question, reply in plain text without calling a tool.",
    ]
    centroid = active_selection.get("centroid") if active_selection else None
    normal = active_selection.get("normal") if active_selection else None
    if centroid:
        lines.append(
            f"Current active canvas selection: centroid {centroid}, normal {normal}. "
            "If the user refers to 'this', 'here', 'that spot', or 'the selected face', "
            "pass this centroid as target_position and this normal as target_normal "
            "(copy the numbers verbatim — do not invent your own coordinates)."
        )
    return "\n".join(lines)


def _resolve_camera_call(call: ToolCall, active_selection: dict[str, Any] | None) -> None:
    preset = str(call.arguments.get("preset") or "iso").strip().lower()
    preset = "iso" if preset.startswith("iso") else preset
    if preset not in _CAMERA_PRESETS:
        preset = "iso"
    target = active_selection.get("centroid") if active_selection else None
    target = target if isinstance(target, list) and len(target) == 3 else [0.0, 0.0, 0.0]
    call.arguments = {"position": list(_CAMERA_PRESETS[preset]), "target": target}


def _finalize_call_arguments(call: ToolCall, active_selection: dict[str, Any] | None) -> None:
    if call.tool_id == "manipulate_camera":
        _resolve_camera_call(call, active_selection)
        return
    if call.tool_id not in ("create_freecad_box", "create_freecad_cylinder", "create_freecad_extrusion"):
        return
    has_anchor = call.arguments.get("target_position") and call.arguments.get("target_normal")
    if not has_anchor and active_selection and _SELECTION_REFERENCE_PATTERN.search(call.raw_text or ""):
        call.arguments["target_position"] = active_selection.get("centroid")
        call.arguments["target_normal"] = active_selection.get("normal")


async def parse_utterance(text: str, active_selection: dict[str, Any] | None = None) -> ToolCall | None:
    """One reason-then-act ReAct step: ask the local LLM whether this
    utterance needs a tool call, via the existing OpenAI-tool-calling bridge
    (``dana.core.model_provider.ModelProvider`` + ``dana.tools.schema``) —
    no separate ReAct while-loop built here, and no regex intent parsing.
    """
    text = (text or "").strip()
    if not text:
        return None

    messages = [
        {"role": "system", "content": build_system_prompt(active_selection)},
        {"role": "user", "content": text},
    ]
    provider = ModelProvider()
    try:
        result = await asyncio.to_thread(
            provider.complete_with_tool_calls,
            messages,
            tools=_llm_tools_schema(),
            provider="ollama",
        )
    except Exception:  # noqa: BLE001 — Ollama unreachable/model missing degrades to "no tool", not a crash
        return None

    tool_calls = result.get("tool_calls") or []
    if not tool_calls:
        return None
    call = tool_calls[0]  # one tool per turn, matching the existing single parse/dispatch DAG shape
    if call.tool_id not in TOOL_HANDLERS:
        return None
    call.raw_text = text
    _finalize_call_arguments(call, active_selection)
    return call


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
    if ok and isinstance(payload, dict) and payload.get("name") and payload.get("path"):
        _OBJECT_PATH_REGISTRY[str(payload["name"])] = str(payload["path"])
    return ToolResult(call.tool_id, ok, payload, message, duration_ms)


def summarize_result(call: ToolCall, result: ToolResult) -> str:
    if not result.ok:
        return f"`{call.tool_id}` failed: {result.message}"
    payload = result.payload
    if call.tool_id in (
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "perform_freecad_boolean",
    ):
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
    "MUTATING_TOOLS",
    "TOOL_HANDLERS",
    "ToolResult",
    "build_system_prompt",
    "describe_tool_call",
    "dispatch_tool_call",
    "driver_state",
    "is_mutating_tool",
    "parse_utterance",
    "plugin_registry_view",
    "summarize_result",
)
