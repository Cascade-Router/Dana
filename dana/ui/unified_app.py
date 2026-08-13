"""Unified Gradio 5 UI — single source of truth for local desktop AND the
Hugging Face Space (see ``hf_space/app.py`` and
``scripts/launchers/launch_gradio_local.py``, both of which just import and
launch ``demo`` from this module).

Every tool dispatch below goes through ``dana.platform.get_control_plane()``
/ ``get_cad_engine()`` — never Win32 or FreeCADCmd directly — so the exact
same code path drives real actuation on a Windows desktop and simulated
telemetry on a Space; only the driver selected by
``dana.platform.factory`` differs.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import gradio as gr

from dana.platform import factory as platform_factory
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.plugin_manager import discover_plugin_dirs, load_all_plugins
from dana.security.dry_run import is_dry_run_enabled
from dana.tools.schema import ToolCall

try:
    import spaces
except ImportError:  # local desktop / any host without the `spaces` package

    class _NoOpSpaces:
        """Stand-in for the `spaces` package outside a real HF ZeroGPU Space."""

        @staticmethod
        def GPU(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

    spaces = _NoOpSpaces()  # type: ignore[assignment]

APP_TITLE = "Dana AI Co-Pilot"

QUICK_PROMPTS = (
    "Build a box 60x40x20",
    "Create a cylinder radius 10 height 30",
    "Resync the workspace",
    "System status",
)


class ToolResult:
    """Dispatch outcome for one ``ToolCall`` — mirrors the broker's shape
    without importing ``dana.tools.broker`` (heavier than this UI needs)."""

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
    return {"ok": True, **_driver_state(engine, cp)}


def _tool_check_plugin_registry(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return {"ok": True, **_plugin_registry_view()}


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "create_freecad_cylinder": _tool_create_cylinder,
    "resync_workspace": _tool_resync_workspace,
    "get_active_display": _tool_get_active_display,
    "prevent_focus_steal": _tool_prevent_focus_steal,
    "system_state": _tool_system_state,
    "check_plugin_registry": _tool_check_plugin_registry,
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
    (re.compile(r"\bresync\b.*\bworkspace\b", re.I), "resync_workspace", ()),
    (re.compile(r"\b(active|open)\s+(windows?|display)\b", re.I), "get_active_display", ()),
    (re.compile(r"\bfocus\b", re.I), "prevent_focus_steal", ()),
    (re.compile(r"\b(system state|status|driver)\b", re.I), "system_state", ()),
    (re.compile(r"\bplugins?\b", re.I), "check_plugin_registry", ()),
)


def _parse_utterance(text: str) -> ToolCall | None:
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


def _dispatch(call: ToolCall, engine: Any, control_plane: Any) -> ToolResult:
    handler = TOOL_HANDLERS.get(call.tool_id)
    if handler is None:
        return ToolResult(call.tool_id, False, {}, f"unknown tool_id '{call.tool_id}'", 0)
    start = time.perf_counter()
    try:
        payload = handler(call.arguments, engine, control_plane)
        ok = bool(payload.get("ok", True))
        message = "ok" if ok else str(payload.get("error") or "tool reported failure")
    except Exception as exc:  # noqa: BLE001 — surface as a failed ToolResult, not a crashed UI
        payload, ok, message = {}, False, str(exc)
    duration_ms = int((time.perf_counter() - start) * 1000)
    return ToolResult(call.tool_id, ok, payload, message, duration_ms)


def _summarize(call: ToolCall, result: ToolResult) -> str:
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
    if call.tool_id == "system_state":
        return (
            f"control_plane={payload.get('control_plane')}, cad_engine={payload.get('cad_engine')}, "
            f"is_hf_space={payload.get('is_hf_space')}, dry_run={payload.get('dry_run')}."
        )
    return f"`{call.tool_id}` completed: {payload}"


def _driver_state(engine: Any | None = None, control_plane: Any | None = None) -> dict[str, Any]:
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


def _plugin_registry_view() -> dict[str, Any]:
    plugins = [p.name for p in discover_plugin_dirs()]
    tools = [{"id": spec.id, "description": spec.description_en} for spec, _fn in load_all_plugins()]
    return {"plugins": plugins, "tools": tools}


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


def chat_fn(user_text: str, history: list[dict[str, str]] | None) -> tuple[Any, ...]:
    history = list(history or [])
    driver_state = _driver_state()
    plugin_view = _plugin_registry_view()

    if not user_text or not user_text.strip():
        return history, "", "", driver_state, plugin_view

    engine = get_cad_engine()
    control_plane = get_control_plane()

    call = _parse_utterance(user_text)
    history = history + [{"role": "user", "content": user_text}]

    if call is None:
        reply = (
            "I didn't match that to a registered tool. Try one of the quick "
            "prompts, or phrase it like a command (e.g. \"build a box 60x40x20\")."
        )
        history.append({"role": "assistant", "content": reply})
        return history, "", "no tool matched", driver_state, plugin_view

    result = _dispatch(call, engine, control_plane)
    reply = _summarize(call, result)
    history.append({"role": "assistant", "content": reply})

    log_text = (
        f"tool_id={call.tool_id}\n"
        f"status={'ok' if result.ok else 'error'} duration_ms={result.duration_ms}\n"
        f"payload={result.payload}"
    )
    return history, "", log_text, _driver_state(engine, control_plane), plugin_view


# ---------------------------------------------------------------------------
# CAD Generator & 3D Viewer tab
# ---------------------------------------------------------------------------


def generate_preview(
    shape: str,
    length: float,
    width: float,
    height: float,
    radius: float,
    hole_radius: float,
    name: str,
) -> tuple[dict[str, Any], str | None]:
    engine = get_cad_engine()
    name = (name or "DanaModel").strip() or "DanaModel"

    if shape == "cylinder":
        result = engine.create_cylinder(float(radius), float(height), name=name)
    elif shape == "box_with_hole":
        box = engine.create_box(float(length), float(width), float(height), name=f"{name}_Base")
        if not box.get("ok"):
            return box, None
        hole = engine.create_cylinder(float(hole_radius), float(height) * 2.2, name=f"{name}_Hole")
        if not hole.get("ok"):
            return hole, None
        result = engine.apply_boolean_cut(box["path"], hole["path"], name=name)
    else:  # "box"
        result = engine.create_box(float(length), float(width), float(height), name=name)

    if not result.get("ok"):
        return result, None

    mesh = engine.export_mesh_stl(result["path"], name=f"{name}_preview")
    if not mesh.get("ok"):
        return {**result, "export_error": mesh.get("error")}, None
    return {**result, "mesh_path": mesh["path"]}, mesh["path"]


# ---------------------------------------------------------------------------
# Blueprint Vision tab — real dana.tools.cad_vision.analyze_cad_blueprint
# (local Ollama VLM first, optional cloud fallback), not a heuristic mock.
# ---------------------------------------------------------------------------


@spaces.GPU
def analyze_blueprint(image: Any) -> dict[str, Any]:
    if image is None:
        return {"ok": False, "error": "no image uploaded"}

    import base64
    import io
    import json

    from dana.tools.cad_vision import analyze_cad_blueprint

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return json.loads(analyze_cad_blueprint(image_b64))


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    driver_state = _driver_state()

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# 🤖 {APP_TITLE}\n"
            "One Gradio codebase for local desktop and the Hugging Face Space — every "
            "tool call runs through `dana.platform.get_control_plane()` / "
            "`get_cad_engine()`, so the same dispatch path drives real Win32/FreeCAD "
            "actuation locally and simulated telemetry on a Space. Active drivers: "
            f"`{driver_state['control_plane']}` / `{driver_state['cad_engine']}`."
        )

        with gr.Tabs():
            with gr.Tab("ReAct Co-Pilot Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(type="messages", height=460, label="Dana")
                        msg = gr.Textbox(
                            placeholder="Ask Dana to do something (e.g. 'build a box 60x40x20')...",
                            show_label=False,
                        )
                        with gr.Row():
                            quick_buttons = [gr.Button(p, size="sm") for p in QUICK_PROMPTS]
                    with gr.Column(scale=2):
                        gr.Markdown("### Tool Dispatch Log")
                        tool_log = gr.Textbox(lines=8, interactive=False, show_label=False)
                        gr.Markdown("### Active Drivers")
                        driver_box = gr.JSON(value=driver_state)
                        gr.Markdown("### Plugin Registry")
                        plugins_box = gr.JSON(value=_plugin_registry_view())

                chat_outputs = [chatbot, msg, tool_log, driver_box, plugins_box]
                msg.submit(chat_fn, [msg, chatbot], chat_outputs)
                for button, prompt in zip(quick_buttons, QUICK_PROMPTS, strict=True):
                    button.click(partial(chat_fn, prompt), [chatbot], chat_outputs)

            with gr.Tab("CAD Generator & 3D Viewer"):
                gr.Markdown(
                    "Parametric geometry generated through the active `BaseCADEngine` "
                    "driver — a real `.FCStd` document via `FreeCADCmd` on a Windows "
                    "desktop, or headless `trimesh` geometry on a Space."
                )
                with gr.Row():
                    with gr.Column():
                        shape = gr.Radio(
                            ["box", "cylinder", "box_with_hole"], value="box", label="Shape"
                        )
                        name_in = gr.Textbox(value="DanaModel", label="Name")
                        length_in = gr.Number(value=60, label="Length (box)")
                        width_in = gr.Number(value=40, label="Width (box)")
                        height_in = gr.Number(value=20, label="Height")
                        radius_in = gr.Number(value=10, label="Radius (cylinder)")
                        hole_radius_in = gr.Number(value=8, label="Hole radius (box_with_hole)")
                        generate_btn = gr.Button("Generate", variant="primary")
                        result_json = gr.JSON(label="Engine result")
                    with gr.Column():
                        model3d = gr.Model3D(label="3D Preview")

                generate_btn.click(
                    generate_preview,
                    [shape, length_in, width_in, height_in, radius_in, hole_radius_in, name_in],
                    [result_json, model3d],
                )

            with gr.Tab("Blueprint Vision"):
                gr.Markdown(
                    "Upload an engineering drawing or CAD viewport screenshot. Runs the "
                    "real `dana.tools.cad_vision.analyze_cad_blueprint` tool — local "
                    "Ollama VLM first, optional cloud fallback if "
                    "`DANA_ALLOW_CLOUD_FALLBACK` is set. Returns an error, not a fake "
                    "result, if neither is reachable (e.g. a fresh Space with no Ollama "
                    "daemon and no fallback configured)."
                )
                with gr.Row():
                    image_in = gr.Image(type="pil", label="Blueprint / viewport image")
                    analysis_json = gr.JSON(label="Extracted entities")
                analyze_btn = gr.Button("Analyze Blueprint", variant="primary")
                analyze_btn.click(analyze_blueprint, [image_in], [analysis_json])

            with gr.Tab("System, Drivers & Plugins"):
                gr.Markdown(
                    "### Adaptive Execution Drivers\n"
                    "`dana.platform.factory.get_control_plane()` / `get_cad_engine()` pick "
                    "the concrete driver at call time: `Win32ControlPlane` + "
                    "`RealFreeCADEngine` on a Windows desktop, `MockControlPlane` + "
                    "`MockFreeCADEngine` on a Hugging Face Space (`SPACE_ID` set) or any "
                    "other non-Windows host, `MacOSControlPlane` (stub) on macOS."
                )
                refresh_btn = gr.Button("Refresh")
                system_json = gr.JSON(value=driver_state, label="Active drivers")
                plugin_json = gr.JSON(value=_plugin_registry_view(), label="Plugin registry")
                refresh_btn.click(
                    lambda: (_driver_state(), _plugin_registry_view()), None, [system_json, plugin_json]
                )

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
