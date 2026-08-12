"""Sandbox re-implementation of Dana's IntentBroker + tool dispatch loop.

Mirrors the shape of the real `dana/tools/schema.py` ToolCall/ToolSpec dataclasses
and `dana/tools/broker.py` IntentBroker (regex/alias intent matching + confidence
scoring), against a small demo tool registry. OS/Win32/FreeCAD actuators are
mocked — this container has no Windows APIs or FreeCAD binary — but the
parse -> match -> dispatch -> log control flow is real.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

HIGH_CONFIDENCE_TOOL_THRESHOLD = 0.95
ALIAS_HIT_CONFIDENCE = 0.85
FALLBACK_CONFIDENCE = 0.4


@dataclass
class ToolCall:
    """Mirrors dana/tools/schema.py::ToolCall."""

    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source_lang: str = "en"
    raw_text: str = ""
    confidence: float = 1.0
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class ToolResult:
    tool_id: str
    call_id: str
    ok: bool
    payload: dict[str, Any]
    message: str
    duration_ms: int
    zero_focus: bool = False
    mocked: bool = True


def _mock(delay_ms: int = 30):
    """Decorator that simulates realistic dispatch latency for a handler."""

    def wrap(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        fn._sim_delay_ms = delay_ms  # type: ignore[attr-defined]
        return fn

    return wrap


@_mock(delay_ms=180)
def create_freecad_box(args: dict[str, Any]) -> dict[str, Any]:
    length = float(args.get("length", 40))
    width = float(args.get("width", 25))
    height = float(args.get("height", 15))
    return {
        "document": "Unnamed.FCStd",
        "object": "Box",
        "dims_mm": {"length": length, "width": width, "height": height},
        "note": "mocked — production path shells out to FreeCADCmd.exe headless",
    }


@_mock(delay_ms=210)
def create_freecad_cylinder(args: dict[str, Any]) -> dict[str, Any]:
    radius = float(args.get("radius", 10))
    height = float(args.get("height", 30))
    return {
        "document": "Unnamed.FCStd",
        "object": "Cylinder",
        "dims_mm": {"radius": radius, "height": height},
        "note": "mocked — production path shells out to FreeCADCmd.exe headless",
    }


@_mock(delay_ms=140)
def modify_existing_freecad_document(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "document": args.get("path", "assembly.FCStd"),
        "operation": args.get("operation", "add_fillet"),
        "note": "mocked — preferred iterative-edit path on a live .FCStd in production",
    }


@_mock(delay_ms=260)
def analyze_cad_blueprint(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "hint": "Upload an image in the 'CAD Blueprint Vision' tab for a real "
        "(or heuristic-mock) geometry extraction — this chat tool call only "
        "demonstrates dispatch, it carries no image payload.",
    }


@_mock(delay_ms=90)
def capture_cad_viewport(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "monitor": "secondary",
        "region_px": [0, 0, 1920, 1080],
        "focus_stolen": False,
        "note": "mocked screen grab — production uses mss for a zero-focus region capture",
    }


@_mock(delay_ms=20)
def get_active_windows(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "windows": [
            {"title": "FreeCAD 1.0 — assembly.FCStd", "pid": 4021, "focused": True},
            {"title": "Dana — Live Trace", "pid": 3110, "focused": False},
            {"title": "Notepad", "pid": 5588, "focused": False},
        ]
    }


@_mock(delay_ms=15)
def move_window_no_activate(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": args.get("title", "FreeCAD 1.0"),
        "to": args.get("position", [1920, 0]),
        "focus_stolen": False,
        "flag": "SWP_NOACTIVATE",
    }


@_mock(delay_ms=10)
def toggle_audio_endpoint(args: dict[str, Any]) -> dict[str, Any]:
    return {"endpoint": args.get("endpoint", "default"), "muted": bool(args.get("mute", False))}


@_mock(delay_ms=25)
def archive_ledger(args: dict[str, Any]) -> dict[str, Any]:
    return {"entries_archived": 12, "ledger": "dana_memory.enc"}


@_mock(delay_ms=5)
def system_state(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "idle_monitor": "USER_ACTIVE",
        "kill_switch_armed": True,
        "kill_switch_hotkey": "F12",
        "dry_run": True,
        "hitl_ticket_required": False,
    }


@_mock(delay_ms=5)
def check_plugin_registry(args: dict[str, Any]) -> dict[str, Any]:
    from . import architecture_docs

    return {"plugins": [p["name"] for p in architecture_docs.DEMO_PLUGINS]}


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "create_freecad_box": create_freecad_box,
    "create_freecad_cylinder": create_freecad_cylinder,
    "modify_existing_freecad_document": modify_existing_freecad_document,
    "analyze_cad_blueprint": analyze_cad_blueprint,
    "capture_cad_viewport": capture_cad_viewport,
    "get_active_windows": get_active_windows,
    "move_window_no_activate": move_window_no_activate,
    "toggle_audio_endpoint": toggle_audio_endpoint,
    "archive_ledger": archive_ledger,
    "system_state": system_state,
    "check_plugin_registry": check_plugin_registry,
}

ZERO_FOCUS_TOOLS = {"capture_cad_viewport", "get_active_windows", "move_window_no_activate"}

# (regex, tool_id, arg_names) — simplified sibling of the real broker's larger
# bilingual alias/regex table in dana/tools/broker.py.
INTENT_PATTERNS: list[tuple[re.Pattern, str, list[str]]] = [
    (re.compile(r"\bbox\b.*?(\d+)\s*(?:x|by|,)\s*(\d+)\s*(?:x|by|,)\s*(\d+)", re.I),
     "create_freecad_box", ["length", "width", "height"]),
    (re.compile(r"\b(box|mounting plate|cube)\b", re.I), "create_freecad_box", []),
    (re.compile(r"\bcylinder\b", re.I), "create_freecad_cylinder", []),
    (re.compile(r"\b(modify|edit|fillet)\b.*\b(document|assembly|fcstd)\b", re.I),
     "modify_existing_freecad_document", []),
    (re.compile(r"\b(analy[sz]e|inspect).*(blueprint|drawing|viewport|geometry)\b", re.I),
     "analyze_cad_blueprint", []),
    (re.compile(r"\b(capture|screenshot).*(viewport|screen)\b", re.I),
     "capture_cad_viewport", []),
    (re.compile(r"\b(active|open)\s+windows?\b", re.I), "get_active_windows", []),
    (re.compile(r"\bmove\b.*\bwindow\b", re.I), "move_window_no_activate", []),
    (re.compile(r"\b(mute|unmute|audio)\b", re.I), "toggle_audio_endpoint", []),
    (re.compile(r"\b(archive|ledger)\b", re.I), "archive_ledger", []),
    (re.compile(r"\b(system state|status|kill.switch|dry.run|hitl)\b", re.I),
     "system_state", []),
    (re.compile(r"\bplugins?\b", re.I), "check_plugin_registry", []),
]


class IntentBroker:
    """Simplified sibling of dana/tools/broker.py::IntentBroker."""

    def parse_utterance(self, text: str) -> ToolCall | None:
        for pattern, tool_id, arg_names in INTENT_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            args: dict[str, Any] = {}
            if arg_names and m.groups():
                for name, value in zip(arg_names, m.groups()):
                    if value is not None:
                        args[name] = value
            confidence = HIGH_CONFIDENCE_TOOL_THRESHOLD if arg_names and args else ALIAS_HIT_CONFIDENCE
            return ToolCall(tool_id=tool_id, arguments=args, raw_text=text, confidence=confidence)
        return None

    def dispatch(self, call: ToolCall) -> ToolResult:
        handler = TOOL_REGISTRY.get(call.tool_id)
        if handler is None:
            return ToolResult(
                tool_id=call.tool_id, call_id=call.call_id, ok=False,
                payload={}, message=f"unknown tool_id '{call.tool_id}'", duration_ms=0,
            )
        delay_ms = getattr(handler, "_sim_delay_ms", 50)
        start = time.perf_counter()
        try:
            payload = handler(call.arguments)
            ok, message = True, "ok"
        except Exception as exc:  # defensive — handlers are mocks and shouldn't raise
            payload, ok, message = {}, False, str(exc)
        duration_ms = max(delay_ms, int((time.perf_counter() - start) * 1000))
        return ToolResult(
            tool_id=call.tool_id, call_id=call.call_id, ok=ok, payload=payload,
            message=message, duration_ms=duration_ms,
            zero_focus=call.tool_id in ZERO_FOCUS_TOOLS,
        )


_broker = IntentBroker()


def run_turn(user_text: str) -> dict[str, Any]:
    """Runs one ReAct-style micro-loop turn: parse -> dispatch -> summarize.

    Returns a dict with `reasoning_steps` (list[str]), `tool_call`, `tool_result`
    (or None if no structured intent matched), and `assistant_text`.
    """
    steps = [f"Parsing utterance via IntentBroker.parse_utterance() — bilingual EN/FA alias matcher"]
    call = _broker.parse_utterance(user_text)

    if call is None:
        steps.append("No structured tool call matched (confidence below threshold) — "
                      "falling back to conversational reasoning, no tool dispatched.")
        reply = (
            "I didn't match that to a registered tool in this sandbox. Try one of the "
            "quick-prompts, or phrase it like a command (e.g. \"create a box 40x25x15\", "
            "\"check plugin registry\", \"system state\")."
        )
        return {"reasoning_steps": steps, "tool_call": None, "tool_result": None, "assistant_text": reply}

    steps.append(
        f"Matched intent -> tool_id='{call.tool_id}' confidence={call.confidence:.2f} "
        f"({'high-confidence force-route' if call.confidence >= HIGH_CONFIDENCE_TOOL_THRESHOLD else 'alias hit'})"
    )
    steps.append(f"Dispatching ToolCall(id={call.call_id}, tool_id='{call.tool_id}') via broker.dispatch() ...")
    result = _broker.dispatch(call)
    focus_note = " [zero-focus: no window activated]" if result.zero_focus else ""
    steps.append(
        f"Tool returned status={'ok' if result.ok else 'error'} in {result.duration_ms}ms{focus_note}"
    )

    reply = _summarize(call, result)
    return {"reasoning_steps": steps, "tool_call": call, "tool_result": result, "assistant_text": reply}


def _summarize(call: ToolCall, result: ToolResult) -> str:
    if not result.ok:
        return f"Tool `{call.tool_id}` failed: {result.message}"
    if call.tool_id in ("create_freecad_box", "create_freecad_cylinder"):
        dims = result.payload.get("dims_mm", {})
        dims_str = ", ".join(f"{k}={v}mm" for k, v in dims.items())
        return (f"Created a mocked `{result.payload.get('object')}` in "
                f"`{result.payload.get('document')}` ({dims_str}). In production this "
                f"shells out to FreeCADCmd.exe — see the CAD Vision tab for a live 3D preview.")
    if call.tool_id == "check_plugin_registry":
        return "Active plugins: " + ", ".join(result.payload.get("plugins", [])) + \
            ". See the Architecture tab for manifests."
    if call.tool_id == "system_state":
        s = result.payload
        return (f"idle_monitor={s['idle_monitor']}, kill_switch armed on "
                f"{s['kill_switch_hotkey']}, dry_run={s['dry_run']}, "
                f"hitl_ticket_required={s['hitl_ticket_required']}.")
    return f"Tool `{call.tool_id}` completed: {result.payload}"
