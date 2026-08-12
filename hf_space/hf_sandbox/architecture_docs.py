"""Static content for the 'System Architecture & Plugin Explorer' tab.

Everything here is documentation, not simulation — the manifests below mirror
the real shape of dana/plugins/*/manifest.json (as loaded by
dana/plugins/plugin_manager.py::load_all_plugins) so the tab is an accurate
map of the production system, not an invented one.
"""

from __future__ import annotations

import json

DEMO_PLUGINS: list[dict] = [
    {
        "name": "freecad",
        "version": "1.2.0",
        "entry_point": "dana.plugins.freecad.engine",
        "tools": [
            {
                "id": "create_freecad_box",
                "function": "create_box",
                "description_en": "Create a parametric box solid in the active or a new FreeCAD document.",
                "parameters": [
                    {"name": "length", "type": "float", "required": False, "description_en": "mm"},
                    {"name": "width", "type": "float", "required": False, "description_en": "mm"},
                    {"name": "height", "type": "float", "required": False, "description_en": "mm"},
                ],
                "aliases_en": {"_intent": ["make a box", "create a cube", "mounting plate"]},
                "aliases_fa": {"_intent": ["یک جعبه بساز"]},
            },
            {
                "id": "modify_existing_freecad_document",
                "function": "modify_document",
                "description_en": "Preferred iterative-edit path: apply an operation to an already-open .FCStd.",
                "parameters": [
                    {"name": "path", "type": "str", "required": True, "description_en": "document path"},
                    {"name": "operation", "type": "str", "required": True, "description_en": "e.g. add_fillet"},
                ],
                "aliases_en": {"_intent": ["edit the assembly", "add a fillet"]},
                "aliases_fa": {"_intent": []},
            },
            {
                "id": "execute_freecad_script",
                "function": "run_script",
                "description_en": "Run an arbitrary Python script via headless FreeCADCmd subprocess (no COM).",
                "parameters": [
                    {"name": "script", "type": "str", "required": True, "description_en": "python source"},
                ],
                "aliases_en": {"_intent": []},
                "aliases_fa": {"_intent": []},
            },
        ],
    },
    {
        "name": "cad_vision",
        "version": "1.0.0",
        "entry_point": "dana.tools.cad_vision",
        "tools": [
            {
                "id": "capture_cad_viewport",
                "function": "capture_cad_viewport",
                "description_en": "Zero-focus screenshot of the active CAD viewport (mss region grab, no window activation).",
                "parameters": [],
                "aliases_en": {"_intent": ["screenshot the viewport", "capture cad screen"]},
                "aliases_fa": {"_intent": []},
            },
            {
                "id": "analyze_cad_blueprint",
                "function": "analyze_cad_blueprint",
                "description_en": "VLM reads geometry off pixels (blueprint or viewport) and returns structured JSON entities.",
                "parameters": [
                    {"name": "image_bytes", "type": "bytes", "required": True, "description_en": "PNG bytes"},
                ],
                "aliases_en": {"_intent": ["analyze this blueprint", "inspect viewport geometry"]},
                "aliases_fa": {"_intent": []},
            },
            {
                "id": "verify_cad_rendering",
                "function": "verify_cad_rendering",
                "description_en": "Diffs a fresh screenshot against an expected geometry spec to confirm a CAD edit landed.",
                "parameters": [],
                "aliases_en": {"_intent": []},
                "aliases_fa": {"_intent": []},
            },
        ],
    },
]


def plugin_manifest_json(plugin_name: str) -> str:
    for plugin in DEMO_PLUGINS:
        if plugin["name"] == plugin_name:
            return json.dumps(plugin, indent=2)
    return json.dumps({"error": f"no demo plugin named '{plugin_name}'"}, indent=2)


def plugin_choices() -> list[str]:
    return [p["name"] for p in DEMO_PLUGINS]


ARCHITECTURE_OVERVIEW_MD = """
## How Dānā is put together

Dānā is a Windows desktop co-pilot built around a small set of composable loops
rather than one monolithic agent:

- **Intent Broker** (`dana/tools/broker.py`) — a bilingual (EN/FA) regex + alias
  matcher scores candidate tools against the utterance, force-routing at
  confidence ≥ 0.95 and falling back to LLM ReAct reasoning otherwise. This
  Space's chat tab runs a simplified sibling of that same parse → dispatch loop.
- **Tool Registry** (`dana/tools/schema.py`, `tools.json`) — every tool is a
  `ToolSpec` (id, bilingual description, typed parameters, aliases) that can be
  hand-registered or loaded dynamically from a plugin.
- **Zero-Touch Plugin System** (`dana/plugins/plugin_manager.py`) — drop a
  folder with a `manifest.json` + entry-point module under `dana/plugins/`, and
  its tools register into the broker with **no edits to broker.py**.
- **Watchdog Compiler** (`dana/swarm/watchdog_graph.py`) — a LangGraph pipeline
  (`dana_coder → ast_static_analyzer → titan_supervisor → repl_executor`) that
  lets an LLM author a small autonomous monitoring script, statically vets it
  for forbidden imports, and runs it sandboxed in `execution_jail/`.
- **Middleware** — `idle_monitor.py` watches `GetLastInputInfo` to know when the
  user is away and reprioritizes background work; `vision_poller.py` runs a
  standalone YOLO daemon publishing `perception.objects` to a shared blackboard.
"""

ZERO_FOCUS_MD = """
## Zero-focus, multi-monitor actuation

Every screen-reading or input-sending tool in `dana/tools/os_control.py` and
`dana/tools/cad_vision.py` is built around one rule: **never steal window
focus**. Concretely:

- Screenshots use region-scoped `mss` grabs against a window's known rect
  (`get_window_rect`) or a secondary monitor (`get_secondary_monitor`) —
  never `set_foreground_window` first.
- Window repositioning uses `move_window_no_activate`, which calls Win32
  `SetWindowPos` with the `SWP_NOACTIVATE` flag so the target window moves
  without becoming the active one.
- Keyboard/mouse actuation (`type_text_sendinput`, `click_left_sendinput`,
  `scroll_wheel_sendinput`, …) goes through raw `SendInput`, letting Dānā
  drive a background CAD window on a secondary monitor while the user keeps
  typing in whatever they had focused.

This Space mocks all of the above (no Win32 APIs in a Linux container) but the
chat tab's tool logs annotate every zero-focus tool call so the pattern is
visible even without the real actuators.
"""

SAFETY_GATES_MD = """
## Safety gates: dry-run, kill-switch, HITL

Three independent layers keep an actuation-capable agent from doing something
irreversible unsupervised:

1. **Dry-run** (`dana/security/dry_run.py`) — a single source of truth,
   `is_dry_run_enabled()`, gated by the `DANA_OS_DRY_RUN` env var. When set,
   actuator tools log their intended action instead of executing it.
2. **F12 panic hotkey** (`dana/middleware/kill_switch.py`) — a global hotkey
   (default `F12`, overridable via `DANA_KILL_HOTKEY`) sets a shared
   `GLOBAL_HALT_EVENT`. Every `SendInput` wrapper in `os_control.py` checks it
   before acting and raises `EmergencyKillSwitchTriggered` if it's set — one
   keypress, from anywhere, stops all input actuation immediately.
3. **HITL ticket gate** (`dana/middleware/hitl_ticket.py`) — for
   higher-stakes tool calls, a LangGraph `interrupt()` opens an approval
   ticket that the Live Trace GUI must Approve/Deny before the graph resumes;
   headless runs auto-approve unless `DANA_HITL_REQUIRE_GUI=1` is set.

This sandbox simulates the gate *decisions* (see the `system_state` tool in
the chat tab) but does not wire a real hotkey or approval UI — there is
nothing here for F12 to stop.
"""
