# Dānā Plugin Development Guide

Dānā's tool surface is extensible via **zero-touch plugins**: self-contained
folders under `dana/plugins/` that ship their own manifest and implementation.
Dropping a valid plugin folder in place and restarting is enough — no changes
to `dana/tools/broker.py` or `dana/core/agent_loop.py` are required. See
[`architecture.md`](architecture.md#6-zero-touch-dynamic-plugin-architecture)
for how the loader and feature gating work under the hood; this document is
the "how do I add one" guide.

The reference implementation to copy from is `dana/plugins/freecad/`.

## 1. Folder layout

```text
dana/plugins/<your_plugin>/
├── manifest.json   # required — declares the plugin + its tools
└── engine.py       # required (or whatever `entry_point` names) — the callables
```

`dana/plugins/plugin_manager.py` discovers a plugin purely by the presence of
`manifest.json` (`discover_plugin_dirs()` globs `dana/plugins/*/manifest.json`)
— there is no separate registration list to edit.

## 2. `manifest.json`

```json
{
  "name": "your_plugin",
  "version": "1.0",
  "entry_point": "engine.py",
  "tools": [
    {
      "id": "your_tool_id",
      "function": "your_python_function",
      "description_en": "One paragraph the LLM sees when deciding to call this tool.",
      "description_fa": "",
      "parameters": [
        {
          "name": "some_arg",
          "type": "number",
          "required": true,
          "description_en": "What this argument controls."
        }
      ],
      "aliases_en": { "_intent": ["do the thing", "your_tool_id"] },
      "aliases_fa": { "_intent": [] }
    }
  ]
}
```

Field notes:

- `name` — plugin id; also used to namespace the imported module as
  `dana.plugins.<name>.<entry_point stem>` in `sys.modules`.
- `entry_point` — defaults to `engine.py` if omitted.
- Each `tools[]` entry needs a globally-unique `id` and a `function` name that
  exists as a callable in the entry-point module. If `function` is omitted it
  falls back to `id`.
- `parameters[]` mirrors `dana.tools.schema.ToolParameterSpec`: `name`,
  `type` (`string`/`number`/`boolean`/...), `required`, optional `enum`,
  and bilingual `description_en`/`description_fa`.
- `aliases_en`/`aliases_fa` under an `_intent` key list example phrases used
  for fuzzy voice-command matching (mailroom routing) before falling back to
  full LLM tool selection.

A malformed manifest, a missing entry point, or an import error in
`engine.py` causes **that one plugin** to be skipped with a logged warning —
it will not take down `load_all_plugins()` for the rest of the tool registry.

## 3. `engine.py`

Write one plain function per declared tool. There's no required base class or
decorator — the manifest's `function` name is resolved with `getattr` against
the loaded module:

```python
# dana/plugins/your_plugin/engine.py
from __future__ import annotations


def your_python_function(some_arg: float) -> dict:
    """Do the thing; return a JSON-serializable result dict."""
    return {"ok": True, "result": some_arg * 2}
```

Guidelines drawn from `dana/plugins/freecad/engine.py`:

- **No COM/mouse/pixel actuation.** If your plugin drives an external
  application, prefer a scriptable CLI/API surface (FreeCAD's plugin shells
  out to `FreeCADCmd` via `subprocess`) over UI automation. If you must touch
  a window, reuse the zero-focus primitives in `dana/tools/os_control.py`
  (`move_window_no_activate`, `get_secondary_monitor`) — see
  [`architecture.md` §7](architecture.md#7-zero-focus-multi-monitor-workspace).
  Never call `set_foreground_window()` from a plugin unless the user
  explicitly requested a foreground action.
- **Return plain, JSON-serializable dicts.** These flow back through the tool
  broker into the LLM's tool-result turn.
- **Fail loudly to your own caller, not the loader.** Catch expected failure
  modes (missing binary, bad args) inside your function and return a
  structured `{"ok": False, "error": "..."}` rather than letting exceptions
  propagate — a raised exception here surfaces as a generic tool-execution
  error to the agent loop instead of an actionable message.

## 4. Wiring into the Feature Manager (optional but recommended)

If your plugin should be independently toggle-able (e.g. because it depends
on optional third-party software), add a `Feature` entry in
`dana/features/feature_manager.py`'s `FEATURES` dict:

```python
Feature(
    id="your_plugin",
    label="Your Plugin",
    tool_ids=("your_tool_id",),  # every tool id from your manifest
),
```

Optionally extend `_detect_default_state()` with a runtime probe (e.g. "is
the vendor CLI on PATH") so the feature defaults to enabled only when it can
actually work — mirroring `detect_freecadcmd()` for the `freecad` feature.
Once registered, your plugin's tools are automatically included in
`apply_feature_gating()` and the live enable/disable toggle in the Settings
UI, with no further code changes. See
[`safety_and_hitl.md`](safety_and_hitl.md) for how the `os_actuator` feature
also gates `DANA_OS_DRY_RUN` — the same pattern is available to any plugin
whose tools should be dry-run-able before going live.

## 5. Testing your plugin

- Unit-test your `engine.py` functions directly like any other module —
  no special harness is required.
- To verify discovery/loading end to end, call
  `dana.plugins.plugin_manager.load_all_plugins(force_refresh=True)` in a
  test and assert your tool id shows up in the returned `(ToolSpec, callable)`
  pairs. `force_refresh=True` bypasses the process-wide plugin cache so your
  test doesn't depend on import order.
- Run the full suite before submitting: `pytest tests/` (see
  [`setup.md`](setup.md#running-tests)).
