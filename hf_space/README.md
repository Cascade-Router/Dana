---
title: Dana AI Copilot Sandbox
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Dānā — AI Co-Pilot Sandbox

**Dānā** is a modular, multimodal AI co-pilot for CAD/engineering workstations: a
ReAct-style tool broker, a zero-touch dynamic plugin system, and a "zero-focus"
actuation layer that lets it drive CAD tools, screenshots, and OS input devices
in the background without stealing window focus.

This Space is a **portfolio sandbox**, not the production agent. Dānā's real
runtime is a Windows desktop application (PySide6 tray app, FreeCAD/AutoCAD
COM bridges, Win32 SendInput actuators, a LangGraph watchdog compiler). None
of that runs inside a Linux HF container, so this sandbox reproduces the
*architecture and decision logic* — the intent broker, the tool-call schema,
the plugin manifest system, the safety gates — against high-fidelity mocks,
while running real geometry generation (via `trimesh`) and, when an API key
is configured as a Space secret, real multimodal blueprint analysis.

## What's real vs. mocked here

| Feature | In this Space |
|---|---|
| Intent parsing → `ToolCall` dispatch | Real logic, simplified from `dana/tools/broker.py` |
| Tool broker / dispatch log | Real, running against a small demo tool registry |
| 3D mesh generation for `gr.Model3D` | Real (`trimesh`-generated STL) |
| Blueprint → geometry JSON (VLM) | Real multimodal call if `ANTHROPIC_API_KEY` is set as a secret; deterministic heuristic mock otherwise |
| FreeCAD / AutoCAD COM actuation | Mocked — no Windows/FreeCAD binary in this container |
| Win32 SendInput, window focus, clipboard | Mocked |
| Plugin manifest scanning | Real scan of the bundled `hf_sandbox/plugins/` demo manifests |
| Kill-switch (F12), dry-run, HITL gates | Explained + simulated, not wired to real hotkeys |

## Layout

```
hf_space/
├── app.py                    # Gradio entry point (3 tabs)
└── hf_sandbox/
    ├── agent_bridge.py       # ToolCall schema + broker/dispatch simulation
    ├── cad_visualizer.py     # STL mesh generation + blueprint→geometry parsing
    └── architecture_docs.py  # Static content for the architecture explorer tab
```

Full source: [github.com](https://github.com) — see the main Dānā repository for
the production agent this sandbox is derived from.
