---
title: Dana AI Copilot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Dānā — AI Co-Pilot

**Dānā** is a modular, multimodal AI co-pilot for CAD/engineering workstations: a
ReAct-style tool broker, a zero-touch dynamic plugin system, and a "zero-focus"
actuation layer that lets it drive CAD tools, screenshots, and OS input devices
in the background without stealing window focus.

This Space runs the exact same Gradio UI (`dana/ui/unified_app.py`) as the local
Windows desktop build — there is no separate mock reimplementation anymore.
What differs is which concrete driver `dana.platform.factory` resolves:

| Driver | Local Windows desktop | This Space |
|---|---|---|
| `BaseControlPlane` | `Win32ControlPlane` — real `ctypes` Win32 window actuation | `MockControlPlane` — structured telemetry, no Win32 in this container |
| `BaseCADEngine` | `RealFreeCADEngine` — real `FreeCADCmd` subprocess IPC, `.FCStd` documents | `MockFreeCADEngine` — headless `trimesh` geometry, `.stl` files |

`SPACE_ID` (set automatically on every HF Space) is what selects the mock
drivers — see `dana/platform/factory.py`.

## What's real here

- Tool-call parsing → dispatch → `BaseCADEngine`/`BaseControlPlane` — the same
  code as the desktop build, not a simplified reimplementation.
- 3D mesh generation for `gr.Model3D` — real (`trimesh`-generated STL via
  `MockFreeCADEngine`).
- Blueprint → geometry JSON — real call to `dana.tools.cad_vision.analyze_cad_blueprint`
  (local Ollama VLM first, optional cloud fallback if `DANA_ALLOW_CLOUD_FALLBACK`
  and a provider key are set as Space secrets). Returns an error, not a fake
  result, if neither is reachable.
- Plugin manifest scanning — real scan of `dana/plugins/*/manifest.json`.

## What's mocked here (and why)

- FreeCAD document actuation and Win32 window management — this container has
  no Windows APIs and no `FreeCADCmd` binary, so `dana.platform.mock` stands in
  with clearly-labeled (`"driver": "mock"`) simulated telemetry.

## Layout

```
hf_space/
├── app.py              # thin launcher — imports dana.ui.unified_app
├── requirements.txt     # lean Space deps (no desktop GUI/audio/ML stacks)
└── README.md
```

`dana/` is staged alongside `app.py` by `.github/workflows/deploy_hf.yml` when
this Space is deployed — it is the same package that ships in the main repo,
not a copy maintained separately.

Full source: see the main Dānā repository this Space is deployed from.
