---
title: Dana AI Copilot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Dānā — AI Co-Pilot

**Dānā** is a modular, multimodal AI co-pilot for CAD/engineering workstations: a
ReAct-style tool broker, a zero-touch dynamic plugin system, and a "zero-focus"
actuation layer that lets it drive CAD tools, screenshots, and OS input devices
in the background without stealing window focus.

This Space runs the headless FastAPI backend (`dana/api/server.py`) with the
built Tauri/React frontend (`frontend/dist/`) served from the same process —
see the repo root `Dockerfile`. It is the same dispatch code as the local
desktop build; only the platform drivers differ.

| Driver | Local Windows desktop | This Space |
|---|---|---|
| `BaseControlPlane` | `Win32ControlPlane` — real `ctypes` Win32 window actuation | `MockControlPlane` — structured telemetry, no Win32 in this container |
| `BaseCADEngine` | `RealFreeCADEngine` — real `FreeCADCmd` subprocess IPC, `.FCStd` documents | `MockFreeCADEngine` — headless `trimesh` geometry, `.stl` files |

`SPACE_ID` (set automatically on every HF Space) is what selects the mock
drivers — see `dana/platform/factory.py`.

Full source: see the main Dānā repository this Space is deployed from.
