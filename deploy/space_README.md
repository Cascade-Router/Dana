---
title: Dānā
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: agpl-3.0
---

# Dānā — Headless Backend (Space)

This Space runs `app.py`, a plain-chat + mesh-preview smoke test for Dānā's
ReAct loop — mock CAD/control-plane drivers, sandbox-hardened tool registry
(no terminal execution, no code-task/codebase-search tools), and a live
`.stl`/`.glb` mesh preview.

The full 3D viewport, workspace explorer, and execution graph live in the
real client (Tauri desktop app / Vercel-hosted web UI), which talks to this
Space's `"chat"`/`"artifacts"` API endpoints rather than this page's own
chat widget.

License: AGPL-3.0. Source: https://github.com/Cascade-Router/Dana
