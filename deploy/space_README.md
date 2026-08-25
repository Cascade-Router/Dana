---
title: Dana AI Copilot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Dānā — AI Co-Pilot

**Dānā** is a modular, multimodal AI co-pilot for CAD/engineering workstations: a
ReAct-style tool broker, a zero-touch dynamic plugin system, and a "zero-focus"
actuation layer that lets it drive CAD tools, screenshots, and OS input devices
in the background without stealing window focus.

This Space runs a headless `gr.Blocks` app (`app.py`) exposing the same ReAct
tool-calling loop (`dana/api/server.py`'s `_process_user_text`) as a `"chat"`
API endpoint — the Vercel-hosted React app is the real UI, talking to this
Space over `@gradio/client` rather than a WebSocket. It is the same dispatch
code as the local desktop build; only the platform drivers differ.

| Driver | Local Windows desktop | This Space |
|---|---|---|
| `BaseControlPlane` | `Win32ControlPlane` — real `ctypes` Win32 window actuation | `MockControlPlane` — structured telemetry, no Win32 in this container |
| `BaseCADEngine` | `RealFreeCADEngine` — real `FreeCADCmd` subprocess IPC, `.FCStd` documents | `MockFreeCADEngine` — headless `trimesh` geometry, `.stl` files |

`SPACE_ID` (set automatically on every HF Space) is what selects the mock
drivers — see `dana/platform/factory.py`.

## LLM provider (Space secrets/variables)

There's no local Ollama daemon in this container, so `DANA_CLOUD_PRIMARY`
must be set for the ReAct loop's tool-calling turn to reach anywhere at all
— see `dana/core/model_provider.py`'s `tool_calling_provider()`.

| Variable | Required | Purpose |
|---|---|---|
| `DANA_CLOUD_PRIMARY` | Yes | `1` — routes tool-calling to cloud instead of a nonexistent local Ollama |
| `DANA_CLOUD_PROVIDER` | Yes | `openrouter` (or `groq` / `gateway` / `gemini_openai` / `openai`) |
| `OPENROUTER_API_KEY` | Yes, for OpenRouter | From [openrouter.ai/keys](https://openrouter.ai/keys) — `LLM_API_KEY` also works as a generic fallback name |
| `DANA_OPENROUTER_MODEL` | No | Any OpenRouter model id, e.g. `meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-001`, `qwen/qwen-2.5-72b-instruct`. Defaults to `meta-llama/llama-3.3-70b-instruct:free` |
| `OPENROUTER_SITE_URL` | No | Sent as `HTTP-Referer` (OpenRouter's attribution header). Defaults to this Space's own URL if `HF_SPACE_URL` is set |
| `OPENROUTER_APP_TITLE` | No | Sent as `X-Title`. Defaults to `Dana CAD Agent` |

Full source: see the main Dānā repository this Space is deployed from.
