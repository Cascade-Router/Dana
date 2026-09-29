# Dānā — an agentic CAD engineer that turns sketches into FreeCAD models

[![Build](https://github.com/Cascade-Router/Dana/actions/workflows/build.yml/badge.svg)](https://github.com/Cascade-Router/Dana/actions/workflows/build.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

Dānā is an open-source, multi-step LLM agent that designs mechanical parts. You describe a part, or hand it a drawing. It plans the build, drives a real FreeCAD kernel through ~90 typed tools, checks its own geometry, and exports a standalone FreeCAD macro that rebuilds the part without the agent.

It began as a commercial product and is now published as an engineering portfolio piece. The README focuses on the four engineering problems that took the most work, with links to the code that solves each one.

| | |
|---|---|
| **Stack** | Python 3.11 · FastAPI + WebSockets · React/TypeScript (Tauri desktop shell) · FreeCAD (headless `FreeCADCmd`) · Ollama · Jinja2 |
| **Models** | Cloud: DeepSeek (primary), Gemini, OpenRouter, Groq · Local: `qwen2.5-coder:14b` (tools), `qwen2.5vl:7b` (vision) |
| **Tests** | 1,000+ pytest cases, blocking in CI, no live model or GPU needed |

---

## 1. Deterministic agentic routing across cloud and local models

**Problem.** A multi-step CAD session mixes very different turns. Some make precise geometry calls, where a wrong argument corrupts the model. Others are cheap bookkeeping, such as a catalog lookup. Letting "whatever model is configured" handle every turn wasted money and let small local models slip wrong parameters into the geometry.

**Solution.** Each turn's model chain is a **pure function of config plus turn state**, with no randomness and no LLM-judged routing:

- **Declarative fleet** ([`routing_config.yaml`](routing_config.yaml), parsed by [`dana/core/llm_router.py`](dana/core/llm_router.py)). Each entry is a provider, model, priority, context window and cost. API keys are referenced by env-var name and never stored in the file.
- **Fit-based selection.** An entry is eligible only if `context_window × safety_margin` fits the turn's estimated token count. Eligible entries are tried in priority order, with a configured terminal fallback (local Qwen) always last.
- **Turn classification, in a fixed order of precedence:**
  1. *Full-local override* (`allow_cloud: false`) returns only Ollama entries, with no silent escape to the cloud.
  2. *Admin/introspection turns* (the last tool call was a catalog or discovery lookup) prefer the local model, which saves cloud quota.
  3. *Geometry-precision turns* **exclude** local models entirely. Small local models were confirmed live to hallucinate tool-argument names on these calls.
  4. A *Planning-Phase Cloud Lock* in [`dana/core/react_dispatch.py`](dana/core/react_dispatch.py) pins Turn 0 (`create_plan`) to a cloud model.
- **Failure accounting.** A failed entry is reported (`report_fleet_entry_failure`) and the chain falls through to the next one. The whole router can be removed: without the YAML file, the legacy `.env` resolution runs unchanged.

In practice, DeepSeek handles planning and geometry, local Qwen handles bookkeeping and full-offline mode, and every routing decision can be reproduced from config.

## 2. Multimodal-to-CAD pipeline: image → JSON blueprint → FreeCAD IR → macro

**Problem.** Vision models are good at recognizing *what* is in a drawing and bad at *how big* it is. A 7B VLM returns JSON that is structurally valid but numerically wrong: all-zero boxes, invented part IDs.

**Solution.** A staged pipeline where every step that touches numbers is deterministic:

1. **OCR dimension floor** ([`dana/plugins/vision/ocr_grounding.py`](dana/plugins/vision/ocr_grounding.py)). EasyOCR reads the printed dimensions from each orthographic view. They are passed to the VLM as hard constraints, and they alone determine the bounding box.
2. **Two-pass VLM extraction** ([`dana/plugins/vision/image_analysis.py`](dana/plugins/vision/image_analysis.py)). Pass 1 extracts primitives (box, cylinder, sphere and their dimensions). Pass 2 extracts relationships and joints given pass 1's output. Each pass tries a local model first and falls back to the cloud, and a JSON decode failure moves on to the next provider instead of aborting.
3. **Numerical-integrity gate.** Degenerate dimensions and dangling `parent_id`/`child_id` references fail loudly. A fabricated blueprint never reaches the CAD kernel as `ok: true`.
4. **Universal CAD IR** ([`dana/plugins/freecad/ir.py`](dana/plugins/freecad/ir.py)). One step-dict schema (`box`, `cylinder`, `boolean`, edge ops, …) is shared by three consumers that used to generate FreeCAD script text separately: live tool execution, session replay, and the composite-skill compiler.
5. **Macro export** ([`dana/plugins/freecad/py_export.py`](dana/plugins/freecad/py_export.py) + [`templates/macro_export.py.jinja2`](dana/plugins/freecad/templates/macro_export.py.jinja2)). The session's call log, including boolean cut/union/intersect steps, is rendered into one standalone `.py` macro that rebuilds the part in stock FreeCAD.
6. **Visual self-check** ([`dana/tools/cad_vision.py`](dana/tools/cad_vision.py)). After every mutating CAD call, the viewport is captured headlessly and a VLM reads it back, so the agent checks what it actually built.

No GPU? Set `DANA_VISION_MOCK_FALLBACK=1`. When no vision model is reachable, the pipeline then continues on a clearly flagged placeholder blueprint (`"mocked": true`) instead of stopping. It is off by default because it is fabricated geometry.

## 3. CI/CD with a blocking test pipeline

**Problem.** An earlier deploy broke even though CI was green, because CI only imported a single module. The fix was to make CI run the real code in the same kind of environment it ships to.

**Solution** ([`.github/workflows/build.yml`](.github/workflows/build.yml)):

- **Full pytest suite as a blocking job:** 1,018 tests at the time of the portfolio pivot. It runs on Python 3.11 under `xvfb`, with `--strict-markers` and a per-test timeout. Failures are parsed out of the JUnit XML into GitHub `::error` annotations, so they show on the PR without opening the log.
- **Space smoke test.** It imports `app.py` from the exact payload `deploy/stage_space.sh` ships, with the pinned Gradio version.
- **Frontend build.** It runs the same `tsc && vite build` that Vercel runs.
- **Gated deploy.** The Hugging Face sync runs on `workflow_run` only after Build succeeds.
- **Headless environment isolation** ([`tests/conftest.py`](tests/conftest.py)). Autouse fixtures sandbox each test's session and OS-tool directories under `tmp_path`, and replace TTS/audio hardware calls with fakes. They block mid-test `.env` reloads (a real leak that once overrode test env vars) and force auto-approve and context distillation off. `DANA_HEADLESS=true` and `DANA_OS_DRY_RUN=1` keep FreeCAD's GUI and physical input out of CI.
- **No live dependencies.** LLM and VLM providers are patched at the call site, and hardware calls (TTS, audio, OS input) go to fakes or dry-run mode. The suite needs no model server, GPU, or API key.

## 4. Context management: 2,000-character rolling core memory

**Problem.** The agent keeps a persistent "core memory" of user preferences, project constraints and learned workflows. It is injected into *every* system prompt and survives restarts. Nothing limited its size, so over weeks of use it would grow without limit into every turn's context.

**Solution** ([`dana/plugins/memory/core_memory.py`](dana/plugins/memory/core_memory.py)):

- **Disk-backed.** Memory is a flat JSON key→value store under the agent workspace. Corrupt or foreign content reads back as "no memory" instead of crashing a turn.
- **Rolling FIFO eviction.** Once the *rendered* block exceeds 2,000 characters, the least recently *written* sections are evicted first. Write order is kept on disk on purpose: sorting keys would have silently turned FIFO into alphabetical eviction.
- **Safety rules.** It never evicts down to zero sections, and it re-applies the cap at render time for files written before the cap existed.
- **Live view.** Updates are broadcast over WebSocket to the UI's Memory Viewer.

---

## Architecture

```text
 React/Tauri UI ──WebSocket /ws/chat──▶ FastAPI (dana/api/server.py)
                                             │
                                  ReAct loop (dana/core/react_dispatch.py)
                                  plan → tool call → observe → verify
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
     LLM router (llm_router.py)     Tool registry (~90 tools)        Core memory (2k, FIFO)
     DeepSeek / Gemini / OpenRouter  CAD · vision · web · OS · skills  + session call log
     / Groq  →  local Qwen fallback          │
                                             ▼
                          FreeCAD plugin (dana/plugins/freecad/)
                          Universal IR → FreeCADCmd (headless)
                          → .FCStd session doc · STEP/STL/URDF · .py macro
```

## Run it locally

Requires Python 3.11, [FreeCAD](https://www.freecad.org/downloads.php) 1.0+, [Ollama](https://ollama.com/), and Node 20+ for the UI.

```bash
git clone https://github.com/Cascade-Router/Dana.git && cd Dana
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                                   # OLLAMA_URL defaults to http://localhost:11434
cp routing_config.yaml.example routing_config.yaml     # optional: enables the deterministic router

ollama pull qwen2.5-coder:14b                          # local tool-calling fallback
ollama pull qwen2.5vl:7b                               # local vision (or set DANA_VISION_MOCK_FALLBACK=1)

(cd frontend && npm install)
./launch_dana.sh                                       # Windows: powershell -File launch_dana.ps1
```

With no cloud keys set, Dānā runs fully locally on Ollama. Add `DEEPSEEK_API_KEY` (or another provider's key) to `.env` to enable cloud routing.

Run the test suite (no GPU, model server or API key needed):

```bash
DANA_HEADLESS=true DANA_OS_DRY_RUN=1 python -m pytest
```

A two-minute walkthrough of the pipeline is scripted in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

## Repository map

| Path | What lives there |
|---|---|
| `dana/api/` | FastAPI server, WebSocket chat protocol, sessions, model registry |
| `dana/core/` | ReAct loop, LLM router, model providers, context management |
| `dana/plugins/freecad/` | FreeCAD engine, universal IR, Jinja2 templates, macro/TechDraw export |
| `dana/plugins/vision/` | OCR grounding and two-pass blueprint extraction |
| `dana/plugins/memory/` | Persistent core memory |
| `frontend/` | React + TypeScript UI (Tauri desktop shell) |
| `tests/` | pytest suite (mirrors the `dana/` layout) |
| `docs/` | Architecture notes and design write-ups ([`ARCHITECTURE.md`](ARCHITECTURE.md)) |

## License

[AGPL-3.0](LICENSE). See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).
