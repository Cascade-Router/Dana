# Dānā Architecture

Dānā is an agentic CAD and manufacturing engineer. A user describes a part or
an assembly in chat; an LLM-driven ReAct loop plans it and builds it in real
FreeCAD (or a headless trimesh stand-in), then verifies, documents and
prepares it for manufacture: STEP/STL/GLB, 2D blueprints, a costed BOM with
tip-over stability, printability checks, slicing to G-code, and a replayable
FreeCAD macro.

This file is also what the agent's `read_system_architecture` tool returns
(capped at 24,000 characters), so it is kept short and current.

## Runtime shape

| Piece | What it is | Entry point |
|---|---|---|
| API server | FastAPI + one WebSocket per chat (`/ws/chat`) | `scripts/launchers/launch_api_server.py` → `dana/api/server.py` |
| Desktop UI | Tauri 2 + React; CAD viewport, Inspector dock, Deliverables bar | `frontend/` |
| Hosted demo | Gradio app on a Hugging Face Space, mock CAD engine; API endpoints `chat`, `artifacts`, `parameters`, `rebuild`, `export_macro` for the Vercel UI | `app.py`, `deploy/` |
| ReAct core | Prompt + tool selection + one LLM step per call | `dana/core/react_dispatch.py` |
| LLM access | Cloud (OpenRouter by default) or local Ollama, OpenAI tool schema | `dana/core/model_provider.py`, `openai_tool_bridge.py`, `llm_router.py` |
| CAD | FreeCAD plugin + Universal CAD IR | `dana/plugins/freecad/` |
| Manufacturing | BOM, stability, printability, slicing, printer dispatch | `dana/plugins/manufacturing/`, `dana/plugins/hardware/` |
| Drivers | Real FreeCAD on Windows, trimesh mock elsewhere | `dana/platform/` (`get_cad_engine`) |

## A chat turn

1. **Turn queue.** Every turn (typed, voice, or a HITL/visual-capture reply)
   runs as a tracked `asyncio.Task` chained per session (`_spawn_turn`), so
   turns never overlap and the socket keeps reading while one runs. Closing
   the socket cancels queued and running turns.
2. **Prompt.** Only `[system_prompt, user_msg]` plus the distilled
   `working_memory.summary` (≤ 1,200 chars, `context_distiller`) — never a
   replay of the whole chat history.
3. **One LLM step** (`next_react_turn`) proposes at most one tool call. The
   reply streams to the UI as `assistant_delta` chunks; the final
   `assistant_message` always replaces the streamed text.
4. **Dispatch.** `dispatch_tool_call` runs in a worker thread
   (`asyncio.to_thread`), so a 60 s FreeCAD call never blocks other sessions.
5. **Loop** until the model answers in text, up to `MAX_REACT_ITERATIONS`
   (30). **Stop** (`abort_turn`) interrupts the LLM stream itself (the HTTP
   read, and `llm_lock` for local models), not just the display.

### Plan-and-Execute state machine

Per session: `planning → executing ⇄ validating → done`
(`_FSM_TABLE`). Geometry tools are gated until `create_plan` succeeds; each
task tracks the tools it expected and the tools that actually ran, and
`mark_task_completed` is refused until they match. A runaway streak of
introspection calls is blocked. These gates keep small and free models on
rails.

### Tool budget

`_TOOL_TOKEN_BUDGET` = 2,000 tokens and `_MAX_ACTIVE_TOOLS` = 30 cap the
optional tools added to a call (`schema_minify.py` sends each description's
first sentence only). In a CAD session the FSM picks the set:

- **planning:** the core tools minus those that need an existing plan;
- **executing a task:** the core tools plus that task's declared tools;
- **plan done:** the core tools (minus the active-plan ones), then the
  read-only checks (`query_topology`, collisions, BOM, blueprint,
  printability) and tools this turn already used, while the budget lasts.

The offline evals (`tests/evals/`) measure every turn: planning 1,476
tokens, executing up to 2,559, plan done at most 2,000 (it was 10,336 when
the whole `freecad_full` domain was offered, more than a local model's
8,192-token context). They fail if a phase grows past that.
Tools are found lazily (`search_tool_catalog` / `load_specific_tool`);
capability domains decay out after a few turns of disuse. Tool retrieval is a
384-d hash embedding on CPU — no neural embedding model.

### Human in the loop

Mutating tools ask for approval unless the user enabled auto-approve.
`ALWAYS_PROMPT_TOOL_IDS` (`execute_freecad_script`,
`modify_existing_freecad_document`, `dispatch_to_printer`) always ask, even
with auto-approve on. `modify_existing_freecad_document` only opens `.FCStd`
files inside Dana's workspace.

## CAD engine

- **One FreeCADCmd subprocess per call**, serialized by the engine lock, run
  through `_run_freecad_script` (with `_RUN_ONCE_WRAPPER`: FreeCADCmd would
  otherwise re-run a failing script as `__main__`).
- **One document per chat:** `freecad_output/sessions/<session_id>/Session_Active.FCStd`.
  Objects are resolved by name (Name, then Label, then case-insensitive).
- **Atomic saves:** a script that saves the session document gets a `.bak`
  copy first; a timeout, a failed run, or a run that leaves the file
  missing/empty restores it.
- **Timeouts:** 60 s per call; 180 s for collision checks, blueprints and
  printability (`DANA_FREECAD_HEAVY_TIMEOUT_S`).
- **Mock engine** (`dana/platform/mock.py`: the Space, CI, non-Windows): one
  trimesh mesh per object under `freecad_output/sessions/<id>/mock/`.
  `query_topology` reports the mesh's smooth regions as faces (same schema;
  its own stable numbering), `check_printability` runs the real
  `build_report` on the mesh, and `validate_assembly_collisions` uses
  manifold3d for overlap volume and clearance. Pipes and helices are real
  sweeps, `position_assembly_part` moves the mesh, and
  `export_assembly_to_urdf` writes per-link meshes with exact mass
  properties. STEP export still needs FreeCAD.
- **GUI:** shown only when `DANA_HEADLESS` is not `true` (the default lock is
  `true`). Dana's FreeCAD window runs with a private `user.cfg` (`-u`) and only
  the PID Dana launched is ever closed or screenshotted.

### Universal CAD IR

`dana/plugins/freecad/ir.py` defines each operation once as a step dict
(`from_args` for live calls, `from_record` for replay) rendered by one Jinja
template (`templates/universal_ir.py.jinja2`). Live execution, the macro
export (`py_export.py`), compiled skills (`skill_compiler.py`) and the
parametric rebuild all use it. Kinds: box, cylinder, pyramid, pipe, helix,
boolean, edge operation (fillet/chamfer), parameter/placement edits, mesh
solidify, pattern, the face-feature composite, and the eight
Sketcher/PartDesign features.

### Zero-token parametric rebuild

`dana/plugins/freecad/rebuild.py`. The session's `CadCallLog` is turned back
into IR steps; the Parameters tab lists their dimensions. A rebuild patches
the steps and renders them once into a **scratch** document, swapped over
`Session_Active.FCStd` only on success — a bad value never touches the live
model. On the mock engine the patched log is replayed through the mock
driver instead. No LLM call either way. Logs containing raw scripts or
assembly tools are reported as unsupported rather than rebuilt with
geometry missing.

## Session persistence

`agent_workspace/data/sessions/<id>.json` (atomic writes) holds the chat,
the distilled summary, the terminal log, and a `state` payload
(`dana/api/session_state.py`): plan, object registry, topology DAG, assembly
state, gates, CAD call log, usage/cost totals. Each save also snapshots the
document to `freecad_output/sessions/<id>.FCStd` with its SHA-256; on resume
a mismatched document is restored from that snapshot or, failing that, the
object list is re-read from the document itself. Approvals, auto-approve,
API keys and suspended HITL turns are never written to disk.

## Hard constraints

- **8 GB VRAM (RTX 2080 reference).** `ollama_num_ctx()` ≤ 8192; local
  requests serialize on `llm_lock`; Ollama models unload (`keep_alive: 0`)
  when the last socket closes. OCR, TTS and tool retrieval stay on CPU; no
  persistent GPU daemons; torch is not in the base `requirements.txt`.
- **Token budget.** No full-history replay; the tool budget above; one tool
  per LLM step; the planning/executing gates.
- **Process isolation.** One FreeCADCmd subprocess per call;
  `DANA_HEADLESS=true` by default; the trimesh mock keeps CI and the Space
  working (mock booleans need `manifold3d`).

## Repository

| Path | Role |
|---|---|
| `dana/api/` | Server, REST routers (sessions, CAD artifacts), session state |
| `dana/core/` | ReAct loop pieces, providers, routing, distillation, telemetry |
| `dana/plugins/` | Lazy plugins: freecad, manufacturing, hardware, planning, memory, os, web, vision, coder |
| `dana/platform/` | CAD/OS drivers (win32, darwin, mock) |
| `dana/tools/` | Tool schema/IR, CAD vision, OS control, misc tools |
| `frontend/` | Tauri/React app |
| `tests/` | pytest suite; `tests/e2e/` Playwright; `tests/evals/` offline agent evals |
| `scripts/ci/` | Boundary and dead-code gates, coverage floor |

## Verifying a change

```bash
DANA_HEADLESS=true DANA_OS_DRY_RUN=1 .venv/Scripts/python -m pytest tests --timeout=300
python scripts/ci/check_boundaries.py
python scripts/ci/check_dead_code_incremental.py --base origin/main
python scripts/run_offline_evals.py   # agent eval scorecard (also part of pytest)
cd frontend && npm run build && npx playwright test
```

Tests marked `e2e` drive real FreeCAD and skip when FreeCADCmd is not found.
