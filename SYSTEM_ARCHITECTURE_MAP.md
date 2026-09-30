# DANA system architecture map

A read-only audit of every git-tracked part of the repository at commit `3a17835` (2026-09-29), updated 2026-09-30 to mark the §3.1 fixes. Each component below is tied to a real file path.

**How "live" is decided.** A module is **live** if a real `import` chain reaches it from a production entry point, not counting tests or docstrings:
- `scripts/launchers/launch_api_server.py`, which runs `uvicorn dana.api.server:app`
- `start_dana.py`
- `app.py`, the Hugging Face Space

A module is **legacy** if only tests, `scripts/`, or other legacy modules import it.

**Recency** figures are commit counts from `git log` since 2026-09-01.

**Tracked vs local.** 462 files are tracked. Several folders visible on disk are gitignored or untracked, and are marked as *local-only*.

---

## 1. Global directory tree

| Path | Tracked | Responsibility |
|---|---|---|
| `dana/` | yes (158 files) | The Python backend: API server, ReAct agent, LLM routing, plugins (FreeCAD, vision, OS, web, planning, memory, coder), tools, audio, and a large legacy layer |
| `frontend/` | yes (95) | React 18 + TypeScript + Vite 5 UI inside a Tauri v2 desktop shell. Also builds for Vercel and runs in "Gradio mode" against the HF Space |
| `website/` | yes (36) | Astro 4 + Tailwind marketing site, deployed to GitHub Pages at `/Dana/` |
| `tests/` | yes (81) | pytest suite (74 Python test files, 1,022 tests at HEAD) plus 2 Playwright specs in `tests/e2e/` that CI does not run |
| `scripts/` | yes (36) | Launchers, CI helpers, diagnostics, manual live tests and demos, and maintenance CLIs |
| `docs/` | yes (16) | Architecture, safety, telemetry, legal and user docs. **No commits since 2026-08-29; most describe removed code** (see §3) |
| `deploy/` | yes (2) | HF Space staging: `stage_space.sh`, `space_README.md` |
| `.github/` | yes (7) | 5 workflows (`build`, `deploy_hf`, `deploy_website`, `release`, `update_models`) and issue templates |
| `assets/` | yes (3) | Brand icon and logo |
| Root files | yes | `app.py` (HF Space Gradio app), `start_dana.py` (dev orchestrator), `launch_dana.ps1`/`.sh` (FreeCAD preflight plus launch), `routing_config.yaml(.example)`, `requirements*.txt`, `pyproject.toml`, `packages.txt` (apt `freecad`), `build_dana.py` + `Dana.spec` (PyInstaller, **broken**), `clean_repo.py`, `introspect_face1.py`, `swarm_mcp.md` (design brief only, no code), `.cursorrules`, `.cursor/` |
| `agent_workspace/` | local-only | Agent sandbox root: `data/core_memory.json`, `data/sessions/*.json`, user skills |
| `exports/`, `freecad_output/`, `captures/`, `logs/`, `urdf_output/` | local-only | Runtime output: macros, `.FCStd`, viewport captures, logs, URDF |
| `memory/`, `dana_memory.enc`, `.shadow_state/` | local-only | Legacy stores: SQLite blackboard, encrypted profile, shadow backups |
| `custom_tools/`, `_offline_archive/` | local-only | Legacy tool mirror and archived old files |
| `rover_urdf/`, `meshes/` | untracked | Sample URDF output (their `.stl` files are ignored) |

Paths referenced by old docs that **do not exist**: `run.py`, `dana/core_agent.py`, `dana/graph/`, `dana/ui/`, `dana/swarm/`, `dana/updater/` (only a stray `__pycache__`), `dana/web/`, `dana_jason_loop/`, `dana_security/`, `execution_jail/`, `tests/evals/`.

---

## 2. Subsystem inventory

### 2.1 API server and transport: `dana/api/` (live)

**Purpose.** A FastAPI app that owns the WebSocket chat loop, human-in-the-loop (HITL) approvals, artifact and mesh serving, sessions, and the settings UIs.

| File | Role |
|---|---|
| `dana/api/server.py` (~2,700 lines) | The app. Chat entry is `WS /ws/chat` (`ws_chat`, ~L2430). It also drives the ReAct loop (`_process_user_text`, `_run_react_loop`, `_execute_and_continue`, `_resolve_react_hitl`, `_finish_turn`). Routes: `GET /api/health`, `/api/plugins`, `/api/config`, `/api/mesh/{token}.stl\|glb\|obj\|urdf`, `/api/audio/{token}.wav`, and static `/api/vision`. The lifespan starts `VoiceService` (L399) |
| `dana/api/sessions.py` | `/api/sessions` list/get/delete/reset. Stores JSON in `agent_workspace/data/sessions/<id>.json` |
| `dana/api/workspace.py` | `/api/workspace/tree\|file\|mounts\|mount` |
| `dana/api/system.py` | `/api/system/env` (masked env viewer/editor) and `/validate` |
| `dana/api/cad.py` | `/api/cad/artifacts`, `/download`, `/print`, `/open-desktop` |
| `dana/api/models.py` | `/api/models/matrix`, `/preferences`, `/{id}/clear-error` |
| `dana/api/artifacts_registry.py` | In-process artifact list (also used by `app.py`) |

**WebSocket protocol.**
- **Client → server:** chat `{text, attachments?, include_desktop_context?}` (no `type` field), `hitl_response`, `set_auto_approve`, `abort_turn`, `update_context`, `update_secrets`, `canvas_selection`, `voice_control`, `audio_playback_complete`, `visual_capture_response`, `export_python_script`.
- **Server → client:** `ready`, `user_message`, `assistant_message`, `dag_node_start/complete`, `tool_dispatch_start/end`, `hitl_approval_required`, `visual_capture_request`, `usage_update`, `plan_update`, `memory_update`, `topology_update`, `camera_animate`, `python_script_exported`, `assistant_audio`, `voice_state`, `server_log`.
- The frontend never sends `export_python_script` or `visual_capture_response`, and never handles their replies.

### 2.2 ReAct agent core: `dana/core/` (live)

**Purpose.** A UI-agnostic multi-step agent. It builds the prompt, picks tools, calls the LLM, gates each tool call, and dispatches it.

| File | Role |
|---|---|
| `dana/core/react_dispatch.py` (~9,350 lines) | **The agent core:**<br>- `TOOL_HANDLERS` (~L4950): 83 native handlers; with plugin tools, ~90 at import<br>- capability domains `_CAPABILITY_TOOL_IDS` (~L5600)<br>- `build_system_prompt` (~L6960)<br>- `next_react_turn` (~L8060)<br>- `_call_llm_once` (~L7415)<br>- `_llm_tools_schema` (~L6050)<br>- `dispatch_tool_call` (~L8880)<br>- `is_mutating_tool` (~L5080)<br>- plan FSM, topology DAG, and the measurement and collision gates |
| `dana/core/model_provider.py` | `ModelProvider.complete_with_tool_calls` (~L914): hands off to the router, falls back from cloud to Ollama, handles vision (`complete_vision`) and cost |
| `dana/core/openai_tool_bridge.py` | HTTP bridge for OpenAI-compatible APIs plus native Ollama `/api/chat` |
| `dana/core/llm_router.py` + `routing_config.py` | Deterministic fleet chain built from `routing_config.yaml` (`resolve_chain`, `select_fleet_chain`, `cloud_allowed`) |
| `dana/core/model_registry.py`, `pricing.py`, `dana/data/models_registry.json` | Model catalog for the UI, runtime error state, and price table. The weekly `update_models.yml` workflow opens a PR to sync the catalog |
| `dana/core/tool_retrieval.py`, `tool_retriever.py` | Top-K tool narrowing (`narrow_tool_ids_by_query`) with an optional inverted-index backend |
| `dana/core/skill_loader.py` | User Python skills in `agent_workspace/skills/*.py`. They run via in-process `exec()` with no sandbox or timeout; every skill is HITL-gated |
| `dana/core/telemetry.py` | Three-tier event logger (INFO/DEBUG/TRACE) |
| `dana/config.py`, `paths.py`, `session_context.py`, `logging.py`, `sanitize.py`, `perf.py`, `system_health.py`, `stdio_boot.py`, `architecture.py` | Support modules: `MAX_REACT_ITERATIONS` = 30, canonical paths, the ContextVar session id, logs with secret redaction, metrics, `llm_lock` and RAM checks, the pythonw stdio guard, and the `read_system_architecture` tool |

### 2.3 LLM routing (live, with one legacy twin)

- **Live:** `dana/core/llm_router.py`. Declarative fleet: DeepSeek (priority 1), Gemini, OpenRouter free, Groq free, and local `ollama_local` `qwen2.5-coder:14b` as terminal fallback. It supports full-local override, admin-turn local preference, and excluding local models from geometry turns. A **Planning-Phase Cloud Lock** in `react_dispatch._call_llm_once` forces Turn 0 to `openrouter` (or to `ollama` when `allow_cloud: false`).
- **Paths that bypass the router:** planning turns, and turns latched to the cloud by the 85%-of-`num_ctx` context handoff in `server._run_react_loop`.
- **Legacy:** `dana/cascade_router.py`, the old MoA qwen/vision/DeepSeek router. It's reached only from `moa_tool_shim.py` (unimported) and `os_control._vision_describe` (unreachable). It lazily imports the deleted `dana.agentic` and `dana.management`.

### 2.4 Plugins and tools: `dana/plugins/`, `dana/tools/`, `dana/platform/`

**Purpose.** Everything the agent can do. Plugins are discovered from `dana/plugins/*/manifest.json` by `plugin_manager.py`. `react_dispatch.refresh_plugin_tools()` merges them into capability domains, **skipping any id that already has a native handler**. Only `freecad` and `coder_plugin` have manifests; the other plugin folders are imported statically as native handlers.

**Capability domains** (`react_dispatch.py`):
- **core** (18, always on): planning, memory, skills, capability loading, `take_canvas_screenshot`.
- **`freecad` / `freecad_full`** (49 native + 2 manifest tools), grouped as:
  - primitives
  - booleans and edge ops
  - sketch/PartDesign
  - assembly, mates and kinematic joints
  - inspection
  - export (STEP/STL, TechDraw, URDF, sim wrapper)
  - standard parts
  - image→3D
  - control-plane window tools
- **`freecad_essential`** (23): the default when a CAD-looking prompt or the UI's `cad` plugin is active.
- **`os_tools`** (11), **`web_tools`** (2), **`vision_tools`** (3, including `execute_vision_analysis` for the live CAD viewport), **`software_engineering`** (4, coder plugin), and **`user_skills`** (dynamic).

**Tool catalog.** `dana/tools/tools.json` has 83 entries, each with a handler in `TOOL_HANDLERS`. The 44 legacy ids with no handler and the duplicate `read_system_architecture` were removed (§3.1 item 7). `search_tool_catalog` and `load_specific_tool` only offer tools that dispatch can run, so non-dispatchable registry entries (the `dana/tools/general/*.py` hot-loads) stay hidden from the agent.

| Domain | Paths | Purpose |
|---|---|---|
| **FreeCAD CAD kernel** | `dana/plugins/freecad/`:<br>- `engine.py` (~4,800 lines; each op is a fresh `FreeCADCmd` subprocess on the session `.FCStd`)<br>- `ir.py` + `templates/universal_ir.py.jinja2` (shared step IR)<br>- `call_log.py`<br>- `py_export.py` + `templates/macro_export.py.jinja2` (standalone macro)<br>- `techdraw_export.py` (2D PDF blueprints)<br>- `skill_compiler.py` (plan → reusable skill)<br>- `standard_parts.py` + `fasteners_bootstrap.py` (auto-installs the Fasteners workbench)<br>- `mesh_ops.py` (mesh → solid)<br>- `error_digest.py` (OCC traceback → actionable error)<br>- `engineering_standards.py` | Design, inspect, assemble and export parts. This is the most actively developed area (15 commits since 09-01) |
| **Vision → CAD** | `dana/plugins/vision/image_analysis.py`, `ocr_grounding.py`; `dana/tools/cad_vision.py` | Two-pass VLM blueprint extraction with an EasyOCR dimension floor and an integrity gate. Also headless viewport capture and visual QA after CAD mutations. The opt-in mock fallback lives in `react_dispatch._tool_analyze_reference_design` |
| **Robotics / simulation** | `dana/tools/urdf_builder.py`, `sim_wrapper_generator.py`, `geometry_analyzer.py`; `engine.define_kinematic_joint` (~L2728), `engine.export_assembly_to_urdf` (~L3160) | URDF from FreeCAD assemblies (fixed/revolute/continuous/prismatic joints, SI units, inertia from volume × density). Simulator loader scripts for **`isaac_sim`** (`omni.isaac.kit`, URDF importer fallbacks, a torch reset/step skeleton), **`gazebo`**, **`ros2`** and **`webots`**. There is **no PyBullet** support. These tools live in the freecad domain; there is no separate robotics domain |
| **Image → 3D** | `dana/tools/image_to_3d.py` | Tripo3D or Meshy REST APIs (API key needed), falling back to HF Spaces TripoSR → CRM → zero123plus |
| **OS / sandboxed workspace** | `dana/plugins/os/`: `file_system.py`, `process_manager.py` (`execute_terminal_command`, **`shell=True`**), `background_services.py`, `desktop_vision.py` (mss capture + VLM) | Agent file I/O confined to `agent_workspace/`, shell commands and services behind HITL, and desktop screenshot Q&A |
| **Win32 control plane** | `dana/tools/os_control.py`, `dana/platform/{base,factory,win32,mock,darwin}.py`, `dana/vision/uia_provider.py` | `os_control` has ctypes `SendInput` keyboard/mouse, clipboard, window moves, and PrintWindow capture. **Only window moves and capture are reachable** (`resync_workspace`, FreeCAD window placement, `cad_vision`); every keystroke/mouse/clipboard function is unreachable from the agent. `platform/factory.py` picks `win32.py` or `mock.py` (~2,200 lines of headless trimesh geometry for CI and the HF Space). `darwin.py` is a stub |
| **Web research** | `dana/plugins/web/research.py` | `search_web` via DuckDuckGo `ddgs` (no key); `read_webpage` via httpx + BeautifulSoup, capped at 10k characters |
| **Coder plugin** | `dana/plugins/coder_plugin/engine.py` + `manifest.json` | Wraps Aider: `search_codebase`, `analyze_codebase`, `run_verification_command` (allowlisted), `execute_code_task`. `app.py` removes it on the HF Space |
| **Planning** | `dana/plugins/planning/task_board.py` | `create_plan`, `mark_task_completed`, `insert_task`, `cancel_*`, and the plan FSM that restricts which tools each phase may call |
| **Tool infrastructure** | `dana/tools/registry.py` (in-memory registry with a hash-embedding index, optional FAISS), `schema.py` (ToolSpec/ToolCall IR), `schema_minify.py` | Registry, schemas, and token-budgeted tool lists (2,000 tokens / 30 tools max) |
| **Safety gates** | `dana/security/dry_run.py` (`DANA_OS_DRY_RUN`), `react_dispatch.is_mutating_tool` (fails closed), `server._HITL_ALWAYS_APPROVED_TOOLS`, `dana/middleware/kill_switch.py` | Dry-run is honored in ~20 FreeCAD sites and in OS control. HITL applies to every mutating tool, with bypasses for parametric CAD CRUD, a per-session allowlist, the UI auto-approve toggle, and `DANA_AUTO_APPROVE`. The F12 kill-switch **listener is never started** by the server |

### 2.5 Memory and context management

| Layer | Path | Storage and limits | Status |
|---|---|---|---|
| Core memory (persistent, in every prompt) | `dana/plugins/memory/core_memory.py` | `agent_workspace/data/core_memory.json`; 2,000-char rendered cap with FIFO eviction by write order | **Live** |
| Working memory (rolling per-session summary) | `dana/core/context_distiller.py` | `session["working_memory"]`, made by local Ollama after each turn; 150 words / 1,200 chars; disable with `DANA_CONTEXT_DISTILL=0` | **Live** (`server._finish_turn`) |
| Per-call context pruning | `dana/core/context_manager.py` | Keeps 2 images and 3 recent tool outputs (truncated to 200+200 chars); JSON-aware compression and trajectory compaction for Ollama | **Live** |
| Session transcript | `dana/api/sessions.py` | `agent_workspace/data/sessions/<id>.json`. **Each turn starts from a fresh `messages` list**; continuity comes from working plus core memory | **Live** |
| SQLite blackboard | `dana/memory/blackboard.py` | `memory/blackboard.db` | Legacy |
| Episodic store | `dana/memory/store.py` | `dana/memory/memory.db` with per-row TTL | Legacy |
| Chroma codebase RAG | `dana/memory/vault.py`, `vector_sync.py`, `compressor.py` | `.dana/vault/` (missing locally) | Legacy |
| Encrypted profile vault | `dana/secure_memory.py`, `dana/vault_service.py` (TCP 127.0.0.1:47475), `scripts/reset_vault.py` | `dana_memory.enc`; PBKDF2 with a **static salt** | Legacy |
| Cross-thread globals | `dana/core/shared_state.py` | Extracted from the deleted `core_agent` | Legacy. Importable since the §3.1 fix (`spatial_context` moved into the package), but no longer on the live STT/TTS path |

### 2.6 Voice and audio: `dana/audio/`, `dana/services/`

| Piece | Path | Status |
|---|---|---|
| Push-to-talk service | `dana/services/voice_service.py`, started in the `server.py` lifespan; driven by `voice_control` / `voice_state` WebSocket messages | Live. Failures are logged (§3.1 item 1, resolved) |
| STT | `dana/audio/stt.py`: transformers `distil-whisper/distil-small.en`, corrected by `dana/tools/stt_corrector.py` + `vocabulary.json` | Live (§3.1 item 1, resolved) |
| TTS | `dana/audio/multi_voice_tts.py`: Piper (`DANA_PIPER_VOICE`, from `tts_models/*.onnx`) → pyttsx3 → silence. `server._speak_reply` sends `assistant_audio` after **every** final reply | Live since the §3.1 item 1 fix (it shared the broken import). Piper is skipped locally because `tts_models/` is missing |
| Wake word, continuous mic, VAD, barge-in | `dana/audio/mic_input.py`, `vad_consumer.py` (Silero), `noise_floor.py`, `dc_blocker.py`, `tts_worker.py` (~3,300 lines), `tts_manager.py`; openWakeWord is configured in `dana/core/constants.py` | Legacy. No tracked file imports `openwakeword`; its predict loop lived in the deleted `core_agent.py` |
| Idle governor | `dana/middleware/idle_monitor.py` | Legacy, never started |

### 2.7 Frontend: `frontend/`

| Area | Paths | Purpose |
|---|---|---|
| Entry and routing | `src/main.tsx` (`#/orb` → `OrbOverlay`, `#/plugin/:id` → `windows/PluginWindowApp`, anything else → `App`), `src/App.tsx` | Main shell |
| Transport | `src/lib/useChatSocket.ts` (WebSocket), `useGradioChat.ts` + `gradioChatClient.ts` (HF Space mode), `useChat.ts` (picks one), `apiBase.ts` | One chat hook shape with two backends |
| CAD plugin | `src/plugins/registry.ts` → `CadPlugin`: `Viewer3D.tsx` (R3F/three: STL/GLB/URDF loaders), `DAGMonitor` (ReactFlow topology), `InspectorDock`, `BlueprintViewer`, `CadToolbar`, `MeshHistoryPicker` | 3D viewer, lineage graph, blueprints, exports |
| Other plugins | `WorkspacePlugin`, `CoderPlugin` | Workspace browser; Aider code-task view |
| Panels | `ChatPanel`, `ChatSidebar`, `PlanChecklist`, `MemoryViewer`, `CostBar`, `ModelRegistryPanel`, `EnvViewerWidget`, `ConfigViewer`, `TerminalDrawer`, `WebDemoBanner` | Chat, sessions, plan, memory, cost, models, env, logs |
| Secrets | `src/secrets/SecretsContext.tsx` | API keys kept in tauri-plugin-store (**plaintext JSON** in the app data dir) and pushed via `update_secrets` |
| Multi-window | `src/windows/windowSync.ts` (`dana://sync`, `child-ready`, `cad-select`, `orb-activate`), `OrbOverlay` + `AssistiveOrb` | The main window owns the socket; child windows mirror its state. Plan, DAG and log are **not** synced to children |
| Native shell | `src-tauri/` (`tauri.conf.json`: `main` window plus a transparent always-on-top `orb` window; CSP allows only localhost:8000; `src/lib.rs` hosts the webview and stops the backend on close) | Desktop app |

### 2.8 Hosted demo, website, CI/CD, packaging

- **HF Space.** `app.py` is a Gradio Blocks app that drives the real `server._process_user_text` through a duck-typed `_GradioSocket`, auto-approving HITL. `_harden_tool_registry()` removes shell and coder tools. It's staged by `deploy/stage_space.sh` and deployed by `deploy_hf.yml`, which runs only after `Build` succeeds on main.
- **Website.** `website/`, Astro. `LiveHfSimulator` iframes the Space, and the global chat bar (`src/utils/hf_api.ts`) calls the Space's `chat` endpoint through `@gradio/client`. `deploy_website.yml` publishes to GitHub Pages.
- **CI.** `.github/workflows/build.yml` has four jobs:
  - `cross-platform`: import check for `setup_startup` only
  - `space-smoke`: imports `app.py` from the staged payload
  - `tests`: full pytest, blocking, Python 3.11, xvfb
  - `frontend`: `npm ci && npm run build`

  Not in CI: Playwright e2e, the Tauri/Cargo build, and the website build (it has its own workflow).
- **Release.** `release.yml`, on `v*` tags, runs tests, builds `dana-engine-<tag>.tar.gz` plus `latest.json`, and publishes a GitHub release. Its comment references the missing `dana/updater/manifest.py`.
- **Packaging.** `build_dana.py` / `Dana.spec` (PyInstaller) build the backend from `scripts/launchers/launch_api_server.py` (§3.1 item 3, resolved).

---

## 3. The "ignored" surface area

### 3.1 Defects found during this audit

All seven items were **resolved on 2026-09-30**. Their original descriptions are kept so the history stays readable.

1. ✅ **Resolved: voice STT (and TTS) never worked in a real launch.** Fixed in `16d72c3`.
   - *Was:* `dana/audio/stt.py` imported `dana.core.shared_state`, whose bare `from spatial_context import …` resolved only through `tests/conftest.py`'s `sys.path` entry. Any `import dana.audio…`, including TTS's `multi_voice_tts`, failed under the launcher, and `VoiceService` swallowed the error.
   - *Now:* `spatial_context` lives at `dana/core/spatial_context.py`. STT and the TTS text sanitizers use `dana/audio/speech_state.py`, so the server no longer loads the legacy `shared_state` stack. `VoiceService` logs every failure path. `tests/services/test_voice_service.py` imports the audio modules in a subprocess with only the repo root on `PYTHONPATH`.
2. ✅ **Resolved: `setup_startup` would overwrite the working launcher.** Fixed in `6c690f3`.
   - *Was:* `entry_script()` returned the deleted `run.py`, and `write_start_bat()` rewrote the tracked `scripts/launchers/start_dana.bat` to launch it.
   - *Now:* `ensure_start_bat()` registers the tracked launcher without modifying it. macOS/Linux autostart runs `scripts/launchers/launch_api_server.py`. Covered by `tests/tools/test_setup_startup.py`.
3. ✅ **Resolved: desktop packaging was dead.** Fixed in `6c690f3` and in the commit that updates this map.
   - *Was:* `build_dana.py` and `Dana.spec` used `run.py` as the entry point with `dana.core_agent` / `dana.ui` hidden imports, and `pyproject.toml`'s ruff and mypy config referenced `dana/core_agent.py`.
   - *Now:* both build files use `launch_api_server.py`, collect all `dana` and `uvicorn` submodules, and bundle `dana/`'s data files. A build from `Dana.spec` produced a `Dana.exe` that served `/api/health` and loaded both plugin manifests. The `core_agent` entries are gone from `pyproject.toml`.
4. ✅ **Resolved: router fallback trap.** Fixed in `e26b98d`.
   - *Was:* with no valid `routing_config.yaml` and `DANA_CLOUD_PROVIDER` unset, `complete_with_tool_calls` resolved to `cloud_provider_name()`'s `"gemini"` default, which the tool-calling bridge rejects with `NotImplementedError`.
   - *Now:* it resolves through `tool_calling_provider()`: local Ollama, or a tool-calling-safe cloud provider when `DANA_CLOUD_PRIMARY` is on. Covered by two tests in `tests/test_llm_router.py`.
5. ✅ **Resolved: Tauri close left the backend running.** Fixed in the commit that marks this item resolved.
   - *Was:* two bugs.
     - `src-tauri/src/lib.rs` identified the repo by the gitignored root `start_dana.bat`, so on a fresh clone the teardown never found the stop script.
     - `scripts/launchers/stop_dana.bat` matched `python.exe` backends only by `run.py` or `-m dana`, so a backend started as `python.exe scripts/launchers/launch_api_server.py` (for example by `start_dana.py`) survived. Only `pythonw.exe` backends matched, through the broader repo-path filter.
   - *Now:* `lib.rs` locates the repo by the tracked `scripts/launchers/stop_dana.vbs` and runs it directly, with Rust unit tests for the lookup. `stop_dana.bat` matches `launch_api_server.py`.
   - Verified: running the stop script killed a live `python.exe` backend. The previous script left it running.
   - macOS/Linux don't use this path: `launch_dana.sh` and `start_dana.py` already stop the backend when the app exits.
6. ✅ **Resolved: website chat client targeted endpoints that don't exist.** Fixed in the commit that marks this item resolved.
   - *Was:* `website/src/utils/hf_api.ts` POSTed `{data: [prompt]}` to `/api/predict` and `/run/predict`. Both return 404 on the live Space, because `app.py` is a `gr.Blocks` app exposing only the named endpoints `chat` and `artifacts`.
   - *Now:* the website calls the Space through `@gradio/client`, the same contract `frontend/src/lib/gradioChatClient.ts` uses: `predict("/chat", { message })`, with the reply in `data[0]`. One client per page keeps the Space-side session. No backend route was added.
   - Verified against the live Space: `view_api()` lists `/chat` with a `message` parameter and a Textbox as its first return value. The website builds. A full chat turn wasn't sent, to avoid spending the Space's LLM credits.
7. ✅ **Resolved: 44 dead tool ids in `tools.json`.** Fixed in the same commit.
   - *Was:* `search_tool_catalog` offered them and `load_specific_tool` reported them loaded, but dispatch rejected them.
   - *Now:* they're removed, along with the duplicate `read_system_architecture` entry (the loader already used the later, `read_only` one). Both catalog tools also check `TOOL_HANDLERS`, which is read at call time, so plugin tools and user skills stay discoverable.
   - Follow-up, also resolved: `execute_vision_analysis` had a handler but belonged to no capability domain, so the agent could never call it. It had been triggered by the regex dispatcher that `e0b69ea` replaced, and was never added to a domain afterwards.
     - It's now in `vision_tools`, a read-only VLM domain that loads on every provider (`freecad_full` is blocked on Ollama).
     - `tests/plugins/vision/test_image_analysis.py` also asserts that every `tools.json` id is reachable from core or some domain.
     - The tool still needs a visible FreeCAD window, so it fails when `DANA_HEADLESS=true`.

### 3.2 Present but sidelined (no commits since 2026-09-01)

| Area | Paths | State |
|---|---|---|
| OS / Win32 actuation | `dana/tools/os_control.py`, `dana/platform/darwin.py`, `dana/vision/uia_provider.py`, `overlay.py`, `dana/tools/rate_limiter.py` | Keyboard, mouse, clipboard and UIA code exists and has tests, but can't be reached from the agent. `rate_limiter.py` has no users (os_control has its own). macOS is a stub |
| OS plugin, web research, coder plugin | `dana/plugins/os/`, `dana/plugins/web/`, `dana/plugins/coder_plugin/`, `plugin_manager.py` | Live and reachable, but untouched since August |
| Safety middleware | `dana/middleware/` (`kill_switch.py`, `idle_monitor.py`, `json_schema_retry.py`, `scratchpad.py`) | The kill switch is imported but its listener is never started. The other three are orphans. Only `toast_notify.py` is live (FreeCAD update toasts) |
| Legacy memory stack | `dana/memory/*`, `secure_memory.py`, `vault_service.py`, `shared_state.py`, `scripts/reset_vault.py`, `scripts/ingest.py`, `dana/tools/task_queue.py` | A whole earlier generation (SQLite blackboard, Chroma RAG, encrypted vault, `execution_jail` task queue), replaced by core memory plus the distiller. `task_queue.drain()` raises `NotImplementedError`. **`vault_service.py` monkeypatches `subprocess.Popen` globally when imported** |
| Legacy voice pipeline | `dana/audio/{mic_input,vad_consumer,noise_floor,tts_worker,tts_manager}.py`, openWakeWord config | Wake word and continuous listening are gone with `core_agent.py`. Only push-to-talk and TTS remain |
| Legacy MoA / LangGraph stack | `dana/cascade_router.py`, `moa_tool_shim.py`, `llm_client.py`, `llm_schemas.py`, `handoff.py`, `schema.py` (the only langgraph import), `dana/telemetry.py`, `bug_tracker.py`, `settings.py`, `vision_tools.py`, `tracker.py` (YOLO), `dana/prompts/` | Orphaned. They import deleted modules (`dana.agentic`, `dana.management`). The corresponding `requirements.txt` pins (langgraph/langchain, chromadb, sentence-transformers, ultralytics, customtkinter, pystray, pyinstaller) are still installed on every CI run and on the Space |
| Tool Forge / dynamic tools | `dana/tools/promotion.py`, `dynamic/generated_tools.py`, `custom/`, `security_policy.json`, `roadmap.json`, `sandbox_io.py`, `general/github_issue_reporter.py`, `general/draft_cursor_prompt.py` | Orphaned, or indexed for search but not dispatchable. Replaced by user skills (`skill_loader.py`) |
| Diagnostics and live scripts | `scripts/diagnostics/*`, `scripts/test_live_actuators.py`, `verify_complex_tasks.py`, `run_logging_refactor_benchmark.py`, `generate_dana_icon.py` | Many import deleted modules (`dana.core.agent_loop`, `dana.graph.*`, `dana.ui.*`, `dana.tools.broker`) and can't run. Still working: `run_e2e_cad.py`, `run_kobayashi_maru.py`, `test_freecad_live.py`, `test_cad_vision_live.py`, `generate_ortho_tests.py` |
| Docs | `ARCHITECTURE.md`, `CONTRIBUTING.md`, `docs/architecture.md`, `docs/WHITE_PAPER.md`, `docs/SYSTEM_ARCHITECTURE_AUDIT.md`, `docs/architecture/*`, `docs/telemetry_and_ui.md`, `docs/README.md`, `docs/setup.md`, `docs/safety_and_hitl.md` | Describe the removed voice-OS / LangGraph / CustomTkinter stack (`run.py`, `dana/graph/`, Florence-2, `dana_jason_loop`). Only `README.md`, `DEMO_SCRIPT.md`, `SECURITY.md` and `docs/plugins.md` are current |
| Website | `website/` | 1 commit since 09-01. `SpindleViewer` needs an untracked `.glb` file; `FloatingLogo.astro` is unused |
| Untested frontend and native code | `tests/e2e/*.spec.ts`, `frontend/src-tauri/` | Never exercised by CI |

**Packages with no test references:** `dana/security/`, `dana/prompts/`, `dana/core/{pricing,tool_retrieval,tool_retriever}.py`, `dana/tools/{image_to_3d,geometry_analyzer,rate_limiter,sandbox_io,ipc}.py`, `dana/plugins/freecad/{skill_compiler,mesh_ops,fasteners_bootstrap}.py`, `dana/plugins/plugin_manager.py`. This is a grep result, not a coverage measurement.

---

## 4. Data flow: one chat turn

```text
React (useChatSocket) ──{text, attachments}──▶ WS /ws/chat  (server.ws_chat)
   ▲                                              │
   │ ready/plan/memory/topology/usage events      ▼
   │                                   _process_user_text
   │                                   ├─ CAD-looking? → unlock freecad_essential
   │                                   ├─ multi-step?  → task_board.create_plan (auto-seeded)
   │                                   └─ messages = [system(build_system_prompt), user]   ← fresh every turn
   │                                              │
   │                                              ▼
   │                                   _run_react_loop  (≤ 30 hops)
   │                                   ├─ context ≥85% of Ollama num_ctx → latch cloud provider
   │                                   └─ next_react_turn → _call_llm_once
   │                                         ├─ prune/compress context (context_manager)
   │                                         ├─ tools = FSM restriction | domains → top-K narrowing → minify → 2k-token cap
   │                                         └─ ModelProvider.complete_with_tool_calls
   │                                               ├─ llm_router.resolve_chain (routing_config.yaml fleet)
   │                                               └─ else legacy provider; cloud failure → Ollama fallback
   │                                              │
   │                         final answer ◀───────┴───────▶ tool call
   │                              │                           ├─ mutating & not auto-approved → hitl_approval_required (suspend)
   │                              │                           └─ _execute_and_continue → dispatch_tool_call
   │                              │                                 ├─ gates: schema · plan/FSM · measurement · collision · streaks
   │                              │                                 ├─ TOOL_HANDLERS[id] → plugin / FreeCADCmd subprocess / platform
   │                              │                                 ├─ error_digest · topology DAG · CadCallLog.record
   │                              │                                 └─ CAD mesh → /api/mesh/{token}.glb (+ optional headless visual QA)
   │                              │                           tool_dispatch_end + tool result → next hop
   │                              ▼
   └──────────── assistant_message ◀─ _finish_turn → sessions.save_session → context_distiller (background)
                 assistant_audio   ◀─ _speak_reply (TTS)
```

**Where memory enters.**
- `build_system_prompt` appends, in order:
  1. working memory ("Recent Session Context", from the distiller)
  2. core memory (`core_memory.format_core_memory_for_prompt`)
  3. the active plan
  4. the FSM anchor
- The agent writes core memory through the always-on `update_core_memory` tool, and the UI's Memory Viewer receives `memory_update` events.
- Each turn is rebuilt from scratch, so these two memories are the **only** carry-over between turns.
- The saved `working_memory` lags one turn, because `save_session` runs before distillation finishes.

**Where plugins enter.**
- Capability domains decide which tool schemas the model sees:
  - the core set is always on
  - `update_context` from the UI plugin toggles adds more
  - `load_capability` unlocks a domain, which decays after 4 turns
  - prompt keywords suggest domains
- During plan execution, the FSM narrows the list further to the current task's tools.
- `dispatch_tool_call` runs the handler with the session's CAD engine and control plane from `dana/platform/factory.py`: real FreeCAD on Windows, trimesh mock on CI and the Space.
