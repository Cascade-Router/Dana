# Dānā Architecture

Dānā (package `dana`) is a **production-hardened, local-first agentic voice OS**: a multi-threaded perception plane, a mode-gated cognitive router, transactional shadow workspaces, hybrid UIA/Florence grounding, and a filesystem execution jail under a single-instance process lock. This document describes the production control paths relevant to operators and contributors.

**White paper:** [`WHITE_PAPER.md`](WHITE_PAPER.md) — 5-phase hardening specs + OSWorld benchmarks  
**Legal & IP:** [`LEGAL_AND_IP.md`](LEGAL_AND_IP.md) — branding, Class 009/042 scope, model license posture  
**License audit:** [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) — third-party package flags  
**OSWorld bench:** `pytest tests/evals/test_osworld_bench.py`  
**Root overview:** [`../ARCHITECTURE.md`](../ARCHITECTURE.md)

**Packages:** `dana/` (core) · `dana_jason_loop/` (Jason supervisor) · `dana_security/` (AST/subprocess gates + `patch_ledger.md`)

**Stage 3 (FSM Bureaucracy & Blackboard):** Dānā is a **deterministic Finite State Machine (FSM) hybrid** — RapidFuzz mailroom routing, Memory Hydration → Supervisor Router (`hydrate_memory` → `planner`), minimized graph state, SQLite Blackboard memory, DeepSeek `<think>` extraction, Pydantic tool guards, and structured `Handoff` capability switches. LLMs generate content; Python owns routing, validation, and state transitions.

### Five hardened capabilities

| Capability | Module |
|------------|--------|
| Transactional Shadow Workspaces (`.dana_scratch/`) | `dana/exec/shadow_workspace.py` |
| Fatal Error Classifier → HITL tickets | `dana/graph/nodes/critic.py` (`FATAL_EXCEPTIONS`) |
| Hybrid Win32 UIA + Crop-and-Zoom Florence-2 | `dana/vision/hybrid_grounding.py` |
| Zero-Copy `raw_state_buffer` + Sub-Graph Retries (N=2) | `dana/graph/buffer.py`, `dana/graph/subgraph_router.py` |
| Spatial TTL (900s) + exponential decay (λ=0.05/hr) | `dana/graph/nodes/memory.py`, `dana/memory/compaction.py` |

---

## 1. Multi-Threaded Ingestion Pipeline

Dānā never blocks the cognitive loop on a single I/O source. Two complementary ingest planes feed the conversation finite-state machine.

### 1.1 MicIngest (continuous audio producer)

| Concern | Behavior |
|---------|----------|
| Thread | `MicIngest` daemon — single shared `sounddevice` `InputStream` |
| Format | 16 kHz mono float frames (wake + VAD consumers) |
| Role | Owns mic open/close; pushes frames onto an internal audio buffer queue |
| Recovery | Soft reopen on PortAudio faults; restart signal when GUI changes `mic_id` |

Downstream consumers:

1. **WakeWord** — OpenWakeWord on the live ring buffer; on hit, sets `is_recording` and yields to VAD.
2. **VAD / Conversation** — WebRTC VAD consumes frames for utterance capture → Whisper STT (JIT / background-loaded) → brain turn.

MicIngest is deliberately **producer-only**: it does not call the LLM, mutate modes, or write the execution jail.

### 1.2 InputIngest (headless / automation plane)

| Concern | Behavior |
|---------|----------|
| Watcher | Polls `execution_jail/input.txt` (silent when empty; ~0.75s back-off) |
| Conversion | Paragraph-splits free-form text → pending objects in `execution_jail/task_queue.json` |
| Session wake | Empty or non-empty `.trigger_ask` on the Main loop starts a conversation session |

**Mode gate:** `allows_react_task_jail()` is **false in Chat mode**. Pending jail tasks drain only when the agent is in a non-chat mode (typically **Developer**). This prevents casual chat turns from executing tool-jail payloads.

Equivalent session inject (including Chat / mode switches): write text to `.trigger_ask` — Main sets an injected transcript and arms the conversation loop without Whisper.

```text
input.txt ──► InputIngest ──► task_queue.json ──► drain on session (non-chat)
.trigger_ask ──► Main ──► injected question / listen ──► Conversation FSM
```

---

## 2. Cascade Router, Mailroom & Mode Map

The **Cascade Router** (`dana/cascade_router.py`) classifies complexity and selects local backends. Modes (`dana/agentic.py`) are a hard process-wide switch that changes which graph edges are legal.

### 2.1 Mailroom (RapidFuzz — Stage 3 Module 2)

**LLM routing is short-circuited** at the absolute top of `decide_route()`:

1. Normalize Whisper ASR (strip wake wrappers / punctuation).
2. Exact substring match against `COMMAND_DICTIONARY` / `STATE_TOGGLE_TRIGGERS`, else **RapidFuzz** similarity (ratio / partial / WRatio).
3. If match **≥ 80%** (`FUZZY_MATCH_THRESHOLD`), set mode / action deterministically and return a `CascadeDecision` — **no MoA / Llama classification**.
4. Known ASR garbles (e.g. `"vision mounts"` → vision) are immunized by fuzzy + dictionary aliases.
5. Fall-through (< 80%) continues to semantic / MoA heuristics and emits `[VOICE_ASR]` forensic telemetry.

Hits emit `[ROUTER]` with `raw_asr`, `matched_command`, `confidence`, and `target_node`.

| Mode | Color (Live Trace) | Cognitive path | Tool jail |
|------|--------------------|----------------|-----------|
| **Chat** | `#10B981` | Lightweight local `llama3.2`; rolling chat memory only | **Blocked** |
| **Developer** | `#8B5CF6` | ReAct / LangGraph tool loop; high-complexity → MoA (`deepseek-r1`) foresight | **Allowed** |
| **Vision** | `#3B82F6` | Scaffolded; enables JIT YOLOv8 tracker path | Allowed (scaffold) |
| **Research** | `#F59E0B` | Scaffolded research heuristics on cascade | Allowed (scaffold) |

### Fast-paths (no LLM)

Mode switches (mailroom ≥80% or exact phrase) and `clear chat memory` short-circuit in the conversation handler: set state, speak canned ack, emit telemetry — **zero** Ollama round-trip for the switch itself.

### Developer / MoA path (Stage 3 Modules 3–4)

1. Intent Broker may force-route tools such as `draft_cursor_prompt`.
2. Cascade foresight tags high-complexity → `route=moa` with DeepSeek-R1 as reasoner (**Stage-1**, no tools bound).
3. **Think extractor** (`extract_r1_think_blocks` in `dana/agentic.py`): R1 `<think>...</think>` (including unclosed / multi-block) is filed to the Blackboard + `[REASONING_TRACE]`; **only sanitized text** reaches the Stage-2 Llama `bind_tools` formatter.
4. Bound-tools ReAct iterations stay on the fast local chat model for reliable `bind_tools` on Ollama.
5. **Pydantic guards** (`dana/tools/guards.py`) validate tool payloads before execution; `ValidationError` triggers **exactly one** local bounce/retry — no supervisor LLM.
6. Capability switches use the structured **`Handoff`** schema (`dana/schema.py` / `dana/handoff.py`), not raw prose intents.

---

## 3. Memory — Blackboard & Minimized LangGraph State (Stage 3 Module 1)

### 3.1 Bureaucratic graph state

`ReactGraphState` (`dana/schema.py`) is **strictly minimized** for durable control:

| Field | Role |
|-------|------|
| `session_id` | Blackboard key — pull history / CoT by ID |
| `current_agent` | Active bureaucratic agent (`Chat_Node`, `MoA_Reasoner`, `Vision_Agent`, …) |
| `active_intent` | Current intent pointer |

Ephemeral turn scratch (`messages`, `iterations`, `last_obs`, …) exists **only** to drive `bind_tools` within a single invoke. It is **not** the durable memory store. Agents must load conversational history via `session_id` from the Blackboard.

### 3.2 Blackboard (SQLite)

| Concern | Path / API |
|---------|------------|
| Database | `memory/blackboard.db` (runtime artifact under workspace) |
| Package | `dana/memory/blackboard.py` |
| Tables | `sessions`, `messages`, `reasoning_traces` |
| APIs | `ensure_session`, `append_message`, `load_messages`, `append_reasoning_trace`, `load_reasoning_traces` |

Chat/ReAct history and DeepSeek chain-of-thought are **permanently offloaded** here. Do not rehydrate full MemorySaver dialogue into graph state as the source of truth.

---

## 4. Single-Instance Socket Lock

**Bind address:** `127.0.0.1:47473` (exclusive TCP listen; no `SO_REUSEADDR`)

Implemented at process entry in `run.py` before `core_agent.main()`:

- First instance holds the socket for the process lifetime.
- Second instance prints:

  `[Main] ERROR: Another instance of Dana is already running. Aborting to protect execution jail.`

  and exits with code `1`.

### Why this is critical

Dānā’s durable control plane lives on disk:

| Artifact | Risk under concurrent writers |
|----------|-------------------------------|
| `execution_jail/task_queue.json` | Double-drain, lost completions, corrupt JSON |
| `execution_jail/input.txt` | Raced clear/ingest; duplicate or dropped tasks |
| `dana_security/patch_ledger.md` | Interleaved ticket writes; `Errno 22` / failed drains |
| `memory/blackboard.db` | Concurrent SQLite writers / torn sessions |
| `.dana_scratch/` | Torn shadow commits if two writers share a session |
| `.trigger_ask` | Two Mains consuming one inject; duplicated sessions |

Headless E2E and Startup-registered `pythonw` launches make multi-instance races likely without a lock. The socket gate is **fail-closed infrastructure**, not a UX nicety.

> Note: an additional singleton bind may exist inside the agent for legacy/dashboard purposes. The **advertised release lock** for jail protection is **`127.0.0.1:47473`** in `run.py`.

---

## 5. Thread Topology (summary)

| Thread / owner | Responsibility |
|----------------|----------------|
| Tk main (`DanaGUI.mainloop`) | Live Trace + settings; only thread that mutates widgets (legacy class name) |
| AgentLoop | Wake/VAD/Whisper/brain/TTS orchestration |
| MicIngest | Mic producer |
| InputIngest watcher | `input.txt` → queue |
| Tracker | JIT YOLO when Vision (or warmed) |
| System tray (`pystray`) | Open Settings / Quit |

Telemetry contracts: Live Trace UI + structured JSONL — [`telemetry_and_ui.md`](telemetry_and_ui.md).

---

## 6. Zero-Touch Dynamic Plugin Architecture

Dānā loads optional capabilities (FreeCAD CAD control, future DLC plugins) from
`dana/plugins/` without any edits to `broker.py` or `agent_loop.py` — see
[`plugins.md`](plugins.md) for the developer guide. This section covers the
loader mechanics; `plugins.md` covers how to author a new plugin.

**Discovery & load** (`dana/plugins/plugin_manager.py`):

1. `discover_plugin_dirs()` globs `dana/plugins/*/manifest.json` and returns
   the sorted parent directories — a folder is a plugin iff it has a manifest.
2. `load_plugin(plugin_dir)` reads `manifest.json`, resolves `entry_point`
   (defaults to `engine.py`), and dynamically imports it via
   `importlib.util.spec_from_file_location`, registering the module into
   `sys.modules` as `dana.plugins.<name>.<stem>`.
3. For each `manifest["tools"]` entry, `getattr(module, tool_def["function"])`
   is resolved and paired with a `ToolSpec` built by `_manifest_to_tool_spec`.
4. `load_all_plugins()` iterates every discovered directory, catches
   `PluginLoadError` (or any exception) **per plugin** so one broken plugin
   never blocks the others, and caches the result (`_cached_plugins`) until
   `force_refresh=True` is requested.

**Manifest shape** (`manifest.json`): `name`, `version`, `entry_point`, and a
`tools[]` array. Each tool entry has `id`, `function` (the callable name in
the entry-point module), `description_en`/`description_fa`, a `parameters[]`
list (`name`/`type`/`required`/description), and `aliases_en`/`aliases_fa`
(with an `_intent` key) for fuzzy voice-command matching. The bundled
reference implementation is `dana/plugins/freecad/` (`manifest.json` +
`engine.py`), which registers 5 tools — box/cylinder/extrusion creation,
document modification, and raw script execution.

**Feature-gated unbinding** ties the plugin layer to the Feature Manager
(§8): `dana/features/feature_manager.py` declares a `Feature` per plugin
(e.g. `freecad` owns the 5 FreeCAD tool ids) and `_detect_default_state()`
auto-detects availability (for `freecad`, via `detect_freecadcmd()`). When a
feature is disabled — at startup or live via the toggle UI —
`apply_feature_gating(broker)` removes the owned tool ids from three surfaces
at once: `broker.registry`, the process-wide `ToolRegistry` singleton
(`get_tool_registry().unregister(tid)`), and the broker's
`_initialized_tools` dispatch cache. `IntentBroker.__init__` and
`IntentBroker.reload_registry()` (`dana/tools/broker.py`) both call
`apply_feature_gating(self)` immediately after rebuilding the registry, so a
flag flip is reflected in the **live broker instance** on the next turn —
no restart required. `set_feature_enabled()` (`feature_manager.py`) persists
the toggle to `feature_flags.json` and calls `reload_registry()`, so
re-enabling a feature restores its tools the same way.

---

## 7. Zero-Focus Multi-Monitor Workspace

Dānā never steals the foreground while it works — CAD documents and other
visual artifacts are rendered and inspected on a **secondary monitor** without
activating a window or moving the user's cursor/focus away from whatever they
are doing on the primary display.

**Primitives** (`dana/tools/os_control.py`):

- `move_window_no_activate(hwnd, x, y, w, h)` calls
  `SetWindowPos(..., SWP_NOACTIVATE | SWP_NOZORDER)` followed by
  `ShowWindow(hwnd, SW_SHOWNOACTIVATE)` — the window is repositioned and shown
  without becoming the active window.
- `get_secondary_monitor()` uses `mss.mss().monitors[1:]` (skipping index 0,
  the virtual bounding box across all displays) and returns the first monitor
  not flagged `is_primary`.
- `get_active_windows()` enumerates windows/titles via `EnumWindows` for
  read-only inspection. `set_foreground_window()` — the activating,
  focus-stealing counterpart — exists for cases that explicitly need it, but
  is **not** called by the background workspace path below.

**Orchestration** (`dana/plugins/freecad/engine.py`, `show_in_freecad_gui`):

1. `_is_freecad_gui_running()` checks for a live FreeCAD process via `psutil`.
2. `_find_freecad_window()` locates the window by a **title-contains-"freecad"
   heuristic** over `get_active_windows()`.
3. `_title_matches_file()` confirms the found window is showing the *right*
   document (`path.stem.lower() in title.lower()`) before touching it.
4. On a match, `_send_to_secondary_monitor()` calls `get_secondary_monitor()`
   + `move_window_no_activate()`, sizing the window to
   `min(1280, width) × min(800, height)` on that display.
5. If the title doesn't match, or the move fails, `_notify_cad_update_ready()`
   falls back to a **silent toast** instead of forcing a window switch.

**Silent toast fallback** (`dana/middleware/toast_notify.py`):
`show_silent_toast()` prefers `win11toast` with `audio={'silent': 'true'}`,
falling back to `_powershell_silent_toast()` (a WinRT toast raised via
PowerShell) when the package isn't available; the whole path is gated off by
`DANA_DISABLE_TOAST`. `show_silent_toast_async()` fires it on a daemon thread
so the caller never blocks on notification delivery.

This is also why Dānā's spoken status lines were revised — e.g. the Jason
Andon-override voice line (`dana/audio/multi_voice_tts.py`) now says
*"Resyncing workspace in background"* rather than language implying it is
seizing the screen, matching what the code actually does.

---

## 8. Feature Manager & Env-Key Toggles

`dana/features/feature_manager.py` is the single source of truth for which
tools are reachable at runtime, independent of whether they're built into
`dana/tools/` or supplied by a plugin (§6).

- **`FEATURES`** declares one `Feature(id, label, tool_ids, implemented)` per
  capability — e.g. `freecad` owns its 5 plugin tool ids, `vision_vlm` owns 2,
  `os_actuator` owns 2.
- **`_detect_default_state()`** auto-detects a sane default per feature
  (`freecad` → `detect_freecadcmd()`; `vision_vlm` → presence of the relevant
  API key env vars; everything else defaults enabled).
- **Persistence**: toggles and pins live in `feature_flags.json` at the
  project root as `{"enabled": {...}, "pinned_tools": [...]}`, merged with
  the detected defaults on load (`load_feature_flags()`).
- **`set_feature_enabled()`** writes the flag, and for `os_actuator`
  specifically also flips the `DANA_OS_DRY_RUN` env var — turning the
  actuator feature off puts OS automation into dry-run mode (§ safety docs),
  not just out of the tool list. It then triggers a live `reload_registry()`
  so the change is gated in immediately (§6).
- **Consumption**: `disabled_tool_ids()` feeds `apply_feature_gating()`;
  `active_feature_manifest_text()` renders a human-readable capability block
  injected into the LLM system prompt, so the model always knows what it
  currently does and doesn't have access to; `describe_feature_access(query)`
  answers direct "do you have access to X" questions deterministically
  without a model round-trip. Tool **pinning** (`pin_tool`/`unpin_tool`) is a
  separate, always-on list in the same JSON file — unrelated to enable/disable.

---

Deeper, narrower design notes live under [`architecture/`](architecture/)
(`dana_architecture.md` — LangGraph topology; `LLM_SYSTEM_MAP.md` — system
data-flow map; `system_overview_unified_canvas.md` — component/telemetry
canvas). See [`plugins.md`](plugins.md) for writing a new plugin and
[`safety_and_hitl.md`](safety_and_hitl.md) for the dry-run/HITL/kill-switch
safety stack referenced above.
