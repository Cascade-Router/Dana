# Donna Architecture

Donna is a **local-first agentic voice OS**: a multi-threaded perception plane, a mode-gated cognitive router, and a filesystem execution jail under a single-instance process lock. This document describes the production control paths relevant to operators and contributors.

**Stage 3 (FSM Bureaucracy & Blackboard):** Donna is no longer a purely probabilistic LangGraph system. She is a **deterministic Finite State Machine (FSM) hybrid** — RapidFuzz mailroom routing, minimized graph state, SQLite Blackboard memory, DeepSeek `<think>` extraction, Pydantic tool guards, and structured `Handoff` capability switches. LLMs generate content; Python owns routing, validation, and state transitions.

---

## 1. Multi-Threaded Ingestion Pipeline

Donna never blocks the cognitive loop on a single I/O source. Two complementary ingest planes feed the conversation finite-state machine.

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

The **Cascade Router** (`donna/cascade_router.py`) classifies complexity and selects local backends. Modes (`donna/agentic.py`) are a hard process-wide switch that changes which graph edges are legal.

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
3. **Think extractor** (`extract_r1_think_blocks` in `donna/agentic.py`): R1 `<think>...</think>` (including unclosed / multi-block) is filed to the Blackboard + `[REASONING_TRACE]`; **only sanitized text** reaches the Stage-2 Llama `bind_tools` formatter.
4. Bound-tools ReAct iterations stay on the fast local chat model for reliable `bind_tools` on Ollama.
5. **Pydantic guards** (`donna/tools/guards.py`) validate tool payloads before execution; `ValidationError` triggers **exactly one** local bounce/retry — no supervisor LLM.
6. Capability switches use the structured **`Handoff`** schema (`donna/schema.py` / `donna/handoff.py`), not raw prose intents.

---

## 3. Memory — Blackboard & Minimized LangGraph State (Stage 3 Module 1)

### 3.1 Bureaucratic graph state

`ReactGraphState` (`donna/schema.py`) is **strictly minimized** for durable control:

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
| Package | `donna/memory/blackboard.py` |
| Tables | `sessions`, `messages`, `reasoning_traces` |
| APIs | `ensure_session`, `append_message`, `load_messages`, `append_reasoning_trace`, `load_reasoning_traces` |

Chat/ReAct history and DeepSeek chain-of-thought are **permanently offloaded** here. Do not rehydrate full MemorySaver dialogue into graph state as the source of truth.

---

## 4. Single-Instance Socket Lock

**Bind address:** `127.0.0.1:47473` (exclusive TCP listen; no `SO_REUSEADDR`)

Implemented at process entry in `run.py` before `core_agent.main()`:

- First instance holds the socket for the process lifetime.
- Second instance prints:

  `[Main] ERROR: Another instance of Donna is already running. Aborting to protect execution jail.`

  and exits with code `1`.

### Why this is critical

Donna’s durable control plane lives on disk:

| Artifact | Risk under concurrent writers |
|----------|-------------------------------|
| `execution_jail/task_queue.json` | Double-drain, lost completions, corrupt JSON |
| `execution_jail/input.txt` | Raced clear/ingest; duplicate or dropped tasks |
| `donna_security/patch_ledger.md` | Interleaved ticket writes; `Errno 22` / failed drains |
| `memory/blackboard.db` | Concurrent SQLite writers / torn sessions |
| `.trigger_ask` | Two Mains consuming one inject; duplicated sessions |

Headless E2E and Startup-registered `pythonw` launches make multi-instance races likely without a lock. The socket gate is **fail-closed infrastructure**, not a UX nicety.

> Note: an additional singleton bind may exist inside the agent for legacy/dashboard purposes. The **advertised release lock** for jail protection is **`127.0.0.1:47473`** in `run.py`.

---

## 5. Thread Topology (summary)

| Thread / owner | Responsibility |
|----------------|----------------|
| Tk main (`DonnaGUI.mainloop`) | Live Trace + settings; only thread that mutates widgets |
| AgentLoop | Wake/VAD/Whisper/brain/TTS orchestration |
| MicIngest | Mic producer |
| InputIngest watcher | `input.txt` → queue |
| Tracker | JIT YOLO when Vision (or warmed) |
| System tray (`pystray`) | Open Settings / Quit |

Telemetry contracts: Live Trace UI + structured JSONL — [`telemetry_and_ui.md`](telemetry_and_ui.md).
