# Dānā LLM System Map

Dense topology for LLM / agent reasoning. Paths and symbols match current code (2026-08). No invented APIs.

---

## 1. System Topology

```
run.py
  └─ verify_environment() → dana.core_agent.main()
       ├─ [GUI] DanaGUI (Tk main thread) + AssistiveTouchOrb
       │     ├─ drain_state_changes ← dana.ui.status_bus
       │     ├─ PerceptionFeed (mss ~8 FPS on Perception tab)
       │     └─ after(…) → agent_loop thread + pystray tray
       └─ [--no-gui] agent_loop() owns process

agent_loop threads (daemon):
  Tracker          → tracker_worker(device)          # YOLO / vision tools
  WakeWord         → wakeword_worker()               # OpenWakeWord + onset
  Conversation     → conversation_worker(...)        # VAD→STT→brain→TTS
  InputIngest      → input_txt_ingest_worker()       # text / file inject
  TTSWorker        → tts_worker()                    # Piper spool consumer
  MicIngest        → mic_ingest_worker()             # single InputStream producer
  Sidekick         → start_sidekick_supervisor()     # vision_poller + actuator_executor
  KillSwitch       → start_kill_switch_listener()    # F12 default
  Telemetry dash   → start_dashboard_thread()

Optional sidecar (not desktop boot path):
  dana.daemon.engine.EngineDaemon @ 127.0.0.1:50051
    methods: ping | system_status | stream_chat | hot_restart
    stream_chat → requires_tool_graph? → run_react_loop | run_lightweight_chat
```

| Layer | Role | Modules / symbols |
|-------|------|-------------------|
| Entry | Desktop / frozen | `run.py` → `core_agent.main` / `agent_loop`; package via `build_dana.py` → `dist/Dana/Dana.exe` |
| UI smoke | Headless paint | `python -m dana.ui.main` constructs `DanaGUI`, auto-destroys unless `--stay` |
| Cognition | Chat vs tools | `dana.agentic.run_lightweight_chat`, `run_react_loop` → `_run_react_loop_langchain` |
| Graph | LangGraph corridor | `dana.agentic_react_graph.compile_dana_react_graph` |
| Intent IR | STT fast-path | `dana.tools.broker.IntentBroker` / `get_broker()` / `parse_utterance` |
| Tools exec | Bound actuators | `core_agent.execute_tool_call` ← LangChain tools + plugins |
| Secure KV | Encrypted profile | `dana.secure_memory.SecureMemory`, `dana.vault_service.VaultClient` `:47475` |
| Code vault | Chroma embeddings | `dana.memory.vault.CodebaseVault` (`.dana/vault/`) |
| Episodic | SQLite facts | `dana.memory.store.EpisodicMemoryStore` / `get_episodic_store` |
| Blackboard | Session / mode | `dana.memory.blackboard` (hydrate corridor) |
| Paths | Cwd-independent | `dana.paths` (`PROJECT_ROOT`, workspace) — do not alter in patches |

**Production graph topology** (`compile_dana_react_graph`):

```
START → hydrate_memory → planner → executor
  ├─ os_worker → verifier → consolidate_memory → END
  └─ agent
       ├─ ticket_validate → jason_ticket_review → ticket_approval → tools
       ├─ tools → critic | fail_closed | agent | verifier
       └─ consolidate_memory → END
```

Cap: `REACT_MAX_ITERS = 3` (`dana.agentic`).

---

## 2. Data Flow Pipelines

### Audio Ingest → Wake → VAD → STT → Inference → TTS / UI

```
OS mic (native rate, often 44100)
  → mic_ingest_worker()
       resolve_live_input_device() / stream_device_kwargs()   # dana.audio.devices
       InputStream → resample_to_16k() → audio_buffer_queue
  → wakeword_worker()
       OpenWakeWord score + onset (WAKE_THRESHOLD / WAKE_ONSET_*)
       optional wake_phrase_confirmed() (WAKE_PHRASE_VERIFY=False by default)
       gates on ollama_ready, engine_engaged, quiet_mic_mode
       is_recording.set() → conversation session
  → record_utterance()
       get_mic_frame() ← queue @ 16 kHz
       DcBlocker → is_speech_frame() (dana.audio.vad_consumer / Silero)
       silence cut VAD_SILENCE_MS=1500; max VAD_MAX_SECONDS / FOLLOWUP_VAD_MAX_SECONDS
       emit_state_change("listening") on dynamic speech floor
  → prepare_audio_for_whisper() / transcribe_audio()
       Distil-Whisper (WHISPER_ID=distil-whisper/distil-small.en)
       hallucination filters: is_whisper_hallucination, is_silent_non_speech_transcript
       correct_known_stt_names()
  → conversation_worker.run_brain_turn()
       fast-paths: clear chat memory, dictation, mode switch, standby/lockdown/time
       tool_router() → IntentBroker.parse_utterance
         switch_vision_source → execute_tool_call (immediate)
         else forced_tool into ReAct
       mode chat → run_lightweight_chat (completion_gate may escalate once)
       else → run_react_loop(..., execute_fn=execute_tool_call, tts_callback=enqueue_speech)
  → enqueue_speech() → tts_queue (text, interruptible, agent_id)
  → tts_worker()
       Piper onnx (lazy load); chunk_text_for_tts; _resample_pcm if host≠Piper rate
       playback_lock; barge via dana.audio.tts_worker / Silero stream barge
  → UI: emit_live_transcript, set_ui_state, status_bus STATE_CHANGE, tray icon
```

### Text / daemon inference (no mic)

| Path | Router | Executor |
|------|--------|----------|
| GUI chat / inject | `set_injected_question` / `input_txt_ingest_worker` | same `run_brain_turn` |
| EngineDaemon `stream_chat` | `requires_tool_graph(message)` | `run_react_loop` or `run_lightweight_chat`; tools via `_daemon_execute_tool` → `execute_tool_call` |

### Intent Broker (bilingual IR)

`IntentBroker.parse_utterance` (`dana/tools/broker.py`):

1. `normalize_text` / lang detect  
2. Longest EN/FA alias match from tool registry  
3. Force-routes (research, vault/chroma, forge, PowerShell, browser, tickets, …)  
4. Validate / coerce → `ToolCall` IR  
5. Deferred bind: most tools enter ReAct; `switch_vision_source` fast-pathed in `tool_router`

---

## 3. State Management

### `dana.ui.status_bus`

- Singleton `StatusEventBus` (`queue.Queue` max 128; silent drop on full).
- Statuses: `idle` | `listening` | `routing` | `executing`.
- Writers (workers, tools, daemon, graph): `emit_state_change(status, tool=, message=)`.
- Readers: `DanaGUI._poll_state_changes` (~5 Hz) → `drain_state_changes` → `_apply_state_change` (System Status line + VAD mic pip).
- Headless-safe: no Tk imports in bus module.

### Conversation / UI phase (`core_agent`)

| Symbol | Purpose |
|--------|---------|
| `ui_state` + `ui_state_lock` | `idle` / `listening` / `followup` / `transcribing` / `thinking` / `speaking` via `set_ui_state` |
| `is_recording` | Wake / trigger starts turn |
| `engine_engaged` | Soft STANDBY vs ACTIVE (Stage 8.9.7) |
| `ollama_ready` / `wakeword_armed` / `piper_voices_ready` | Boot gate for ready chime |
| `quiet_mic_mode` | Text-only / adaptive floors when resolve → `quiet_mic` |
| `conversation_history` + lock | Last 6 user/assistant msgs for ReAct priors |
| `VAULT_HOT_CACHE` | Prefetched SecureMemory identity keys |
| `latest_frame` / `latest_dets` + locks | Tracker → vision prompt |
| `active_vision_tool` | `ScreenAgent` vs `VideoAgent` |
| `tts_queue` / `tts_busy` / `speech_idle` / `tts_interrupt_event` | Spool + half-duplex |
| `vad_capture_active` / `vad_abort_event` | Mic ownership / text override abort |
| `stop_event` | Process shutdown |

### Daemon IPC

- Framing: newline JSON (`dana.daemon.protocol`).
- Session: in-memory `conversation` / `task_state` + `ProcessWatchdog` persist for `hot_restart`.
- Heartbeats every 2s while graph runs (client sock timeout ~10s).
- Emits `STATE_CHANGE` events on wire + local `emit_state_change`.

### UI threads

| Component | Threading model |
|-----------|-----------------|
| `DanaGUI` | Tk mainloop owns UI; agent_loop in background thread |
| `AssistiveTouchOrb` | Frameless topmost Toplevel; dictation / HITL / dashboard callbacks; agent label via `get_active_tts_agent` |
| Theme | `dana.ui.theme` — Obsidian Mint / Cyber Amber / Ghost Light; `apply_dana_ctk_theme` |
| Perception feed | Worker thread `_capture_perception_frame` → `after(0, _apply)`; schedule `after(125)` |
| Trace | `emit_trace` / `TraceEventBus` → Live Trace panel |

---

## 4. Perception / Actuation

### Perception

| Capability | Entry | Notes |
|------------|-------|-------|
| YOLO tracker | `tracker_worker` | Updates `latest_dets`, `SPATIAL_AGGREGATOR` |
| Screen / camera tools | `ScreenAgent` / `VideoAgent` | `switch_vision_source` via broker |
| Hybrid grounding | `dana.graph.nodes.vision.get_hybrid_grounding` | UIA → Florence crop/zoom |
| SpatialIR | `spatial_context.SPATIAL_AGGREGATOR` | `synthesize_prompt_block`, transcript updates |
| OCR actuator | `dana.tools.vision.analyze_visual_context` | mss → Pillow → pytesseract; `emit_state_change(executing)` |
| GUI preview | `DanaGUI._schedule_perception_feed` | mss primary monitor, thumbnail 480×270 |
| Overlay | `dana.vision.overlay.get_overlay` | ROI HUD for grounding |

### Actuation / tools

| Tool surface | Module | Behavior |
|--------------|--------|----------|
| PowerShell | `dana.tools.powershell.execute_powershell` | Job-object sandbox; `DANGEROUS_COMMANDS_RE` → `SECURITY_VIOLATION` |
| Shell / write | `dana.tools.actuators.execute_command`, `write_to_file` | Windows → PowerShell; same danger regex |
| Browser fetch | `dana.tools.browser.fetch_webpage` | Playwright headless Chromium; http(s) only |
| Vault KV | `VaultClient` + SecureMemory tools | Loopback daemon caches Fernet key in RAM |
| Chroma vault | `ingest_local_directory` / `search_vault` (`dana.memory.vault`) | Codebase chunks; not chat memory |
| Ticket / HITL | graph `ticket_validate` → `jason_ticket_review` → `ticket_approval` | interrupt/resume via checkpointer |
| Tool Forge | broker forge hints + registry | Dynamic tools; security gates in `dana_security` (do not patch casually) |

`execute_tool_call` in `core_agent` is the live dispatcher for ReAct / daemon / broker fast-path.

---

## 5. Memory

| Store | Path / binding | API |
|-------|----------------|-----|
| SecureMemory (AES profile) | `dana_memory.enc` via `default_vault_path()` | `unlock_dana_memory`, `VaultClient` session token |
| Hot cache | process `VAULT_HOT_CACHE` | `populate_vault_hot_cache` — skip ReAct for identity |
| Chroma codebase | `.dana/vault/` collection `dana_codebase_vault` | `CodebaseVault.ingest_local_directory` / `search_vault` |
| Episodic SQLite | `EpisodicMemoryStore` (`episodic_facts` + TTL) | `hydrate_memory_node` / `consolidate_memory_node` |
| Blackboard | `memory/blackboard.db` | Session mode, DeepSeek traces, voice mode |
| Rolling chat | `agentic` chat memory + `conversation_history` | Clear via `parse_clear_chat_memory` (not Chroma) |

---

## 6. Current Constraints

| Constraint | Reality in code |
|------------|-----------------|
| **44.1k vs 16k** | Capture at device native rate (`AUDIO_INPUT_RATE`); always resampled to `SAMPLE_RATE=16000` before VAD/Whisper. TTS: Piper rate vs host — `_resample_pcm` avoids 2× speed on 44100 WASAPI/Sonar. |
| **Sonar / quiet mic** | Prefer live OS default via `resolve_live_input_device`; SteelSeries names can lock selection. Low RMS → `quiet_mic_mode` (Text-Only), adaptive Whisper gain (`WHISPER_TARGET_RMS`), wake disarmed until live. Sonar often ~rms 0.003. MME preferred for Sonar InputStream stability. |
| **Audio defaults** | `agent_loop` forces `mic_id, speaker_id = None, None` (System Default) then `ensure_live_mic`. |
| **Threading / locks** | Single mic producer, many consumers. Critical: `mic_lock`, `playback_lock` (RLock), `_tts_enqueue_lock`, frame/dets/vision/history locks. Half-duplex: VAD ignores onset while `tts_busy`. |
| **PortAudio hangs** | `MIC_STREAM_OPEN_TIMEOUT_S` / `MIC_STREAM_READ_TIMEOUT_S`; `audio_hardware_fault` → `soft_recover_audio_hardware`. |
| **Ollama dependency** | Brain is local Ollama (`ask_ollama_messages`); wake waits `ollama_ready`. Unreachable → spoken `OLLAMA_UNREACHABLE_SPEECH`. |
| **Chat vs graph** | Chat = no tools; filler/verbal-tool → one escalate to `run_react_loop`. Daemon uses `requires_tool_graph` (small-talk stays lightweight). |
| **run.py vs dist exe** | Dev: `python run.py` (singleton port `47473`, torch major check). Ship: `build_dana.py` packs `run.py` → `dist/Dana/Dana.exe` (not `dana.ui.main` / daemon). Frozen assets: `sys._MEIPASS` / `sys.frozen` in `dana.resources`, `dana.ui.logo`. |
| **UI smoke vs product** | `dana.ui.main` = short-lived `DanaGUI` smoke. Product UI = `core_agent.DanaGUI` from `main()`. |
| **English-only release** | Settings language lock logged at boot; Distil-Whisper `.en`. |
| **Security / paths** | Do not modify ToolForge gates, offline routing structure, or `dana/paths.py` in casual patches. |
| **Wiretap** | No `wiretap` symbol in current tree (no removal needed). |

---

## Quick symbol index

| Need | Go to |
|------|-------|
| Boot + threads | `core_agent.agent_loop`, `main` |
| Live mic bind | `audio.devices.resolve_live_input_device`, `ensure_live_mic` |
| VAD | `record_utterance`, `audio.vad_consumer.is_speech_frame` |
| STT | `transcribe_audio`, `load_whisper` / `ensure_whisper_bundle` |
| Brain turn | `conversation_worker` → `run_brain_turn` |
| ReAct graph | `agentic_react_graph.compile_dana_react_graph`, `agentic.run_react_loop` |
| Broker | `tools.broker.get_broker`, `parse_utterance`, `tool_router` |
| TTS spool | `enqueue_speech`, `tts_worker` |
| Status UI | `ui.status_bus`, `DanaGUI._poll_state_changes` |
| Daemon | `daemon.engine.EngineDaemon`, `daemon.protocol.METHODS` |
| Chroma / episodic | `memory.vault.CodebaseVault`, `memory.store.EpisodicMemoryStore` |
