# Telemetry and Live Trace UI

Donna’s operator UI is a **decoupled CustomTkinter surface**. Perception, cognition, and tool workers must never call Tk APIs directly. All Live Trace updates cross a single thread-safe queue and are applied on the GUI main thread.

**Stage 3 also adds a structured JSONL logger** (`donna/telemetry.py` → `logs/donna_telemetry.jsonl`) for forensic / queryable bureaucracy events. Live Trace and JSONL are complementary: UI for operators, JSONL for diagnostics and soak tests.

---

## Design Goals

1. **Crash isolation** — Tkinter is not thread-safe; cross-thread widget updates cause intermittent freezes and hard crashes on Windows.
2. **Transparency** — Contributors and operators can see stage lifecycle (`active` → `completed` / `bypassed`) without reading logs.
3. **Mode awareness** — Header accent colors track Chat / Developer / Vision / Research without polling widgets from workers.
4. **Queryable bureaucracy** — JSONL tags make mailroom hits, think traces, tool guards, and handoffs greppable without scraping free-form runtime logs.

---

## Components

| Symbol | Location | Role |
|--------|----------|------|
| `gui_telemetry_queue` | `donna/core_agent.py` | Process-wide `queue.Queue` of trace event dicts |
| `emit_trace(...)` | `donna/core_agent.py` | Safe producer API for any thread |
| `DonnaGUI.process_telemetry` | `donna/core_agent.py` | Consumer: `get_nowait` + `after(100, ...)` |
| `TraceCell` | `donna/core_agent.py` | One stage row (icon + message + border pulse) |
| `emit_tagged` / helpers | `donna/telemetry.py` | Structured JSONL writer (Stage 3) |
| `TELEMETRY_JSONL_PATH` | `logs/donna_telemetry.jsonl` | Append-only forensic event stream |

```text
  Worker threads                    Tk main thread
  ─────────────                    ──────────────
  emit_trace(...) ──put_nowait──► gui_telemetry_queue
                                         │
                                         ▼
                              process_telemetry()
                                   TraceCell.pack / update_status
                                   border pulse after(500)

  Worker threads                    Disk
  ─────────────                    ────
  emit_tagged / log_* ───────────► logs/donna_telemetry.jsonl
```

---

## Structured JSONL Telemetry (Stage 3)

Path: **`logs/donna_telemetry.jsonl`** (one JSON object per line).

| Tag | Helper | When |
|-----|--------|------|
| `[VOICE_ASR]` | `log_voice_asr` | Whisper / user transcript; mailroom fall-through |
| `[ROUTER]` | `log_router` | Mailroom ≥80% hit; cascade / session route decisions |
| `[REASONING_TRACE]` | `log_reasoning_trace` | Extracted DeepSeek `<think>` CoT (+ latency metadata) |
| `[TOOL_EXECUTION]` | `log_tool_execution` | Guarded tool call success / error |
| `[HANDOFF]` | `log_handoff` | Deterministic `Handoff` capability switch |

Each record includes `ts`, `tag`, `message`, `session_id`, `current_agent`, `active_intent`, optional `latency_ms` / `payload`. Unknown tags raise at emit time — do not invent free-form labels.

Dashboard markdown (`dashboard.md`) remains a separate ~45s operator surface and is **not** a substitute for JSONL forensics.

---

## `emit_trace()` Contract

```python
emit_trace(stage: str, status: str, message: str, mode: str | None = None) -> None
```

| Field | Meaning |
|-------|---------|
| `stage` | Stable key for a pipeline node (e.g. `"STT"`, `"Router"`, `"Mode"`). Re-emits **update** the same `TraceCell`. |
| `status` | `"active"` \| `"completed"` \| `"bypassed"` |
| `message` | Human-readable line shown in the cell |
| `mode` | Optional; updates the header mode indicator (`chat`, `developer`, `vision`, `research`) |

**Icons**

| Status | Icon |
|--------|------|
| `active` | ⏳ |
| `completed` | ✅ |
| `bypassed` | ⏭️ |

**Mode palette**

| Mode | Hex |
|------|-----|
| Chat | `#10B981` |
| Developer | `#8B5CF6` |
| Vision | `#3B82F6` |
| Research | `#F59E0B` |
| Idle / bypassed accent | `#9CA3AF` |

Implementation notes for contributors:

- Prefer `put_nowait` semantics (already used inside `emit_trace`) so a stalled UI cannot block audio/LLM threads.
- Invalid `status` values coerce to `"active"`.
- Do **not** hold references to `DonnaGUI` widgets from worker code; only call `emit_trace`.
- For FSM / soak diagnostics, also emit the matching JSONL tag via `donna.telemetry` helpers.

---

## Consumer Loop (`process_telemetry`)

Scheduled with `self.after(100, self.process_telemetry)` (~10 Hz):

1. Drain `gui_telemetry_queue` via `get_nowait` until empty.
2. For each event: create a `TraceCell` if `stage` is new; otherwise `update_status`.
3. Optionally scroll the `CTkScrollableFrame` to the latest cell.
4. Reschedule itself.

**Pulse:** `after(500, _pulse_active_cells)` toggles border color on cells whose status is `"active"` between the current mode accent and a dim gray — visual proof the stage is in-flight.

---

## Contributor Checklist

When adding a new pipeline stage or tool node:

1. Call `emit_trace("YourStage", "active", "…")` when work starts.
2. Call `emit_trace("YourStage", "completed", "…")` or `"bypassed"` on terminal outcomes.
3. Pass `mode=...` only when the event itself changes or clarifies agent mode.
4. Never import or invoke CustomTkinter from MicIngest, VAD, Whisper, Ollama, or tool handlers.
5. Emit the appropriate JSONL tag (`[TOOL_EXECUTION]`, `[HANDOFF]`, etc.) for bureaucratic events.
6. Validate the new stage appears in Live Trace during the **Headless E2E State Reachability** suite (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

---

## Opening the UI

The window starts withdrawn (tray-resident). Use system tray → **Open Settings** to raise the Live Trace + Stats / Audio / Transcript tabs. Boot hooks emit sample `Boot` / `STT` / `Router` events so the scroll area is non-empty immediately after launch.
