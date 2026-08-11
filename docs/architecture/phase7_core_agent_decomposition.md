# Phase 7 scope: retiring the last ~34 functions from `dana/core_agent.py`

## Context

Phases 5 and 6 cut `dana/core_agent.py` from 7815 to 1922 lines by extracting
the GUI (`dana/ui/app_gui.py`, `dana/ui/tray_icon.py`) and the process
lifecycle bucket (`dana/core/app_runtime.py`). The directive's original goal
was a `<50-line`, zero-logic re-export shim; what's left is ~34 functions
that couldn't move in Phase 6 without creating a real circular import
between `core_agent.py` and `app_runtime.py`, because `dana/core/agent_loop.py`
(Phase 4's module) already depends on several of them via an established
lazy-import pattern.

This document is the scoping pass for finishing the job — it does not move
any code. It exists so Phase 7 can be executed (by anyone, in a future
session) without re-deriving the dependency graph from scratch.

## The hard constraint Phase 7 must respect

`dana/core/agent_loop.py` has exactly four lazy-import call sites reaching
into `dana.core_agent`:

```
agent_loop.py:134   from dana.core_agent import format_class_list
agent_loop.py:190   from dana.core_agent import flush_conversation_memory
agent_loop.py:1306  from dana.core_agent import speak_tool_working_ack
agent_loop.py:1607  from dana.core_agent import (
    _nt_hide_console_if_mp_child, clear_context_spoken_reply,
    compile_and_append_voice_prompt, emit_trace, execute_lockdown_shutdown,
    flush_conversation_memory, format_class_list, format_vision_context_for_llm,
    get_spatial_memory_labels, is_clear_context_command, is_engine_engaged,
    is_lockdown_command, is_standby_command, is_time_command,
    parse_yolo_results, pop_injected_question_ex, remember_spatial_labels,
    speak_tool_working_ack, wall_clock_spoken_reply, yolo_device_arg,
)
```

(`execute_lockdown_shutdown` already moved to `dana.core.app_runtime` in
Phase 6 — this call site keeps working today only because `core_agent.py`
re-exports it. That re-export is proof this coordination point is real, not
theoretical.)

**Any of the other 19 names in that list that move in Phase 7 must have this
import block updated in the same commit**, or `dana/core/agent_loop.py`
breaks immediately on the next `conversation_worker` call. This is the one
thing that makes Phase 7 riskier than Phases 5/6: it's not a self-contained
file move, it's a two-file coordinated change.

## Full inventory (34 functions + 1 dead one)

Verified via `grep -rln` across `dana/`, `tests/`, `scripts/`, `run.py` for
every name (excluding `core_agent.py` itself), then manually confirmed
comment-only hits vs. real imports.

| Function | Real external dependents (beyond core_agent.py) | Proposed home |
|---|---|---|
| `tracker_worker` | none (comment-only in `dana/tools/vision.py`) | `dana/vision/tracker_worker.py` |
| `spatial_zone` | none (internal to `tracker_worker`) | `dana/vision/tracker_worker.py` |
| `parse_yolo_results` | `dana/core/agent_loop.py` (lazy) | `dana/vision/tracker_worker.py` |
| `remember_spatial_labels` | `dana/core/agent_loop.py` (lazy) | `dana/vision/tracker_worker.py` |
| `get_spatial_memory_labels` | `dana/core/agent_loop.py` (lazy) | `dana/vision/tracker_worker.py` |
| `format_class_list` | `dana/core/agent_loop.py` (lazy, 2 call sites) | `dana/vision/tracker_worker.py` |
| `format_vision_context_for_llm` | `dana/core/agent_loop.py` (lazy) | `dana/vision/tracker_worker.py` |
| `yolo_device_arg` | `dana/core/agent_loop.py` (lazy) | `dana/vision/tracker_worker.py` (sibling of `select_device`/`select_dtype`, which already moved to `app_runtime.py` in Phase 6 — reunite if a shared torch-device helper module is preferred instead) |
| `_keyword_hit` | **none, anywhere, including internally** | dead code — delete, don't migrate |
| `wakeword_worker` | `scripts/debug_wakeword.py` (real import) | `dana/audio/wakeword_worker.py` |
| `wake_score_hit` | `scripts/debug_wakeword.py` (real import) | `dana/audio/wakeword_worker.py` |
| `_normalize_wake_text` | none (internal) | `dana/audio/wakeword_worker.py` |
| `_wake_text_matches_dana` | none (internal) | `dana/audio/wakeword_worker.py` |
| `wake_phrase_confirmed` | none (internal) | `dana/audio/wakeword_worker.py` |
| `input_txt_ingest_worker` | none (docstring-only mention in `dana/audio/mic_input.py`) | `dana/ingestion/text_injection.py` (new package) |
| `abort_vad_listening` | none (internal) | `dana/ingestion/text_injection.py` |
| `prioritize_text_input` | `tests/test_dead_mic_silence_gate.py` | `dana/ingestion/text_injection.py` |
| `set_injected_question` | `dana/ui/app_gui.py` (the one deliberate lazy-import exception from Phase 5 — see its comment) | `dana/ingestion/text_injection.py` |
| `clear_injected_question` | `tests/test_stage810_silent_text_chat.py` | `dana/ingestion/text_injection.py` |
| `pop_injected_question` | none (internal, thin wrapper over `pop_injected_question_ex`) | `dana/ingestion/text_injection.py` |
| `pop_injected_question_ex` | `dana/core/agent_loop.py` (lazy), `tests/test_stage810_silent_text_chat.py` | `dana/ingestion/text_injection.py` |
| `_clear_text_injection_file` | none (internal) | `dana/ingestion/text_injection.py` |
| `pop_text_injection` | `tests/test_text_injection.py` | `dana/ingestion/text_injection.py` |
| `compile_and_append_voice_prompt` | `dana/core/agent_loop.py` (lazy) | `dana/ingestion/text_injection.py` |
| `emit_trace` | `dana/core/agent_loop.py`, `dana/core/app_runtime.py`, `dana/middleware/hitl_ticket.py`, `dana/ui/app_gui.py` | **`dana/core/shared_state.py`** — its only remaining deps (`gui_telemetry_queue`, `_TRACE_STATUS_ICONS`) already live there; not vision/audio/ingestion |
| `_nt_hide_console_if_mp_child` | `dana/audio/mic_input.py`, `dana/audio/tts_worker.py`, `dana/core/agent_loop.py` (all lazy) | **`dana/paths.py`** (already owns process-bootstrap concerns) or a new `dana/core/bootstrap.py` — not vision/audio/ingestion |
| `is_engine_engaged` | `dana/core/agent_loop.py` (lazy), `tests/test_stage897_engine_toggle.py` | **`dana/core/shared_state.py`** as a first-class accessor (mirrors `set_ui_state`/`get_ui_state`) — it's already a 1-line wrapper over `shared_state.engine_engaged` |
| `set_engine_engaged` | `tests/test_default_audio_devices.py`, `tests/test_stage810_silent_text_chat.py`, `tests/test_stage897_engine_toggle.py` | **`dana/core/shared_state.py`**, same reasoning |
| `is_standby_command` | `dana/core/agent_loop.py` (lazy) | `dana/agentic.py` or new `dana/core/command_classifiers.py` |
| `is_clear_context_command` | `dana/core/agent_loop.py` (lazy), `tests/test_multi_turn.py` | same as above |
| `flush_conversation_memory` | `dana/core/agent_loop.py` (lazy, 2 call sites), `tests/test_multi_turn.py` | same as above |
| `clear_context_spoken_reply` | `dana/core/agent_loop.py` (lazy), `tests/test_multi_turn.py` | same as above |
| `is_lockdown_command` | `dana/core/agent_loop.py` (lazy) | same as above |
| `is_time_command` | `dana/core/agent_loop.py` (lazy) | same as above |
| `wall_clock_spoken_reply` | `dana/core/agent_loop.py` (lazy) | same as above |
| `speak_tool_working_ack` | `dana/core/agent_loop.py` (lazy, 2 call sites), `tests/test_text_pipeline.py` | same as above |

## Why "vision / audio / ingestion" doesn't quite cover everything

Your three target directories cleanly absorb 24 of the 34 functions:

- **`dana/vision/tracker_worker.py`** (new file in the existing `dana/vision/`
  package) — 8 functions: the tracker thread + its YOLO/spatial-memory helpers.
- **`dana/audio/wakeword_worker.py`** (new file in the existing `dana/audio/`
  package) — 5 functions: the wake-word thread + its scoring/matching helpers.
- **`dana/ingestion/`** (new package — doesn't exist yet) — 9 functions:
  text-injection state, the `.trigger_ask`/`input.txt` bridge, and the
  ingest worker thread.

The remaining **10 functions** are cross-cutting "core" concerns that don't
belong in any of the three (forcing them in would just relocate the same
architectural mismatch, not fix it):

- `emit_trace`, `is_engine_engaged`, `set_engine_engaged` → `dana/core/shared_state.py`
  (each is a thin wrapper over state that already lives there)
- `_nt_hide_console_if_mp_child` → `dana/paths.py` (process bootstrap, not audio-specific — it's used by two audio files today only because they happen to spawn multiprocessing workers)
- `is_standby_command`, `is_clear_context_command`, `flush_conversation_memory`,
  `clear_context_spoken_reply`, `is_lockdown_command`, `is_time_command`,
  `speak_tool_working_ack`, `wall_clock_spoken_reply` → conversational
  command-classification helpers. `dana/agentic.py` (which already owns
  `get_dana_mode`/`set_dana_mode`/mode-switch parsing) is the thematic fit,
  but it's already 3207 lines — checked, and that's too big to grow further
  without its own decomposition problem. Use a new
  `dana/core/command_classifiers.py` instead.

Plus one deletion: `_keyword_hit` has zero call sites anywhere (verified —
not even referenced internally), so it's dead code left over from an earlier
version of the keyword-matching logic. Phase 7 should delete it rather than
migrate it.

## Proposed sequencing

Do the two coordination-free groups first — they're pure file moves with
zero risk to `dana/core/agent_loop.py`:

1. **`dana/audio/wakeword_worker.py`** (5 functions) — also update
   `scripts/debug_wakeword.py`'s imports. Zero `dana/core/agent_loop.py`
   coordination needed.
2. **`dana/ingestion/text_injection.py`** (9 functions, new package) — also
   update `dana/ui/app_gui.py`'s one deliberate lazy-import (the
   `set_injected_question` exception documented in its Phase 5 comment —
   this becomes a clean top-level import once the function has a real home
   outside `core_agent.py`), plus 3 test files
   (`test_dead_mic_silence_gate.py`, `test_stage810_silent_text_chat.py`,
   `test_text_injection.py`). **One function in this group
   (`pop_injected_question_ex`) IS in `dana/core/agent_loop.py`'s lazy-import
   block** — coordinate that one function's move with step 4 below, or move
   it in this step and update just that one name in the block early.
3. Delete `_keyword_hit`.

Then the two that require touching `dana/core/agent_loop.py`'s big
lazy-import block — do these together, in one commit, verified by the same
targeted-test-then-full-suite approach used in Phases 5/6:

4. **`dana/vision/tracker_worker.py`** (8 functions, 6 of which are in the
   lazy-import block) + update `dana/core/agent_loop.py`'s import block.
5. **The 10 cross-cutting functions** to `dana/core/shared_state.py`,
   `dana/paths.py`, and `dana/agentic.py` (or `dana/core/command_classifiers.py`)
   + update `dana/core/agent_loop.py`'s import block (this removes the block
   entirely, since every name it imports will have moved).

Once step 5 lands, `dana/core/agent_loop.py` has no remaining dependency on
`dana.core_agent` at all — at that point `core_agent.py` can finally become
the `<50-line`, zero-logic shim originally targeted, re-exporting `main`,
`DanaGUI`, `agent_loop`, `execute_tool_call`, `state` (per the original
directive's Step 3) with no leftover business logic.

## Risks worth flagging before starting

- Steps 4-5 are the only genuinely risky part — a missed name in
  `dana/core/agent_loop.py`'s import block fails at first call, not at
  import time (it's a lazy import), so a quick `python -c "import
  dana.core_agent"` sanity check won't catch it — the targeted test files
  that actually exercise `conversation_worker` need to run (`tests/test_multi_turn.py`,
  `tests/test_text_pipeline.py`, and the stage3/full_tree_text_suite files
  are the ones that reach deep enough into that code path).
- `dana/agentic.py` is already 3207 lines, so the command-classifier
  functions go to a new `dana/core/command_classifiers.py` instead of
  growing it further (see above).
- Same lazy-import-to-avoid-a-cycle pattern from Phases 5/6 may recur:
  moving `emit_trace`/`is_engine_engaged`/`set_engine_engaged` into
  `dana/core/shared_state.py` is very likely clean (shared_state doesn't
  import from core_agent), but verify with the same
  `ruff --select F401,F821,F811` + fresh-process import check used in
  Phases 5/6 before trusting it.
