# Safety Architecture: Dry-Run, HITL, and the Panic Switch

Dānā can drive OS-level input (mouse/keyboard/window control, §7 of
[`architecture.md`](architecture.md#7-zero-focus-multi-monitor-workspace)) and
CAD tooling via plugins. Three independent, layered safety mechanisms bound
that capability: a **dry-run gate** that can neutralize actuation entirely, a
**human-in-the-loop (HITL) ticket** approval step before higher-risk plans
run, and a **hardware panic switch** that can abort in-flight actions at any
time. They are independent — any one of them tripping stops execution
regardless of the state of the other two.

## 1. Dry-Run Gate — `DANA_OS_DRY_RUN`

Single source of truth: `dana/security/dry_run.py`.

```python
def is_dry_run_enabled(env_var: str = "DANA_OS_DRY_RUN") -> bool:
    return os.environ.get(env_var, "").strip().lower() in {"1", "true", "yes", "on"}
```

This exists because, before it was centralized, several actuator modules
each re-implemented the same env-var check slightly differently (some
modules silently ignored `"on"`), so the same env var could mean different
things depending which one read it. Every actuator (`dana/tools/*_actuator.py`,
`dana/operators/*.py`, `dana/os_automation.py`) now calls this one function.

**How to use it:**

- Set `DANA_OS_DRY_RUN=1` in the environment before starting Dānā to make all
  Win32 SendInput / window actuation a no-op that still logs what *would*
  have happened — useful for testing a new plugin or tool without touching
  real windows.
- The Feature Manager (`dana/features/feature_manager.py`) also flips this
  flag programmatically: disabling the `os_actuator` feature via
  `set_feature_enabled()` sets `DANA_OS_DRY_RUN` for you, so turning that
  feature off is a real actuation kill switch, not just a tool-list filter.
  See [`architecture.md` §8](architecture.md#8-feature-manager--env-key-toggles).

## 2. HITL Ticket Ledger

Two distinct pieces work together here — an **approval bridge** that pauses
execution for a human decision, and an **archive ledger** that's a durable
record of tickets over time.

### Approval bridge (`dana/middleware/hitl_ticket.py`)

Higher-risk plans pause inside a LangGraph `ticket_approval` node via
`interrupt()`. The ReAct runner publishes the drafted ticket with
`publish_pending(payload, thread_id=...)`; the Live Trace GUI (or a test)
resolves it with `submit_decision(approved, ...)`; the graph worker blocks in
`wait_for_decision()` until that happens.

- **Master switch:** `hitl_enabled()` reads `DANA_HITL_TICKET` (default on;
  set to `0`/`false`/`off`/`no` to disable the gate entirely).
- **Headless/test behavior:** `should_auto_resolve()` auto-approves when no
  GUI is listening (`_gui_listening()` checks for a live Tk instance),
  *unless* `DANA_HITL_REQUIRE_GUI=1` forces a real human decision.
  `DANA_HITL_AUTO_APPROVE` / `DANA_HITL_AUTO_DENY` force a specific outcome
  deterministically — set one of these in CI so ticket-gated tests don't
  hang waiting on a GUI that doesn't exist.
- **Consecutive-denial tracking:** `begin_ticket_hitl()` /
  `record_hitl_decision()` maintain a process-wide counter keyed by a ticket
  fingerprint (objective + context), reset when the fingerprint changes (a
  genuinely new task) or on Approve, incremented on Deny — surfaced to both
  the GUI and graph state via `get_consecutive_denials()`.

### Archive ledger (`dana/tools/archive_ledger.py`)

A separate, durable Markdown record: `dana_security/patch_ledger.md` holds
`### Ticket:` blocks with a `**Status:** \`[PENDING|RESOLVED|FAILED]\`` line
each. `archive_completed_tickets()` sweeps `RESOLVED`/`FAILED` blocks out into
`patch_ledger_archive.md`, leaving only `PENDING` tickets in the live ledger.
This is a paper trail of what was approved/denied and why — distinct from
the in-memory approval bridge above, which only tracks the *current* pending
decision.

## 3. F12 Panic / Kill-Switch

Implementation: `dana/middleware/kill_switch.py`. Armed at startup in
`dana/core/app_runtime.py` ("Stage 7.2 — hardware panic button").

- **Latch:** `GLOBAL_HALT_EVENT` (a `threading.Event`) is the single source
  of truth for "operators must stop now." `is_halted()` / `halt_if_requested()`
  read it; `clear_global_halt()` resets it once it's safe to resume.
- **Hotkey:** `start_kill_switch_listener()` registers a global hotkey
  (default **F12**, override via `DANA_KILL_HOTKEY`) using the `keyboard`
  package on a daemon thread. Set `DANA_DISABLE_KILL_SWITCH=1` to skip
  arming it (e.g. in a sandboxed CI container without input hook
  permissions) — the panic API (`trigger_halt()`) is still callable
  programmatically even with the hotkey disabled.
- **On trigger** (`trigger_halt()`): sets `GLOBAL_HALT_EVENT`, then calls
  `cancel_action_queue()`, which marks every pending/running Blackboard
  `action_queue` row `cancelled` with reason `"halted by GLOBAL_HALT_EVENT"`.
- **Enforcement at the lowest level:** `EmergencyKillSwitchTriggered`
  subclasses `OSError` specifically so it's caught for free by the
  `except OSError` / `except Exception` handlers already wrapping Win32
  SendInput calls throughout `dana/tools/`. `os_control._check_kill_switch()`
  raises it *before* any hardware input API is touched, so a halt can never
  race a single more keystroke or click through. Operator SEA loops
  (`ghost_typist.py`, `keystroke.py`, `nav_and_click.py`, the `*_actuator.py`
  modules) also poll `halt_if_requested()` directly between steps.

## How the three layers combine

| Layer | Stops what | Scope | Reversible? |
|-------|-----------|-------|-------------|
| Dry-run gate | All Win32 actuation (becomes a logged no-op) | Whole process, until env var is unset/flipped | Yes — flip the flag back |
| HITL ticket | A specific drafted plan, before it starts | One ReAct turn's `ticket_approval` node | Yes — Approve resumes it |
| F12 kill switch | Everything in-flight, immediately | Whole process, instantly | Yes — `clear_global_halt()` |

None of these three require the others to be present to work — a plugin
author does not need to implement any of them directly; using the shared
`os_control`/`*_actuator` primitives and going through the normal ReAct tool
path gets all three automatically.
