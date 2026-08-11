"""Phase 1/2 — Adaptive compute governor (idle monitor + background research).

Polls Win32 ``GetLastInputInfo`` on a daemon thread. While the user is active
(idle < 5 minutes), background child processes are deprioritized. When the user
is away (idle >= 5 minutes), priorities restore and one synthetic research task
is injected into ``execution_jail/input.txt`` for the existing InputIngest →
task_queue → ``run_react_loop`` path (no new agent loops).
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path
from typing import Any, Deque, Literal

IdleState = Literal["USER_ACTIVE", "USER_AWAY"]

USER_ACTIVE: IdleState = "USER_ACTIVE"
USER_AWAY: IdleState = "USER_AWAY"

IDLE_THRESHOLD_S = 300.0
POLL_INTERVAL_S = 2.0

# Swarm-priority rotating research topics (Phase 2).
DEFAULT_RESEARCH_TOPICS: tuple[str, ...] = (
    "[BACKGROUND TASK] Deep research into distributed multi-agent coordination "
    "frameworks using ROS2 and Nav2. Summarize the architectural patterns and "
    "ingest into Chroma.",
    "[BACKGROUND TASK] Background research on optimizing PyTorch deep learning "
    "pipelines for end-to-end autonomous navigation. Extract key techniques and "
    "ingest into the vault.",
    "[BACKGROUND TASK] Deep research on best practices for integrating headless "
    "browser automation with ChromaDB indexing for local AI assistants. "
    "Consolidate memory with the findings.",
    "[BACKGROUND TASK] Search the web for recent advanced C++ patterns for "
    "low-latency robotic motion planning. Summarize and ingest into the vault.",
)

_LOCK = threading.Lock()
_STATE: IdleState = USER_ACTIVE
_HEAVY_COMPUTE = threading.Event()  # set when USER_AWAY
_MONITOR: "IdleMonitor | None" = None
_STARTED = False
_START_LOCK = threading.Lock()
# PIDs exempt from the USER_ACTIVE ~65% throttle (sandbox / heavy jobs).
_PRIORITY_WHITELIST: set[int] = set()
_JOB_PIDS: dict[str, int] = {}
# Fraction of logical CPUs background children may use while USER_ACTIVE.
_ACTIVE_CPU_FRACTION = 0.65


class ProactiveNotificationQueue:
    """Thread-safe inbox for background job/swarm completions while USER_AWAY."""

    def __init__(self, *, maxlen: int = 64) -> None:
        self._lock = threading.Lock()
        self._items: Deque[dict[str, Any]] = deque(maxlen=max(1, int(maxlen)))

    def push(
        self,
        *,
        job_id: str,
        status: str,
        summary: str,
        kind: str = "job",
    ) -> None:
        event = {
            "job_id": str(job_id or "").strip() or "unknown",
            "status": str(status or "").strip() or "completed",
            "summary": str(summary or "").strip()[:400],
            "kind": str(kind or "job").strip() or "job",
            "ts": time.time(),
        }
        with self._lock:
            self._items.append(event)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out

    def peek(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


_PROACTIVE_Q = ProactiveNotificationQueue()


def push_proactive_notification(
    *,
    job_id: str,
    status: str,
    summary: str,
    kind: str = "job",
) -> None:
    """Enqueue a structured background-completion event (any idle state)."""
    _PROACTIVE_Q.push(
        job_id=job_id, status=status, summary=summary, kind=kind
    )


def drain_proactive_notifications() -> list[dict[str, Any]]:
    """Pop and clear all pending proactive events."""
    return _PROACTIVE_Q.drain()


def queue_if_user_away(
    *,
    job_id: str,
    status: str,
    summary: str,
    kind: str = "job",
) -> bool:
    """Push only when ``USER_AWAY``; return True if queued."""
    try:
        if get_idle_state() != USER_AWAY:
            return False
    except Exception:  # noqa: BLE001
        return False
    push_proactive_notification(
        job_id=job_id, status=status, summary=summary, kind=kind
    )
    return True


def _deliver_proactive_briefing() -> int:
    """On USER_ACTIVE: StatusEventBus badge + optional voice, then clear queue."""
    events = drain_proactive_notifications()
    if not events:
        return 0
    n = len(events)
    first = events[0]
    status = str(first.get("status") or "completed")
    job_id = str(first.get("job_id") or "job")
    kind = str(first.get("kind") or "job")
    detail = str(first.get("summary") or "").strip()
    if n == 1:
        msg = (
            f"Your background {kind} ({job_id}) finished with status {status}."
        )
        if detail:
            msg = f"{msg} {detail[:160]}"
        spoken = (
            f"Your background job finished with status {status}."
        )
    else:
        msg = (
            f"{n} background jobs finished while you were away "
            f"(latest: {job_id} → {status})."
        )
        spoken = (
            f"You have {n} background job updates. "
            f"The latest finished with status {status}."
        )
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change(
            "idle",
            tool="proactive_briefing",
            message=msg,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"WARNING: proactive StatusEventBus emit failed ({exc})")
    try:
        from dana.audio.tts_manager import enqueue_speech_impl as enqueue_speech

        enqueue_speech(spoken)
    except Exception:  # noqa: BLE001
        pass
    _log(f"Proactive briefing delivered ({n} event(s))")
    return n


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("IdleMonitor", msg)
    except Exception:  # noqa: BLE001
        print(f"[IdleMonitor] {msg}", flush=True)


def idle_seconds() -> float:
    """Seconds since last OS input (Win32). Returns 0.0 on failure / non-Windows."""
    if os.name != "nt":
        return 0.0
    try:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 0.0
        tick = int(ctypes.windll.kernel32.GetTickCount())
        idle_ms = (tick - int(lii.dwTime)) & 0xFFFFFFFF
        return float(idle_ms) / 1000.0
    except Exception:  # noqa: BLE001
        return 0.0


def get_idle_state() -> IdleState:
    with _LOCK:
        return _STATE


def heavy_compute_cleared() -> bool:
    """True when USER_AWAY — heavy background work is cleared to run."""
    return _HEAVY_COMPUTE.is_set()


def ollama_keep_alive() -> int | str:
    """Ollama keep_alive for API payloads.

    Default is ``0`` (immediate unload after each call) for zero-latency VRAM
    reclaim during multi-epic runs. Override with ``DANA_OLLAMA_KEEP_ALIVE``
    (e.g. ``5m``) when warm-cache is preferred. Never returns ``-1`` / ``"-1"``.
    """
    override = (os.environ.get("DANA_OLLAMA_KEEP_ALIVE") or "").strip()
    if override:
        if override in {"-1", "-1.0"}:
            return 0
        try:
            return int(override)
        except ValueError:
            return override
    # Zero-latency memory management — unload weights after every generation.
    return 0


def _set_state(state: IdleState) -> None:
    global _STATE
    with _LOCK:
        _STATE = state
    if state == USER_AWAY:
        _HEAVY_COMPUTE.set()
    else:
        _HEAVY_COMPUTE.clear()


def _emit_compute_mode(state: IdleState) -> None:
    try:
        from dana.ui.status_bus import emit_state_change

        if state == USER_AWAY:
            emit_state_change(
                "idle",
                tool="compute_high",
                message="Compute: high (~80%) — USER_AWAY",
            )
        else:
            emit_state_change(
                "idle",
                tool="compute_low",
                message="Compute: medium (~65%) — USER_ACTIVE",
            )
    except Exception:  # noqa: BLE001
        pass


def _boost_pid(pid: int) -> None:
    """Raise one child to NORMAL/HIGH so a sandbox job can finish promptly."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return
    try:
        proc = psutil.Process(int(pid))
    except Exception:  # noqa: BLE001
        return
    for name in ("HIGH_PRIORITY_CLASS", "ABOVE_NORMAL_PRIORITY_CLASS", "NORMAL_PRIORITY_CLASS"):
        val = getattr(psutil, name, None)
        if val is None:
            continue
        try:
            proc.nice(val)
            return
        except Exception:  # noqa: BLE001
            continue


def register_priority_override(pid: int, *, job_id: str = "") -> None:
    """Whitelist a child PID so USER_ACTIVE throttling does not cap it (~65%)."""
    try:
        pid_i = int(pid)
    except Exception:  # noqa: BLE001
        return
    if pid_i <= 0:
        return
    with _LOCK:
        _PRIORITY_WHITELIST.add(pid_i)
        key = str(job_id or "").strip()
        if key:
            _JOB_PIDS[key] = pid_i
    _boost_pid(pid_i)
    _log(f"Priority override ON pid={pid_i} job_id={job_id or '-'}")


def unregister_priority_override(
    pid: int | None = None,
    *,
    job_id: str = "",
) -> None:
    """Remove a whitelist entry and re-apply the current idle throttle policy."""
    pid_i: int | None = None
    with _LOCK:
        key = str(job_id or "").strip()
        if key and key in _JOB_PIDS:
            pid_i = _JOB_PIDS.pop(key, None)
        if pid is not None:
            try:
                pid_i = int(pid)
            except Exception:  # noqa: BLE001
                pid_i = pid_i
        if pid_i is not None:
            _PRIORITY_WHITELIST.discard(int(pid_i))
    if pid_i is not None:
        _log(f"Priority override OFF pid={pid_i} job_id={job_id or '-'}")
    # Re-apply global policy so non-whitelisted children stay throttled.
    _apply_child_priorities(get_idle_state())


def _active_cpu_affinity() -> list[int] | None:
    """Logical CPU indices for the ~65% USER_ACTIVE quota (at least one core)."""
    try:
        import psutil

        n = int(psutil.cpu_count(logical=True) or 0)
    except Exception:  # noqa: BLE001
        return None
    if n <= 0:
        return None
    keep = max(1, int(round(n * float(_ACTIVE_CPU_FRACTION))))
    keep = min(n, keep)
    return list(range(keep))


def _set_child_affinity(child: Any, cpus: list[int] | None) -> None:
    if cpus is None:
        return
    try:
        child.cpu_affinity(list(cpus))
    except Exception:  # noqa: BLE001
        pass


def _restore_child_affinity(child: Any) -> None:
    try:
        import psutil

        n = int(psutil.cpu_count(logical=True) or 0)
        if n > 0:
            child.cpu_affinity(list(range(n)))
    except Exception:  # noqa: BLE001
        pass


def _apply_child_priorities(state: IdleState) -> None:
    """Throttle background child processes; never touch the main agent PID.

    Whitelisted PIDs (active sandbox / heavy jobs) stay at NORMAL/HIGH even
    while ``USER_ACTIVE`` applies the ~65% cap to everyone else.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return

    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
    except Exception:  # noqa: BLE001
        return

    with _LOCK:
        whitelist = set(_PRIORITY_WHITELIST)

    # Drop dead whitelist entries.
    alive = {c.pid for c in children}
    stale = whitelist - alive
    if stale:
        with _LOCK:
            for dead in stale:
                _PRIORITY_WHITELIST.discard(dead)
            for jid, p in list(_JOB_PIDS.items()):
                if p in stale:
                    _JOB_PIDS.pop(jid, None)
        whitelist -= stale

    if state == USER_ACTIVE:
        # ~65% quota: BELOW_NORMAL (not IDLE) + affinity to ~65% of logical CPUs.
        prio = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
        if prio is None:
            prio = getattr(psutil, "NORMAL_PRIORITY_CLASS", None)
        affinity = _active_cpu_affinity()
        for child in children:
            if child.pid in whitelist:
                _boost_pid(child.pid)
                _restore_child_affinity(child)
                continue
            if prio is not None:
                try:
                    child.nice(prio)
                except Exception:  # noqa: BLE001
                    pass
            _set_child_affinity(child, affinity)
    else:
        normal = getattr(psutil, "NORMAL_PRIORITY_CLASS", None)
        for child in children:
            if child.pid in whitelist:
                _boost_pid(child.pid)
                _restore_child_affinity(child)
                continue
            if normal is not None:
                try:
                    child.nice(normal)
                except Exception:  # noqa: BLE001
                    continue
            _restore_child_affinity(child)


def compress_idle_research_output(raw_text: str, *, topic: str = "") -> str:
    """Phase 5 hook: compress completed USER_AWAY research into ``idle_compressed``.

    Called from the Conversation drain path after ``[BACKGROUND TASK]`` ReAct turns.
    """
    try:
        from dana.memory.compressor import ingest_idle_compressed

        return ingest_idle_compressed(
            raw_text,
            source="idle_research",
            topic=topic,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: compress_idle_research_output failed: {exc}"


def _inject_research_via_input_txt(payload: str) -> bool:
    """Append one topic to input.txt and wake Conversation via empty .trigger_ask.

    Reuses InputIngest → task_queue → drain_structured_task_queue → run_react_loop.
    """
    text = (payload or "").strip()
    if not text:
        return False
    try:
        from dana.paths import TEXT_INJECTION_PATH, TRIGGER_ASK_PATH

        target = Path(TEXT_INJECTION_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n\n")
        preview = text if len(text) <= 120 else text[:117] + "..."
        _log(f'Queued background research via input.txt: "{preview}"')
    except Exception as exc:  # noqa: BLE001
        _log(f"WARNING: input.txt inject failed ({exc})")
        return False

    # Wake Conversation when idle (agent_loop leaves the file if busy).
    try:
        Path(TRIGGER_ASK_PATH).write_text("", encoding="utf-8")
        _log("Wrote empty .trigger_ask to wake agent for background research")
    except Exception as exc:  # noqa: BLE001
        _log(f"WARNING: trigger wake failed ({exc})")
    return True


class IdleMonitor:
    """Background daemon that toggles USER_ACTIVE / USER_AWAY and injects research."""

    def __init__(
        self,
        *,
        threshold_s: float = IDLE_THRESHOLD_S,
        poll_s: float = POLL_INTERVAL_S,
        topics: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.threshold_s = float(threshold_s)
        self.poll_s = float(poll_s)
        seed = list(topics) if topics is not None else list(DEFAULT_RESEARCH_TOPICS)
        self._topics: Deque[str] = deque(seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(
            target=self._run,
            name="IdleMonitor",
            daemon=True,
        )
        t.start()
        self._thread = t
        _log(
            f"Started (threshold={self.threshold_s:.0f}s, "
            f"topics={len(self._topics)}, state={get_idle_state()})"
        )

    def stop(self) -> None:
        self._stop.set()

    def _next_topic(self) -> str | None:
        if not self._topics:
            return None
        topic = self._topics[0]
        self._topics.rotate(-1)
        return topic

    def _on_enter_active(self) -> None:
        _set_state(USER_ACTIVE)
        _apply_child_priorities(USER_ACTIVE)
        _emit_compute_mode(USER_ACTIVE)
        _log("Transition → USER_ACTIVE (~65% background / warm Ollama 5m)")
        try:
            _deliver_proactive_briefing()
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: proactive briefing failed ({exc})")

    def _on_enter_away(self) -> None:
        _set_state(USER_AWAY)
        _apply_child_priorities(USER_AWAY)
        _emit_compute_mode(USER_AWAY)
        _log("Transition → USER_AWAY (high-compute / unload Ollama after infer)")
        topic = self._next_topic()
        if topic:
            _inject_research_via_input_txt(topic)

    def _run(self) -> None:
        # Seed initial state from current idle age (no research inject on boot).
        idle = idle_seconds()
        if idle >= self.threshold_s:
            _set_state(USER_AWAY)
            _apply_child_priorities(USER_AWAY)
            _emit_compute_mode(USER_AWAY)
            _log(f"Boot state USER_AWAY (idle={idle:.1f}s); research inject deferred")
        else:
            _set_state(USER_ACTIVE)
            _apply_child_priorities(USER_ACTIVE)
            _emit_compute_mode(USER_ACTIVE)
            _log(f"Boot state USER_ACTIVE (idle={idle:.1f}s)")

        prev = get_idle_state()
        while not self._stop.is_set():
            try:
                idle = idle_seconds()
                nxt: IdleState = (
                    USER_AWAY if idle >= self.threshold_s else USER_ACTIVE
                )
                if nxt != prev:
                    if nxt == USER_AWAY:
                        self._on_enter_away()
                    else:
                        self._on_enter_active()
                    prev = nxt
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: poll failed ({exc})")
            self._stop.wait(timeout=self.poll_s)
        _log("Stopped.")


def start_idle_monitor(
    *,
    threshold_s: float = IDLE_THRESHOLD_S,
    poll_s: float = POLL_INTERVAL_S,
) -> IdleMonitor | None:
    """Idempotent start of the IdleMonitor daemon."""
    global _MONITOR, _STARTED
    with _START_LOCK:
        if _STARTED and _MONITOR is not None:
            return _MONITOR
        if os.environ.get("DANA_DISABLE_IDLE_MONITOR", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _log("disabled via DANA_DISABLE_IDLE_MONITOR")
            return None
        mon = IdleMonitor(threshold_s=threshold_s, poll_s=poll_s)
        mon.start()
        _MONITOR = mon
        _STARTED = True
        return mon


def stop_idle_monitor() -> None:
    global _STARTED
    with _START_LOCK:
        if _MONITOR is not None:
            _MONITOR.stop()
        _STARTED = False


__all__ = (
    "DEFAULT_RESEARCH_TOPICS",
    "IDLE_THRESHOLD_S",
    "USER_ACTIVE",
    "USER_AWAY",
    "IdleMonitor",
    "ProactiveNotificationQueue",
    "compress_idle_research_output",
    "drain_proactive_notifications",
    "get_idle_state",
    "heavy_compute_cleared",
    "idle_seconds",
    "ollama_keep_alive",
    "push_proactive_notification",
    "queue_if_user_away",
    "register_priority_override",
    "start_idle_monitor",
    "stop_idle_monitor",
    "unregister_priority_override",
)
