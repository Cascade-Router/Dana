"""Headless Meta-Broker / shadow-workspace stress monitor.

Launches ``run.py --no-gui``, injects two complex prompts via ``.trigger_ask``,
streams stdout/stderr for crash signatures, and appends a PSOL to
``logs/context_diagnostic_psol.md`` on failure.

Smart blocking: each prompt waits up to ``PROMPT_BUDGET_S`` (default 300s) for
artifact files and/or completion signatures before the next inject.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_monitor.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRIGGER = ROOT / ".trigger_ask"
PSOL_PATH = ROOT / "logs" / "context_diagnostic_psol.md"
RUNTIME_LOG = ROOT / "logs" / "dana_runtime.log"
SUITE_MARKER = "## Suite Stress — Headless Crash Monitor"

PROMPT_1 = (
    "Use the Meta-Broker to build a new rate-limiting utility for Cascade Router. "
    "Epic 1: Write a pytest suite in tests/test_rate_limiter.py. "
    "Epic 2: Implement rate_limiter.py. "
    "Ensure the runtime harness auto-fixes bugs."
)
PROMPT_2 = (
    "Write a Python Tkinter script named popup_animation.py with a bouncing ball "
    "animation and execute it using the shadow workspace."
)

BOOT_DEADLINE_S = 120.0
# Patience budget for heavy LLM / DAG / REPL work per prompt.
PROMPT_BUDGET_S = 300.0
# After a strong completion signature, require this quiet period (stream idle).
QUIET_AFTER_COMPLETE_S = 8.0
# Hard ceiling for the whole session (boot + 2× prompt budgets + margin).
TOTAL_DEADLINE_S = 780.0

HEAP_SIG = "0xC0000374"
HEAP_EXIT = 0xC0000374  # 3221226356
TRACEBACK_SIG = "Traceback (most recent call last):"

# Strong completion signals — NOT the first Agentic reply / "Action queued".
_COMPLETION_RE = re.compile(
    r"("
    r"All\s+\d+\s+epic\(s\)\s+completed"
    r"|epic\s+\d+\s+validated\s+OK"
    r"|broker_phase['\"]?\s*[:=]\s*['\"]?done"
    r"|status=['\"]completed['\"]"
    r"|graph\s+END"
    r"|MetaBroker\].*completed"
    r"|Task complete"
    r"|runtime harness.*pass"
    r"|pytest.*passed"
    r"|shadow\s+committed"
    r"|OK:\s*committed\s+session"
    r"|popup_animation\.py"
    r"|bouncing\s+ball"
    r")",
    re.I,
)

# Activity that means the agent is still working — resets quiet timer.
_BUSY_RE = re.compile(
    r"("
    r"\[Agentic\]"
    r"|\[Cascade\]"
    r"|\[MoAShim\]"
    r"|\[MetaBroker\]"
    r"|\[DagSupervisor\]"
    r"|\[DagWorker\]"
    r"|tool_call|file_editor|python_repl|write_to_file"
    r"|architect_new_tool|run_terminal_command"
    r"|Loading weights|ChatOllama"
    r")",
    re.I,
)


def _rate_limiter_paths() -> tuple[Path, list[Path]]:
    test_path = ROOT / "tests" / "test_rate_limiter.py"
    impls = [
        ROOT / "rate_limiter.py",
        ROOT / "dana" / "rate_limiter.py",
        ROOT / "execution_jail" / "rate_limiter.py",
        ROOT / "dana" / "tools" / "rate_limiter.py",
    ]
    return test_path, impls


def _popup_paths() -> list[Path]:
    hits = list(ROOT.rglob("popup_animation.py"))
    # Prefer non-venv / non-scratch noise.
    return [
        p
        for p in hits
        if ".venv" not in p.parts
        and "site-packages" not in p.parts
        and ".dana_scratch" not in p.parts
    ]


def artifacts_prompt1_ready() -> bool:
    test_path, impls = _rate_limiter_paths()
    return test_path.is_file() and any(p.is_file() for p in impls)


def artifacts_prompt2_ready() -> bool:
    return bool(_popup_paths())


def artifact_snapshot() -> dict[str, bool]:
    test_path, impls = _rate_limiter_paths()
    return {
        "test_rate_limiter.py": test_path.is_file(),
        "rate_limiter.py": any(p.is_file() for p in impls),
        "popup_animation.py": bool(_popup_paths()),
    }


class StreamBuffer:
    """Thread-safe rolling byte/text capture from a pipe."""

    def __init__(self, *, maxlen: int = 400_000) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self.text = ""
        self.saw_heap = False
        self.saw_traceback = False
        self.total_chars = 0

    def feed(self, data: str) -> None:
        if not data:
            return
        with self._lock:
            self._chunks.append(data)
            self.text = "".join(self._chunks)
            self.total_chars += len(data)
            if len(self.text) > self._maxlen:
                self.text = self.text[-self._maxlen :]
                self._chunks = [self.text]
            low = data
            if HEAP_SIG.lower() in low.lower() or "c0000374" in low.lower():
                self.saw_heap = True
            if TRACEBACK_SIG in data or "Traceback (most recent call last)" in data:
                self.saw_traceback = True

    def snapshot(self, *, tail: int = 6000) -> str:
        with self._lock:
            return self.text[-tail:]

    def chars(self) -> int:
        with self._lock:
            return self.total_chars


def _reader_lines(pipe, buf: StreamBuffer, mirror, label: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            buf.feed(line)
            try:
                mirror.write(f"[{label}] {line}")
                if not line.endswith("\n"):
                    mirror.write("\n")
                mirror.flush()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        buf.feed(f"\n[{label} reader error: {exc}]\n")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _tail_runtime_log(n: int = 80) -> str:
    try:
        if not RUNTIME_LOG.is_file():
            return ""
        lines = RUNTIME_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:  # noqa: BLE001
        return f"(runtime log unread: {exc})"


def _append_psol(
    *,
    problem: str,
    solution: str,
    outcome: str,
    learnings: list[str],
    context: str,
    exit_code: int | None,
) -> Path:
    PSOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Updated: {now}_",
            f"_Exit code: `{exit_code}`_",
            "",
            "### Problem",
            "",
            problem,
            "",
            "### Solution Space",
            "",
            solution,
            "",
            "### Outcome",
            "",
            outcome,
            "",
            "### Learnings",
            "",
            *[f"- {x}" for x in learnings],
            "",
            "### Crash / stream context (tail)",
            "",
            "```text",
            (context or "(empty)")[-6000:],
            "```",
            "",
            "### Runtime log tail",
            "",
            "```text",
            _tail_runtime_log(60) or "(unavailable)",
            "```",
            "",
        ]
    )
    if PSOL_PATH.is_file():
        existing = PSOL_PATH.read_text(encoding="utf-8", errors="replace")
        if SUITE_MARKER in existing:
            head = existing.split(SUITE_MARKER)[0].rstrip()
            PSOL_PATH.write_text(head + "\n" + section, encoding="utf-8")
        else:
            PSOL_PATH.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    else:
        PSOL_PATH.write_text(
            "# Dānā Context Awareness — Aggregated PSOL Diagnostic\n\n"
            f"_Generated: {now}_\n"
            + section,
            encoding="utf-8",
        )
    return PSOL_PATH


def _write_trigger(prompt: str) -> None:
    TRIGGER.write_text(prompt.strip() + "\n", encoding="utf-8")


def _clear_trigger() -> None:
    try:
        if TRIGGER.is_file():
            TRIGGER.unlink()
    except OSError:
        pass


def _wait_boot(proc: subprocess.Popen, buf: StreamBuffer, deadline: float) -> bool:
    needles = (
        "Headless mode: engine auto-ENGAGED",
        "Headless mode (--no-gui)",
        "=== CAMGRASPER Donna voice agent ===",
        "Noise floor calibrated",
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        snap = buf.snapshot(tail=20_000)
        if any(n in snap for n in needles):
            time.sleep(2.0)
            return True
        if buf.saw_heap or buf.saw_traceback:
            return False
        time.sleep(0.25)
    return False


def _wait_prompt_resolved(
    proc: subprocess.Popen,
    buf: StreamBuffer,
    *,
    budget_s: float,
    prior_len: int,
    prior_chars: int,
    artifact_ready,
    label: str,
) -> str:
    """Block until artifacts land, a strong completion signature appears, or budget.

    Does **not** treat the first ``[Conversation] Dānā:`` / Agentic iter as done —
    those fire too early during tool queuing / LLM warmup.
    """
    deadline = time.time() + budget_s
    saw_complete_sig = False
    quiet_since: float | None = None
    last_chars = prior_chars
    last_progress_log = 0.0

    while time.time() < deadline:
        if proc.poll() is not None:
            # Process died — still accept if artifacts already exist.
            if artifact_ready():
                return "artifacts_ready"
            return "process_exited"

        if buf.saw_heap:
            return "heap_corruption"
        if buf.saw_traceback:
            return "traceback"

        if artifact_ready():
            # Artifacts on disk are the hard gate (especially Meta-Broker TDD).
            print(
                f"[monitor] {label}: artifacts present — treating prompt as resolved",
                flush=True,
            )
            time.sleep(1.0)
            return "artifacts_ready"

        snap = buf.snapshot(tail=60_000)
        fresh = snap[prior_len:] if prior_len < len(snap) else snap
        chars = buf.chars()
        growing = chars > last_chars
        if growing:
            last_chars = chars
            quiet_since = None  # still streaming

        if _COMPLETION_RE.search(fresh):
            saw_complete_sig = True
            if quiet_since is None and not growing:
                quiet_since = time.time()
            elif quiet_since is not None and not growing:
                if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                    # Signature alone is weaker than artifacts — only accept if
                    # we also see artifact files OR a very strong epic-complete line.
                    if artifact_ready():
                        return "artifacts_ready"
                    if re.search(
                        r"All\s+\d+\s+epic\(s\)\s+completed|Task complete",
                        fresh,
                        re.I,
                    ):
                        return "completion_signature"
                    # Keep waiting for files until budget ends.
                    quiet_since = time.time()  # extend patience

        if _BUSY_RE.search(fresh[-2000:] if len(fresh) > 2000 else fresh):
            quiet_since = None

        now = time.time()
        if now - last_progress_log >= 30.0:
            remaining = max(0.0, deadline - now)
            arts = artifact_snapshot()
            print(
                f"[monitor] {label}: waiting… {remaining:.0f}s left "
                f"complete_sig={saw_complete_sig} artifacts={arts}",
                flush=True,
            )
            last_progress_log = now

        time.sleep(0.5)

    if artifact_ready():
        return "artifacts_ready"
    if saw_complete_sig:
        return "timeout_with_signature"
    return "timeout"


def _normalize_exit(code: int | None) -> int | None:
    if code is None:
        return None
    if code < 0:
        return code & 0xFFFFFFFF
    return code


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print("=== Dana headless stress monitor ===", flush=True)
    print(f"python={py}", flush=True)
    print(f"launch={run_py} --no-gui", flush=True)
    print(
        f"[monitor] patience: {PROMPT_BUDGET_S:.0f}s/prompt "
        f"(artifact + completion-signature gated)",
        flush=True,
    )

    _clear_trigger()
    buf = StreamBuffer()
    t0 = time.time()

    proc = subprocess.Popen(
        [py, str(run_py), "--no-gui"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None
    threads = [
        threading.Thread(
            target=_reader_lines,
            args=(proc.stdout, buf, sys.stdout, "out"),
            daemon=True,
        ),
        threading.Thread(
            target=_reader_lines,
            args=(proc.stderr, buf, sys.stderr, "err"),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    failure_kind: str | None = None
    prompts_done = 0
    status1 = status2 = ""

    try:
        if not _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S):
            if proc.poll() is not None:
                failure_kind = "boot_exit"
            elif buf.saw_heap:
                failure_kind = "heap_corruption"
            elif buf.saw_traceback:
                failure_kind = "traceback"
            else:
                failure_kind = "boot_timeout"
        else:
            print("[monitor] boot OK — injecting prompt 1 (Meta-Broker TDD)", flush=True)
            prior = len(buf.snapshot(tail=80_000))
            prior_chars = buf.chars()
            _write_trigger(PROMPT_1)
            status1 = _wait_prompt_resolved(
                proc,
                buf,
                budget_s=PROMPT_BUDGET_S,
                prior_len=prior,
                prior_chars=prior_chars,
                artifact_ready=artifacts_prompt1_ready,
                label="prompt1",
            )
            print(f"[monitor] prompt 1 status={status1}", flush=True)
            if status1 in {"heap_corruption", "traceback", "process_exited"}:
                failure_kind = status1
            elif status1 in {"artifacts_ready", "completion_signature"}:
                prompts_done = 1
            elif status1 == "timeout_with_signature" and artifacts_prompt1_ready():
                prompts_done = 1
            else:
                # Timed out without artifacts — still attempt prompt 2 only if
                # process alive; record incomplete TDD.
                print(
                    "[monitor] WARNING: prompt 1 did not produce rate-limiter "
                    "artifacts before budget; not injecting prompt 2 until TDD settles",
                    flush=True,
                )
                # One more short grace if files appear late.
                grace_deadline = time.time() + 45.0
                while time.time() < grace_deadline and proc.poll() is None:
                    if artifacts_prompt1_ready():
                        status1 = "artifacts_ready"
                        prompts_done = 1
                        break
                    time.sleep(1.0)

            if (
                prompts_done >= 1
                and time.time() - t0 < TOTAL_DEADLINE_S
                and proc.poll() is None
            ):
                print(
                    "[monitor] prompt 1 resolved — injecting prompt 2 (Tkinter popup)",
                    flush=True,
                )
                time.sleep(3.0)
                prior = len(buf.snapshot(tail=80_000))
                prior_chars = buf.chars()
                _write_trigger(PROMPT_2)
                status2 = _wait_prompt_resolved(
                    proc,
                    buf,
                    budget_s=PROMPT_BUDGET_S,
                    prior_len=prior,
                    prior_chars=prior_chars,
                    artifact_ready=artifacts_prompt2_ready,
                    label="prompt2",
                )
                print(f"[monitor] prompt 2 status={status2}", flush=True)
                if status2 in {"heap_corruption", "traceback", "process_exited"}:
                    failure_kind = status2
                elif status2 in {"artifacts_ready", "completion_signature"}:
                    prompts_done = 2
                elif artifacts_prompt2_ready():
                    prompts_done = 2
    finally:
        if proc.poll() is None:
            print("[monitor] terminating headless agent...", flush=True)
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait(timeout=5)
        for t in threads:
            t.join(timeout=2.0)
        _clear_trigger()

    exit_code = _normalize_exit(proc.returncode)
    created = artifact_snapshot()
    print(
        f"[monitor] child exit_code={exit_code} prompts_done={prompts_done} "
        f"status1={status1!r} status2={status2!r}",
        flush=True,
    )
    print(f"[monitor] artifacts: {created}", flush=True)

    crash = False
    reasons: list[str] = []
    if buf.saw_heap or (exit_code is not None and exit_code == HEAP_EXIT):
        crash = True
        reasons.append("Windows heap corruption (0xC0000374)")
    if buf.saw_traceback:
        crash = True
        reasons.append("Python Traceback detected in stream")
    if failure_kind in {
        "boot_timeout",
        "boot_exit",
        "heap_corruption",
        "traceback",
        "process_exited",
    }:
        crash = True
        reasons.append(f"monitor failure_kind={failure_kind}")
    if exit_code not in (0, None) and exit_code != HEAP_EXIT:
        if failure_kind or buf.saw_heap or buf.saw_traceback:
            crash = True
            reasons.append(f"nonzero exit_code={exit_code}")

    context = buf.snapshot(tail=8000)

    if crash:
        path = _append_psol(
            problem=(
                "Headless stress monitor detected a crash or failure while injecting "
                "Meta-Broker + shadow-workspace prompts. "
                + "; ".join(reasons)
            ),
            solution=(
                "1. Isolate MicIngest / PortAudio from headless DAG runs "
                "(disable audio threads under --no-gui).\n"
                "2. Harden Meta-Broker / RuntimeHarness against heap-corrupting "
                "native extensions (torch/PyAudio) during long tool loops.\n"
                "3. Re-run stress monitor after audio isolation; confirm both "
                "prompts complete with artifacts on disk."
            ),
            outcome=(
                f"failure_kind={failure_kind!r}; exit_code={exit_code}; "
                f"prompts_done={prompts_done}; status1={status1!r}; "
                f"status2={status2!r}; artifacts={created}"
            ),
            learnings=[
                "0xC0000374 on Windows often appears after PortAudio / CUDA / Tk "
                "interactions during concurrent agent work.",
                "Headless injects require engine ENGAGE; STANDBY silently drops "
                ".trigger_ask payloads.",
                "First Agentic reply is not task completion — wait for artifacts "
                "or epic-complete signatures (300s patience).",
            ],
            context=context,
            exit_code=exit_code,
        )
        print(f"[monitor] FAIL — PSOL appended → {path}", flush=True)
        print(f"[monitor] reasons: {reasons}", flush=True)
        return 1

    if not created["test_rate_limiter.py"] or not created["rate_limiter.py"]:
        print(
            "[monitor] INCOMPLETE — no crash, but Meta-Broker TDD artifacts missing",
            flush=True,
        )
        return 2
    if not created["popup_animation.py"]:
        print(
            "[monitor] INCOMPLETE — rate-limiter OK but popup_animation.py missing",
            flush=True,
        )
        return 2

    print("[monitor] PASS — no crash; both artifact sets present", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
