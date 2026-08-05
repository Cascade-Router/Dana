"""Diagnostic Suite 7 — headless E2E quality audit (Meta-Broker + Tkinter).

Launches ``run.py --no-gui``, injects two stress prompts via ``.trigger_ask``,
runs post-execution quality gates (pytest / py_compile / AST content checks),
and appends ``### Diagnostic Suite 7: E2E Quality Audit`` to
``logs/context_diagnostic_psol.md``.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_quality_audit.py
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRIGGER = ROOT / ".trigger_ask"
PSOL_PATH = ROOT / "logs" / "context_diagnostic_psol.md"
RUNTIME_LOG = ROOT / "logs" / "dana_runtime.log"
SUITE_MARKER = "### Diagnostic Suite 7: E2E Quality Audit"

PROMPT_1 = (
    "/broker Build a new rate-limiting utility for Cascade Router. "
    "Epic 1: Write a pytest suite in tests/test_rate_limiter.py. "
    "Epic 2: Implement rate_limiter.py. "
    "Ensure the runtime harness auto-fixes bugs."
)
PROMPT_2 = (
    "Write a simple Python Tkinter script named popup_animation.py with a "
    "bouncing ball animation and execute it using the shadow workspace."
)

BOOT_DEADLINE_S = 120.0
# Generous budget for local LLM + Meta-Broker closed-loop repair.
PROMPT_BUDGET_S = 600.0
QUIET_AFTER_COMPLETE_S = 8.0
TOTAL_DEADLINE_S = 1500.0

HEAP_SIG = "0xC0000374"
HEAP_EXIT = 0xC0000374
TRACEBACK_SIG = "Traceback (most recent call last):"

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


def _rate_limiter_impl_candidates() -> list[Path]:
    return [
        ROOT / "rate_limiter.py",
        ROOT / "dana" / "rate_limiter.py",
        ROOT / "execution_jail" / "rate_limiter.py",
        ROOT / "dana" / "tools" / "rate_limiter.py",
        ROOT / "execution_jail" / "broker_diag" / "rate_limiter.py",
    ]


def _find_rate_limiter() -> Path | None:
    for p in _rate_limiter_impl_candidates():
        if p.is_file():
            return p
    # Last resort: recent non-venv hits named rate_limiter.py
    hits = [
        p
        for p in ROOT.rglob("rate_limiter.py")
        if ".venv" not in p.parts
        and "site-packages" not in p.parts
        and ".dana_scratch" not in p.parts
    ]
    return hits[0] if hits else None


def _test_rate_limiter_path() -> Path:
    return ROOT / "tests" / "test_rate_limiter.py"


def _find_popup() -> Path | None:
    hits = [
        p
        for p in ROOT.rglob("popup_animation.py")
        if ".venv" not in p.parts
        and "site-packages" not in p.parts
        and ".dana_scratch" not in p.parts
    ]
    preferred = ROOT / "popup_animation.py"
    if preferred.is_file():
        return preferred
    return hits[0] if hits else None


def artifacts_prompt1_ready() -> bool:
    """True when both rate-limiter artifacts exist and TokenBucket imports cleanly."""
    if not (_test_rate_limiter_path().is_file() and _find_rate_limiter() is not None):
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)]
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from rate_limiter import TokenBucket; TokenBucket",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return int(proc.returncode) == 0


def artifacts_prompt2_ready() -> bool:
    """True only when popup_animation.py is valid Python with Tkinter markers.

    Stale HTML dumps from prior runs must not short-circuit the Task 2 wait loop.
    """
    path = _find_popup()
    if path is None or not path.is_file():
        return False
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if re.search(r"(?i)<(?:!DOCTYPE\s+html|html|script|style)\b", body):
        return False
    if "tkinter" not in body.lower() and "Tkinter" not in body:
        return False
    compile_proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    return int(compile_proc.returncode) == 0


class StreamBuffer:
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
            if HEAP_SIG.lower() in data.lower() or "c0000374" in data.lower():
                self.saw_heap = True
            # Ignore subprocess pipe reader encoding faults (cp1252) — not agent crashes.
            if TRACEBACK_SIG in data and "_readerthread" not in data:
                self.saw_traceback = True

    def snapshot(self, *, tail: int = 6000) -> str:
        with self._lock:
            return self.text[-tail:]

    def chars(self) -> int:
        with self._lock:
            return self.total_chars


def _reader_bytes(pipe, buf: StreamBuffer, mirror, label: str) -> None:
    """Read subprocess pipes as bytes; decode with ``errors='replace'``.

    Avoids Windows cp1252 ``UnicodeDecodeError`` crashes on UTF-8 agent logs.
    """
    pending = ""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", errors="replace")
            else:
                text = str(chunk)
            pending += text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line + "\n"
                buf.feed(line)
                try:
                    mirror.write(f"[{label}] {line}")
                    mirror.flush()
                except Exception:
                    pass
        if pending:
            buf.feed(pending)
            try:
                mirror.write(f"[{label}] {pending}")
                if not pending.endswith("\n"):
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


def _force_hybrid_planner_for_audit() -> bool:
    """Enable Hybrid Broker in settings.json and load ``.env`` API keys.

    Returns the previous ``hybrid_planner_enabled`` value for restore.
    """
    from dana.graph.cloud_planner import (
        cloud_planner_key_present,
        ensure_dotenv_loaded,
        planner_mode_label,
    )
    from dana.settings import (
        is_hybrid_planner_enabled,
        set_hybrid_planner_enabled,
    )

    ensure_dotenv_loaded()
    previous = bool(is_hybrid_planner_enabled())
    set_hybrid_planner_enabled(True)
    # Also mirror into dana/settings.json if present (docs refer to this path).
    dana_settings = ROOT / "dana" / "settings.json"
    try:
        import json as _json

        from dana.paths import SETTINGS_PATH

        payload = {}
        if SETTINGS_PATH.is_file():
            payload = _json.loads(
                SETTINGS_PATH.read_text(encoding="utf-8", errors="replace")
            )
        if not isinstance(payload, dict):
            payload = {}
        payload["hybrid_planner_enabled"] = True
        SETTINGS_PATH.write_text(
            _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        dana_settings.parent.mkdir(parents=True, exist_ok=True)
        dana_settings.write_text(
            _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa] WARNING: could not mirror settings.json ({exc})", flush=True)

    key_ok = cloud_planner_key_present()
    mode = planner_mode_label()
    print(
        f"[qa] hybrid_planner_enabled=True (was {previous}); "
        f"api_key_present={key_ok}; planner_mode=[{mode}]",
        flush=True,
    )
    if not key_ok:
        print(
            "[qa] WARNING: No GEMINI_API_KEY / GOOGLE_API_KEY in .env — "
            "Hybrid toggle is on but planner will fall back to LOCAL Ollama.",
            flush=True,
        )
    return previous


def _restore_hybrid_planner(previous: bool | None) -> None:
    if previous is None:
        return
    try:
        from dana.settings import set_hybrid_planner_enabled

        set_hybrid_planner_enabled(bool(previous))
        print(
            f"[qa] restored hybrid_planner_enabled={bool(previous)}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa] WARNING: could not restore hybrid flag ({exc})", flush=True)


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
    deadline = time.time() + budget_s
    saw_complete_sig = False
    quiet_since: float | None = None
    last_chars = prior_chars
    last_progress_log = 0.0

    while time.time() < deadline:
        if proc.poll() is not None:
            if artifact_ready():
                return "artifacts_ready"
            return "process_exited"
        if buf.saw_heap:
            return "heap_corruption"
        if buf.saw_traceback:
            return "traceback"
        if artifact_ready():
            print(
                f"[qa] {label}: artifacts present — resolving",
                flush=True,
            )
            time.sleep(1.5)
            return "artifacts_ready"

        snap = buf.snapshot(tail=60_000)
        fresh = snap[prior_len:] if prior_len < len(snap) else snap
        chars = buf.chars()
        growing = chars > last_chars
        if growing:
            last_chars = chars
            quiet_since = None

        if _COMPLETION_RE.search(fresh):
            saw_complete_sig = True
            if quiet_since is None and not growing:
                quiet_since = time.time()
            elif quiet_since is not None and not growing:
                if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                    if artifact_ready():
                        return "artifacts_ready"
                    if re.search(
                        r"All\s+\d+\s+epic\(s\)\s+completed|Task complete",
                        fresh,
                        re.I,
                    ):
                        return "completion_signature"
                    quiet_since = time.time()

        if _BUSY_RE.search(fresh[-2000:] if len(fresh) > 2000 else fresh):
            quiet_since = None

        now = time.time()
        if now - last_progress_log >= 30.0:
            remaining = max(0.0, deadline - now)
            print(
                f"[qa] {label}: waiting… {remaining:.0f}s left "
                f"complete_sig={saw_complete_sig} "
                f"p1={artifacts_prompt1_ready()} p2={artifacts_prompt2_ready()}",
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


def audit_task1_rate_limiter() -> dict[str, Any]:
    """Quality Audit 1 — files exist + pytest suite passes."""
    test_path = _test_rate_limiter_path()
    impl = _find_rate_limiter()
    result: dict[str, Any] = {
        "task": "Meta-Broker TDD (rate limiter)",
        "test_path": str(test_path),
        "impl_path": str(impl) if impl else None,
        "test_exists": test_path.is_file(),
        "impl_exists": impl is not None and impl.is_file(),
        "pytest_exit_code": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    missing = []
    if not result["test_exists"]:
        missing.append("tests/test_rate_limiter.py")
    if not result["impl_exists"]:
        missing.append("rate_limiter.py")
    if missing:
        result["detail"] = f"missing files: {', '.join(missing)}"
        return result

    # Ensure pytest can import the implementation from project root / impl dir.
    env = os.environ.copy()
    py_paths = [str(ROOT)]
    if impl is not None:
        py_paths.insert(0, str(impl.parent))
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_paths + ([prev] if prev else []))

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result["pytest_exit_code"] = -1
        result["pytest_stderr"] = "TIMEOUT: pytest exceeded 60s (likely import-time hang)"
        result["detail"] = result["pytest_stderr"]
        return result
    result["pytest_exit_code"] = int(proc.returncode)
    result["pytest_stdout"] = (proc.stdout or "")[-2000:]
    result["pytest_stderr"] = (proc.stderr or "")[-2000:]
    if proc.returncode == 0:
        result["verdict"] = "PASS"
        result["detail"] = "pytest exit_code=0"
    else:
        result["detail"] = (
            f"pytest exit_code={proc.returncode}; "
            f"stderr={(proc.stderr or '')[:400]!r}"
        )
    return result


def audit_task2_popup_animation() -> dict[str, Any]:
    """Quality Audit 2 — file exists, compiles, contains tkinter + Canvas."""
    path = _find_popup()
    result: dict[str, Any] = {
        "task": "Tkinter bouncing-ball animation",
        "path": str(path) if path else None,
        "exists": path is not None and path.is_file(),
        "py_compile_exit_code": None,
        "py_compile_stderr": "",
        "has_tkinter_import": False,
        "has_canvas": False,
        "verdict": "FAIL",
        "detail": "",
    }
    if not result["exists"] or path is None:
        result["detail"] = "missing file: popup_animation.py"
        return result

    compile_proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    result["py_compile_exit_code"] = int(compile_proc.returncode)
    result["py_compile_stderr"] = (compile_proc.stderr or "")[-1000:]
    if compile_proc.returncode != 0:
        result["detail"] = (
            f"py_compile exit_code={compile_proc.returncode}; "
            f"stderr={(compile_proc.stderr or '')[:400]!r}"
        )
        return result

    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["detail"] = f"read failed: {exc}"
        return result

    has_import = bool(
        re.search(
            r"(?m)^\s*(import\s+tkinter(\s+as\s+\w+)?|from\s+tkinter\s+import\b)",
            src,
        )
    )
    has_canvas = bool(re.search(r"\bCanvas\b", src))
    result["has_tkinter_import"] = has_import
    result["has_canvas"] = has_canvas
    if has_import and has_canvas:
        result["verdict"] = "PASS"
        result["detail"] = "py_compile=0; tkinter import + Canvas present"
    else:
        gaps = []
        if not has_import:
            gaps.append("missing `import tkinter` / `import tkinter as tk`")
        if not has_canvas:
            gaps.append("missing Canvas widget")
        result["detail"] = "; ".join(gaps)
    return result


def _tail_runtime_log(n: int = 60) -> str:
    try:
        if not RUNTIME_LOG.is_file():
            return ""
        lines = RUNTIME_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:  # noqa: BLE001
        return f"(runtime log unread: {exc})"


def append_suite7_psol(
    *,
    audit1: dict[str, Any],
    audit2: dict[str, Any],
    status1: str,
    status2: str,
    engine_exit: int | None,
    stream_tail: str,
) -> Path:
    PSOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    v1 = audit1.get("verdict", "FAIL")
    v2 = audit2.get("verdict", "FAIL")
    overall = "PASS" if v1 == "PASS" and v2 == "PASS" else "FAIL"

    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Updated: {now}_",
            f"_Overall: `{overall}`_",
            f"_Engine exit: `{engine_exit}`_",
            f"_Inject status: prompt1=`{status1}` prompt2=`{status2}`_",
            "",
            "Headless E2E quality audit of Meta-Broker TDD + Tkinter shadow-workspace "
            "artifacts (pytest / py_compile / content gates).",
            "",
            "| Task | Verdict | Detail |",
            "|---|---|---|",
            (
                f"| `7.1` Meta-Broker rate limiter | `{v1}` | "
                f"test_exists={audit1.get('test_exists')}; "
                f"impl_exists={audit1.get('impl_exists')}; "
                f"pytest_exit={audit1.get('pytest_exit_code')}; "
                f"impl=`{audit1.get('impl_path')}`; {audit1.get('detail')} |"
            ),
            (
                f"| `7.2` Tkinter popup_animation | `{v2}` | "
                f"exists={audit2.get('exists')}; "
                f"py_compile_exit={audit2.get('py_compile_exit_code')}; "
                f"tkinter={audit2.get('has_tkinter_import')}; "
                f"Canvas={audit2.get('has_canvas')}; "
                f"path=`{audit2.get('path')}`; {audit2.get('detail')} |"
            ),
            "",
            "#### Task 7.1 — Meta-Broker TDD",
            "",
            f"- **Verdict:** `{v1}`",
            f"- **Prompt:** `{PROMPT_1}`",
            f"- **test_rate_limiter.py:** `{audit1.get('test_path')}` "
            f"(exists={audit1.get('test_exists')})",
            f"- **rate_limiter.py:** `{audit1.get('impl_path')}` "
            f"(exists={audit1.get('impl_exists')})",
            f"- **pytest exit_code:** `{audit1.get('pytest_exit_code')}`",
            f"- **Detail:** {audit1.get('detail')}",
            "",
            "```text",
            (audit1.get("pytest_stdout") or "(no pytest stdout)")[-1500:],
            "```",
            "",
            "#### Task 7.2 — Tkinter animation",
            "",
            f"- **Verdict:** `{v2}`",
            f"- **Prompt:** `{PROMPT_2}`",
            f"- **popup_animation.py:** `{audit2.get('path')}` "
            f"(exists={audit2.get('exists')})",
            f"- **py_compile exit_code:** `{audit2.get('py_compile_exit_code')}`",
            f"- **import tkinter:** `{audit2.get('has_tkinter_import')}`",
            f"- **Canvas widget:** `{audit2.get('has_canvas')}`",
            f"- **Detail:** {audit2.get('detail')}",
            "",
            "### Runtime log tail",
            "",
            "```text",
            _tail_runtime_log(50) or "(unavailable)",
            "```",
            "",
            "### Stream tail",
            "",
            "```text",
            (stream_tail or "(empty)")[-4000:],
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


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print("=== Dana Diagnostic Suite 7 — E2E Quality Audit ===", flush=True)
    print(f"python={py}", flush=True)
    print(f"launch={run_py} --no-gui", flush=True)
    print(f"[qa] patience: {PROMPT_BUDGET_S:.0f}s/prompt", flush=True)

    previous_hybrid: bool | None = None
    try:
        previous_hybrid = _force_hybrid_planner_for_audit()
    except Exception as exc:  # noqa: BLE001
        print(f"[qa] WARNING: hybrid setup failed ({exc})", flush=True)

    _clear_trigger()
    buf = StreamBuffer()
    t0 = time.time()
    status1 = status2 = "not_run"

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    # Binary pipes + manual utf-8/replace decode (Windows-safe).
    proc = subprocess.Popen(
        [py, str(run_py), "--no-gui"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        env=child_env,
    )
    assert proc.stdout is not None and proc.stderr is not None
    threads = [
        threading.Thread(
            target=_reader_bytes,
            args=(proc.stdout, buf, sys.stdout, "out"),
            daemon=True,
        ),
        threading.Thread(
            target=_reader_bytes,
            args=(proc.stderr, buf, sys.stderr, "err"),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    try:
        if not _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S):
            status1 = "boot_failed"
            print("[qa] BOOT FAILED — skipping injects", flush=True)
        else:
            print("[qa] boot OK — injecting Task 1 (Meta-Broker TDD)", flush=True)
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
                label="task1",
            )
            print(f"[qa] task1 inject status={status1}", flush=True)

            # Always attempt Task 2 if the engine is still alive.
            if proc.poll() is None and time.time() - t0 < TOTAL_DEADLINE_S:
                print("[qa] injecting Task 2 (Tkinter popup)", flush=True)
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
                    label="task2",
                )
                print(f"[qa] task2 inject status={status2}", flush=True)
            else:
                status2 = "skipped_engine_dead"
    finally:
        if proc.poll() is None:
            print("[qa] terminating headless agent...", flush=True)
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
        _restore_hybrid_planner(previous_hybrid)

    engine_exit = _normalize_exit(proc.returncode)
    print("[qa] running quality audits…", flush=True)
    audit1 = audit_task1_rate_limiter()
    audit2 = audit_task2_popup_animation()
    print(
        f"[qa] 7.1={audit1['verdict']} ({audit1['detail']})",
        flush=True,
    )
    print(
        f"[qa] 7.2={audit2['verdict']} ({audit2['detail']})",
        flush=True,
    )

    path = append_suite7_psol(
        audit1=audit1,
        audit2=audit2,
        status1=status1,
        status2=status2,
        engine_exit=engine_exit,
        stream_tail=buf.snapshot(tail=8000),
    )
    print(f"[qa] PSOL appended → {path}", flush=True)

    if audit1["verdict"] == "PASS" and audit2["verdict"] == "PASS":
        print("[qa] OVERALL PASS", flush=True)
        return 0
    print("[qa] OVERALL FAIL", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
