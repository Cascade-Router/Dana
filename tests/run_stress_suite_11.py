"""Diagnostic Suite 11 — Local Qwen Tkinter & Algorithmic TDD.

Headless Meta-Broker stress with Hybrid OFF so Planner + Worker stay on the
local Qwen2.5-Coder (or configured DONNA_OLLAMA_MODEL) stack.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_suite_11.py
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
SUITE_MARKER = "### Diagnostic Suite 11: Local Qwen Tkinter & Algorithmic TDD"

PROMPT = (
    "/broker Epic 1: Write a Python Tkinter script in popup_animation.py that "
    "displays a bouncing ball animation on a Canvas. "
    "Epic 2: Write maze_solver.py with a BFS pathfinding algorithm. "
    "Epic 3: Write tests/test_maze.py to verify the BFS logic against a known grid."
)

ARTIFACTS = (
    "popup_animation.py",
    "maze_solver.py",
    "tests/test_maze.py",
)

# Prefer a small-enough coder tag; override with DONNA_OLLAMA_MODEL if set.
DEFAULT_QWEN_MODEL = "qwen2.5-coder:7b"

BOOT_DEADLINE_S = 120.0
PROMPT_BUDGET_S = 1500.0
QUIET_AFTER_COMPLETE_S = 10.0

_COMPLETION_RE = re.compile(
    r"("
    r"All\s+\d+\s+epic\(s\)\s+completed"
    r"|epic\s+\d+\s+exhausted\s+repairs"
    r"|graph\.invoke\s+END"
    r"|MetaBroker\].*completed"
    r"|status=['\"]completed['\"]"
    r"|broker_phase['\"]?\s*[:=]\s*['\"]?done"
    r")",
    re.I,
)
_SETTLED_RE = re.compile(
    r"("
    r"graph\.invoke\s+END"
    r"|epic\s+\d+\s+exhausted\s+repairs"
    r"|All\s+\d+\s+epic\(s\)\s+completed"
    r")",
    re.I,
)
_HARNESS_EXIT_RE = re.compile(
    r"\[RuntimeHarness\]\s+END\s+success=(\w+)\s+exit=(-?\d+)",
    re.I,
)
_RAM_BREAKER_RE = re.compile(
    r"CRITICAL:\s*RAM usage exceeds safe limits",
    re.I,
)


class StreamBuffer:
    def __init__(self, *, maxlen: int = 600_000) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self.text = ""
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

    def snapshot(self, *, tail: int = 8000) -> str:
        with self._lock:
            return self.text[-tail:]

    def chars(self) -> int:
        with self._lock:
            return self.total_chars


class RamPeakMonitor:
    """Background sampler for peak process/system RAM during the Suite run."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.peak_system_pct = 0.0
        self.peak_process_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "peak_system_ram_pct": round(self.peak_system_pct, 2),
            "peak_process_rss_mb": round(self.peak_process_mb, 2),
            "samples": self.samples,
        }

    def _run(self) -> None:
        try:
            import psutil
        except Exception:  # noqa: BLE001
            return
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                sys_pct = float(psutil.virtual_memory().percent)
                rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                # Also sample children of the headless agent if we know the pid.
                self.peak_system_pct = max(self.peak_system_pct, sys_pct)
                self.peak_process_mb = max(self.peak_process_mb, rss_mb)
                self.samples += 1
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)


def _reader_bytes(pipe, buf: StreamBuffer, mirror, label: str) -> None:
    try:
        pending = ""
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            pending += text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line += "\n"
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


def _force_local_planner() -> bool:
    """Force hybrid_planner_enabled=False for a pure-local Qwen run."""
    from dana.graph.cloud_planner import planner_mode_label
    from dana.settings import is_hybrid_planner_enabled, set_hybrid_planner_enabled

    previous = bool(is_hybrid_planner_enabled())
    set_hybrid_planner_enabled(False)
    try:
        import json as _json

        from dana.paths import SETTINGS_PATH

        payload: dict[str, Any] = {}
        if SETTINGS_PATH.is_file():
            payload = _json.loads(
                SETTINGS_PATH.read_text(encoding="utf-8", errors="replace")
            )
        if not isinstance(payload, dict):
            payload = {}
        payload["hybrid_planner_enabled"] = False
        SETTINGS_PATH.write_text(
            _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: settings mirror failed ({exc})", flush=True)
    print(
        f"[qa11] hybrid_planner_enabled=False (was {previous}); "
        f"planner_mode=[{planner_mode_label()}]",
        flush=True,
    )
    return previous


def _restore_hybrid(previous: bool | None) -> None:
    if previous is None:
        return
    try:
        from dana.settings import set_hybrid_planner_enabled

        set_hybrid_planner_enabled(bool(previous))
        print(f"[qa11] restored hybrid_planner_enabled={bool(previous)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: hybrid restore failed ({exc})", flush=True)


def _resolve_qwen_model() -> str:
    wanted = (os.environ.get("DONNA_OLLAMA_MODEL") or DEFAULT_QWEN_MODEL).strip()
    try:
        import requests

        tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
        names = [str(m.get("name") or "") for m in (tags.get("models") or [])]
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: cannot list Ollama models ({exc})", flush=True)
        return wanted

    def _present(name: str) -> bool:
        base = name.split(":")[0].lower()
        for n in names:
            nl = n.lower()
            if nl == name.lower() or nl.startswith(base + ":"):
                return True
        return False

    if _present(wanted):
        print(f"[qa11] local model ready: {wanted}", flush=True)
        return wanted

    # Prefer smaller coder tags if the preferred one is missing.
    candidates = [
        wanted,
        "qwen2.5-coder:7b",
        "qwen2.5-coder:3b",
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:latest",
    ]
    for cand in candidates:
        if _present(cand):
            print(f"[qa11] using installed model {cand!r} (wanted {wanted!r})", flush=True)
            return cand

    print(f"[qa11] pulling {wanted!r} via ollama…", flush=True)
    try:
        pull_env = os.environ.copy()
        pull_env["PYTHONIOENCODING"] = "utf-8"
        pull_env["PYTHONUTF8"] = "1"
        proc = subprocess.run(
            ["ollama", "pull", wanted],
            cwd=str(ROOT),
            capture_output=True,
            timeout=1800,
            check=False,
            env=pull_env,
        )
        # Re-list after pull (stdout may contain progress glyphs that break cp1252).
        try:
            tags2 = requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
            names = [str(m.get("name") or "") for m in (tags2.get("models") or [])]
        except Exception:  # noqa: BLE001
            pass
        if proc.returncode == 0 and _present(wanted):
            print(f"[qa11] pull OK: {wanted}", flush=True)
            return wanted
        # Pull may have succeeded despite a non-zero / decode quirk.
        if _present(wanted):
            print(f"[qa11] model present after pull attempt: {wanted}", flush=True)
            return wanted
        print(
            f"[qa11] WARNING: pull failed rc={proc.returncode}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: pull raised ({exc})", flush=True)
        try:
            tags2 = requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
            names = [str(m.get("name") or "") for m in (tags2.get("models") or [])]
            if _present(wanted):
                print(f"[qa11] model present after pull error: {wanted}", flush=True)
                return wanted
        except Exception:  # noqa: BLE001
            pass

    # Prefer any installed qwen2.5-coder over unrelated drafts.
    for n in names:
        if "qwen2.5-coder" in n.lower():
            print(f"[qa11] falling back to installed coder {n!r}", flush=True)
            return n
    fallback = names[0] if names else wanted
    print(f"[qa11] falling back to installed model {fallback!r}", flush=True)
    return fallback


def _write_trigger(prompt: str) -> None:
    TRIGGER.write_text(prompt.strip() + "\n", encoding="utf-8")


def _wait_boot(proc: subprocess.Popen, buf: StreamBuffer, deadline: float) -> bool:
    needle = re.compile(r"Dana is ready|File trigger|Headless mode", re.I)
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if needle.search(buf.snapshot(tail=20_000)):
            return True
        time.sleep(0.5)
    return False


def _artifacts_ready() -> bool:
    return all((ROOT / rel).is_file() for rel in ARTIFACTS)


def _wait_prompt_resolved(
    proc: subprocess.Popen,
    buf: StreamBuffer,
    *,
    budget_s: float,
    prior_chars: int,
    label: str,
    agent_pid: int | None = None,
    ram_mon: RamPeakMonitor | None = None,
) -> str:
    deadline = time.time() + budget_s
    quiet_since: float | None = None
    last_chars = prior_chars
    busy_re = re.compile(
        r"\[MetaBroker\]|\[DagSupervisor\]|\[DagWorker\]|\[RuntimeHarness\]|"
        r"\[CloudPlanner\]|Loading weights|ChatOllama",
        re.I,
    )
    try:
        import psutil
    except Exception:  # noqa: BLE001
        psutil = None  # type: ignore[assignment]

    while time.time() < deadline:
        if proc.poll() is not None:
            return "proc_exited"
        # Track peak system + agent RSS if possible.
        if ram_mon is not None and psutil is not None:
            try:
                ram_mon.peak_system_pct = max(
                    ram_mon.peak_system_pct,
                    float(psutil.virtual_memory().percent),
                )
                if agent_pid:
                    try:
                        rss = float(psutil.Process(agent_pid).memory_info().rss)
                        ram_mon.peak_process_mb = max(
                            ram_mon.peak_process_mb, rss / (1024.0 * 1024.0)
                        )
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

        snap = buf.snapshot(tail=50_000)
        chars = buf.chars()
        settled = bool(_SETTLED_RE.search(snap))
        artifacts = _artifacts_ready()
        busy = bool(busy_re.search(snap[-4000:]))
        if artifacts and settled:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                print(
                    f"[qa11] {label}: artifacts+broker settled — resolving",
                    flush=True,
                )
                return "artifacts_ready"
        elif settled and _COMPLETION_RE.search(snap):
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "complete_sig"
        elif artifacts and not busy and chars == last_chars:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= 60.0:
                print(
                    f"[qa11] {label}: artifacts ready + quiet stall — resolving",
                    flush=True,
                )
                return "artifacts_stalled"
        else:
            quiet_since = None
        if chars != last_chars:
            last_chars = chars
            if busy:
                quiet_since = None
        left = int(deadline - time.time())
        if left % 30 < 2:
            print(
                f"[qa11] {label}: waiting… {left}s left "
                f"settled={settled} artifacts={artifacts}",
                flush=True,
            )
        time.sleep(1.0)
    if _artifacts_ready():
        return "timeout_with_artifacts"
    if _COMPLETION_RE.search(buf.snapshot(tail=40_000)):
        return "timeout_with_signature"
    return "timeout"


def audit_suite11() -> dict[str, Any]:
    popup = ROOT / "popup_animation.py"
    maze = ROOT / "maze_solver.py"
    test_path = ROOT / "tests" / "test_maze.py"
    result: dict[str, Any] = {
        "popup_exists": popup.is_file(),
        "maze_exists": maze.is_file(),
        "test_exists": test_path.is_file(),
        "paths": {
            "popup_animation.py": str(popup),
            "maze_solver.py": str(maze),
            "tests/test_maze.py": str(test_path),
        },
        "file_verdicts": {},
        "tkinter_compile_ok": False,
        "tkinter_import_ok": False,
        "tkinter_canvas_ok": False,
        "tkinter_verdict": "FAIL",
        "pytest_exit_code": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    result["file_verdicts"] = {
        "popup_animation.py": "PASS" if result["popup_exists"] else "FAIL",
        "maze_solver.py": "PASS" if result["maze_exists"] else "FAIL",
        "tests/test_maze.py": "PASS" if result["test_exists"] else "FAIL",
    }
    missing = [k for k, v in result["file_verdicts"].items() if v == "FAIL"]
    if missing:
        result["detail"] = f"missing files: {', '.join(missing)}"
        return result

    # GUI validation: compile + token presence.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)]
    )
    try:
        compile_proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(popup)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
            check=False,
        )
        result["tkinter_compile_ok"] = compile_proc.returncode == 0
        if compile_proc.returncode != 0:
            result["detail"] = (
                f"py_compile failed rc={compile_proc.returncode}; "
                f"stderr={(compile_proc.stderr or '')[:300]!r}"
            )
    except subprocess.TimeoutExpired:
        result["detail"] = "TIMEOUT: py_compile popup_animation.py"
        return result

    try:
        src = popup.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["detail"] = f"cannot read popup_animation.py: {exc}"
        return result
    result["tkinter_import_ok"] = bool(
        re.search(r"\bimport\s+tkinter\b|\bfrom\s+tkinter\s+import\b", src)
    )
    result["tkinter_canvas_ok"] = "Canvas" in src
    if (
        result["tkinter_compile_ok"]
        and result["tkinter_import_ok"]
        and result["tkinter_canvas_ok"]
    ):
        result["tkinter_verdict"] = "PASS"
    else:
        result["tkinter_verdict"] = "FAIL"
        result["detail"] = (
            f"tkinter checks compile={result['tkinter_compile_ok']} "
            f"import={result['tkinter_import_ok']} canvas={result['tkinter_canvas_ok']}"
        )

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result["pytest_exit_code"] = -1
        result["detail"] = "TIMEOUT: pytest exceeded 120s"
        return result
    result["pytest_exit_code"] = int(proc.returncode)
    result["pytest_stdout"] = (proc.stdout or "")[-2000:]
    result["pytest_stderr"] = (proc.stderr or "")[-2000:]
    if (
        result["tkinter_verdict"] == "PASS"
        and proc.returncode == 0
        and not missing
    ):
        result["verdict"] = "PASS"
        result["detail"] = "tkinter PASS + pytest exit_code=0"
    elif proc.returncode != 0:
        result["detail"] = (
            f"pytest exit_code={proc.returncode}; "
            f"stderr={(proc.stderr or '')[:400]!r}"
        )
    return result


def _tail_runtime_log(n: int = 80) -> str:
    if not RUNTIME_LOG.is_file():
        return ""
    try:
        lines = RUNTIME_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def append_psol(
    *,
    audit: dict[str, Any],
    inject_status: str,
    stream_tail: str,
    ram_stats: dict[str, Any],
    model_id: str,
    ram_breaker_seen: bool,
) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v = audit.get("verdict") or "FAIL"
    fv = audit.get("file_verdicts") or {}
    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Generated: {now}_",
            f"_Overall: `{v}`_",
            "",
            "Headless **local-only** Meta-Broker stress (Hybrid OFF) for Tkinter "
            "animation + BFS maze TDD on Qwen2.5-Coder / configured Ollama model.",
            "",
            "| Check | Verdict | Detail |",
            "|---|---|---|",
            (
                f"| `11.1` popup_animation.py | "
                f"`{fv.get('popup_animation.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('popup_animation.py')}` |"
            ),
            (
                f"| `11.2` maze_solver.py | "
                f"`{fv.get('maze_solver.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('maze_solver.py')}` |"
            ),
            (
                f"| `11.3` tests/test_maze.py | "
                f"`{fv.get('tests/test_maze.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('tests/test_maze.py')}` |"
            ),
            (
                f"| `11.4` Tkinter validation | "
                f"`{audit.get('tkinter_verdict')}` | "
                f"compile={audit.get('tkinter_compile_ok')}; "
                f"import_tkinter={audit.get('tkinter_import_ok')}; "
                f"Canvas={audit.get('tkinter_canvas_ok')} |"
            ),
            (
                f"| `11.5` pytest maze | "
                f"`{'PASS' if audit.get('pytest_exit_code') == 0 else 'FAIL'}` | "
                f"exit_code=`{audit.get('pytest_exit_code')}`; {audit.get('detail')} |"
            ),
            (
                f"| `11.6` Peak RAM | "
                f"`{'OK' if not ram_breaker_seen else 'BREAKER_SEEN'}` | "
                f"peak_system=`{ram_stats.get('peak_system_ram_pct')}%`; "
                f"peak_suite_rss=`{ram_stats.get('peak_process_rss_mb')}MB`; "
                f"agent_peak_rss=`{ram_stats.get('peak_agent_rss_mb')}MB` |"
            ),
            "",
            "#### Prompt",
            "",
            f"`{PROMPT}`",
            "",
            f"- **Inject status:** `{inject_status}`",
            f"- **hybrid_planner_enabled:** `False` (forced local)",
            f"- **OLLAMA / Qwen model:** `{model_id}`",
            f"- **Tkinter verdict:** `{audit.get('tkinter_verdict')}`",
            f"- **pytest exit_code (test_maze):** `{audit.get('pytest_exit_code')}`",
            f"- **Peak system RAM %:** `{ram_stats.get('peak_system_ram_pct')}`",
            f"- **OOM breaker tripped during run:** "
            f"`{'YES' if ram_breaker_seen else 'NO'}`",
            "",
            "```text",
            (audit.get("pytest_stdout") or "(no pytest stdout)")[-1500:],
            "```",
            "",
            "### Runtime log tail",
            "",
            "```text",
            _tail_runtime_log(60) or "(unavailable)",
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
    PSOL_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def _kill_stale_lock() -> None:
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            if s.connect_ex(("127.0.0.1", 47473)) != 0:
                return
        finally:
            s.close()
        cmd = (
            "Get-NetTCPConnection -LocalPort 47473 -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty OwningProcess -Unique | "
            "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            cwd=str(ROOT),
            capture_output=True,
            timeout=15,
            check=False,
        )
        time.sleep(1.0)
        print("[qa11] cleared stale lock on 47473 (if any)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: lock clear failed ({exc})", flush=True)


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print(
        "=== Dana Diagnostic Suite 11 — Local Qwen Tkinter & Maze TDD ===",
        flush=True,
    )
    print(f"python={py}", flush=True)
    print(f"launch={run_py} --no-gui", flush=True)

    _kill_stale_lock()

    for rel in ARTIFACTS:
        p = ROOT / rel
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass

    model_id = _resolve_qwen_model()
    previous_hybrid: bool | None = None
    try:
        previous_hybrid = _force_local_planner()
    except Exception as exc:  # noqa: BLE001
        print(f"[qa11] WARNING: hybrid disable failed ({exc})", flush=True)

    ram_mon = RamPeakMonitor(interval_s=1.0)
    ram_mon.start()
    try:
        import psutil

        ram_mon.peak_system_pct = float(psutil.virtual_memory().percent)
    except Exception:  # noqa: BLE001
        pass

    buf = StreamBuffer()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_env["DONNA_HEADLESS"] = "1"
    child_env["DONNA_DISABLE_TTS"] = "1"
    child_env["DONNA_DISABLE_MIC"] = "1"
    child_env["DONNA_DISABLE_IDLE_MONITOR"] = "1"
    child_env["DONNA_DISABLE_TOAST"] = "1"
    child_env["DONNA_OLLAMA_MODEL"] = model_id
    child_env["OLLAMA_MODEL"] = model_id
    # Host is often already >85%; suite still needs to exercise local rails.
    child_env["DONNA_SKIP_RAM_BREAKER"] = "1"

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

    inject_status = "boot_failed"
    peak_agent_mb = 0.0
    try:
        if not _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S):
            print("[qa11] BOOT FAILED — skipping inject", flush=True)
        else:
            print(
                "[qa11] boot OK — injecting Tkinter+Maze Meta-Broker prompt",
                flush=True,
            )
            prior_chars = buf.chars()
            _write_trigger(PROMPT)
            inject_status = _wait_prompt_resolved(
                proc,
                buf,
                budget_s=PROMPT_BUDGET_S,
                prior_chars=prior_chars,
                label="suite11",
                agent_pid=int(proc.pid or 0) or None,
                ram_mon=ram_mon,
            )
            print(f"[qa11] inject status={inject_status}", flush=True)
            try:
                import psutil

                if proc.pid:
                    peak_agent_mb = float(
                        psutil.Process(int(proc.pid)).memory_info().rss
                    ) / (1024.0 * 1024.0)
                    ram_mon.peak_process_mb = max(
                        ram_mon.peak_process_mb, peak_agent_mb
                    )
            except Exception:  # noqa: BLE001
                pass
    finally:
        print("[qa11] terminating headless agent...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _restore_hybrid(previous_hybrid)
        ram_stats = ram_mon.stop()
        ram_stats["peak_agent_rss_mb"] = round(
            max(peak_agent_mb, float(ram_stats.get("peak_process_rss_mb") or 0)),
            2,
        )

    print("[qa11] running quality audits…", flush=True)
    audit = audit_suite11()
    blob = f"{buf.snapshot(tail=120_000)}\n{_tail_runtime_log(200)}"
    ram_breaker_seen = bool(_RAM_BREAKER_RE.search(blob))
    print(
        f"[qa11] files={audit.get('file_verdicts')} "
        f"tkinter={audit.get('tkinter_verdict')} "
        f"pytest={audit.get('pytest_exit_code')} verdict={audit.get('verdict')}",
        flush=True,
    )
    print(
        f"[qa11] peak_ram_system={ram_stats.get('peak_system_ram_pct')}% "
        f"breaker_seen={ram_breaker_seen} model={model_id}",
        flush=True,
    )
    psol = append_psol(
        audit=audit,
        inject_status=inject_status,
        stream_tail=buf.snapshot(tail=6000),
        ram_stats=ram_stats,
        model_id=model_id,
        ram_breaker_seen=ram_breaker_seen,
    )
    print(f"[qa11] PSOL appended -> {psol}", flush=True)
    overall = audit.get("verdict") == "PASS"
    print(f"[qa11] OVERALL {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
