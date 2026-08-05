"""Animation & BFS Meta-Broker diagnostic suite.

Launches ``run.py --no-gui``, injects the Animation/BFS/Tests broker prompt via
``.trigger_ask``, captures telemetry, then audits artifacts.

Usage::

    .venv\\Scripts\\python.exe tests/run_animation_bfs_suite.py
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
SUITE_LOG = ROOT / "logs" / "animation_bfs_suite.log"
RUNTIME_LOG = ROOT / "logs" / "dana_runtime.log"
PSOL_PATH = ROOT / "logs" / "context_diagnostic_psol.md"
SUITE_MARKER = "### Diagnostic Suite: Animation & BFS"

BROKER_PROMPT = (
    "/broker Epic 1: Write a Python Tkinter script in popup_animation.py that "
    "displays a bouncing ball animation on a Canvas. Epic 2: Write maze_solver.py "
    "with a BFS pathfinding algorithm. Epic 3: Write tests/test_maze.py to verify "
    "the BFS logic against a known grid."
)

BOOT_DEADLINE_S = 150.0
PROMPT_BUDGET_S = 1200.0
QUIET_AFTER_COMPLETE_S = 10.0
TOTAL_DEADLINE_S = 1400.0

HEAP_SIG = "0xC0000374"
TRACEBACK_SIG = "Traceback (most recent call last):"

_COMPLETION_RE = re.compile(
    r"("
    r"All\s+\d+\s+epic\(s\)\s+completed"
    r"|epic\s+\d+\s+validated\s+OK"
    r"|MetaBroker\].*completed"
    r"|Meta-Broker finished status=completed"
    r"|OK:\s*meta_broker status=completed"
    r"|graph\.invoke END status='completed'"
    r")",
    re.I,
)
_FAIL_RE = re.compile(
    r"("
    r"status=['\"]failed['\"]"
    r"|MetaBroker\].*failed"
    r"|graph\.invoke END status='failed'"
    r"|OK:\s*meta_broker status=failed"
    r"|ERROR:\s*meta_broker"
    r"|CRITICAL:\s*RAM usage"
    r")",
    re.I,
)
_BUSY_RE = re.compile(
    r"("
    r"\[MetaBroker\]"
    r"|\[Cascade\]"
    r"|\[Agentic\]"
    r"|\[DagSupervisor\]"
    r"|\[DagWorker\]"
    r"|qwen2\.5-coder"
    r"|file_editor|python_repl|write_to_file"
    r"|runtime harness"
    r")",
    re.I,
)
_MODEL_RE = re.compile(r"qwen2\.5-coder:7b", re.I)
_RAM_RE = re.compile(
    r"(?:ram[=:\s]+|RAM usage[=:\s]+|health ram=)(\d+(?:\.\d+)?)\s*%?",
    re.I,
)


class StreamBuffer:
    def __init__(self, *, maxlen: int = 600_000) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._maxlen = maxlen
        self.text = ""
        self.saw_heap = False
        self.saw_traceback = False
        self.total_chars = 0
        self.ram_samples: list[float] = []
        self.model_hits = 0

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
            if TRACEBACK_SIG in data and "_readerthread" not in data:
                self.saw_traceback = True
            self.model_hits += len(_MODEL_RE.findall(data))
            for m in _RAM_RE.finditer(data):
                try:
                    self.ram_samples.append(float(m.group(1)))
                except ValueError:
                    pass

    def snapshot(self, *, tail: int = 8000) -> str:
        with self._lock:
            return self.text[-tail:]

    def chars(self) -> int:
        with self._lock:
            return self.total_chars

    def full(self) -> str:
        with self._lock:
            return self.text


def _reader_bytes(pipe, buf: StreamBuffer, mirror, label: str) -> None:
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


def _write_trigger(prompt: str) -> None:
    TRIGGER.write_text(prompt.strip() + "\n", encoding="utf-8")


def _clear_trigger() -> None:
    try:
        if TRIGGER.is_file():
            TRIGGER.unlink()
    except OSError:
        pass


def _find_file(name: str) -> Path | None:
    preferred = ROOT / name
    if preferred.is_file():
        return preferred
    if name.startswith("tests/"):
        p = ROOT / name
        if p.is_file():
            return p
    hits = [
        p
        for p in ROOT.rglob(Path(name).name)
        if ".venv" not in p.parts
        and "site-packages" not in p.parts
        and ".git" not in p.parts
    ]
    # Prefer project root / tests over shadow scratch.
    hits.sort(
        key=lambda p: (
            0 if p.parent == ROOT else 1,
            0 if "tests" in p.parts else 1,
            0 if "shadow" not in p.parts else 1,
            len(p.parts),
        )
    )
    return hits[0] if hits else None


def artifacts_ready() -> bool:
    popup = _find_file("popup_animation.py")
    maze = _find_file("maze_solver.py")
    test = _find_file("tests/test_maze.py") or _find_file("test_maze.py")
    if not (popup and maze and test):
        return False
    for p in (popup, maze, test):
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if len(body.strip()) < 40:
            return False
    return True


def _wait_boot(proc: subprocess.Popen, buf: StreamBuffer, deadline: float) -> bool:
    needles = (
        "Headless mode: engine auto-ENGAGED",
        "Headless mode (--no-gui)",
        "=== CAMGRASPER Donna voice agent ===",
        "Noise floor calibrated",
        "Dana is ready",
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        snap = buf.snapshot(tail=30_000)
        if any(n in snap for n in needles):
            time.sleep(2.0)
            return True
        if buf.saw_heap:
            return False
        time.sleep(0.25)
    return False


def _wait_prompt_resolved(
    proc: subprocess.Popen,
    buf: StreamBuffer,
    *,
    budget_s: float,
    prior_chars: int,
) -> str:
    deadline = time.time() + budget_s
    saw_complete = False
    saw_fail = False
    quiet_since: float | None = None
    last_chars = prior_chars
    last_progress = 0.0

    while time.time() < deadline:
        if proc.poll() is not None:
            if artifacts_ready():
                return "artifacts_ready"
            return "process_exited"
        if buf.saw_heap:
            return "heap_corruption"

        snap = buf.full()
        chars = buf.chars()
        growing = chars > last_chars
        if growing:
            last_chars = chars
            quiet_since = None

        if artifacts_ready() and (_COMPLETION_RE.search(snap) or saw_complete):
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "artifacts_ready"

        if _COMPLETION_RE.search(snap):
            saw_complete = True
            if artifacts_ready():
                if quiet_since is None and not growing:
                    quiet_since = time.time()
                elif quiet_since is not None and not growing:
                    if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                        return "completion_signature"
            elif quiet_since is None and not growing:
                quiet_since = time.time()
            elif quiet_since is not None and not growing:
                if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                    return "completion_signature"

        if _FAIL_RE.search(snap[-8000:]):
            saw_fail = True
            # Hard abort signals (RAM breaker) — do not wait the full budget.
            if re.search(r"CRITICAL:\s*RAM usage|could not derive epics", snap, re.I):
                if not artifacts_ready():
                    return "broker_failed"
            if quiet_since is None and not growing:
                quiet_since = time.time()
            elif quiet_since is not None and not growing:
                if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                    return "broker_failed"

        if _BUSY_RE.search(snap[-3000:]):
            quiet_since = None

        now = time.time()
        if now - last_progress >= 30.0:
            try:
                import psutil

                buf.ram_samples.append(float(psutil.virtual_memory().percent))
            except Exception:
                pass
            remaining = max(0.0, deadline - now)
            print(
                f"[suite] waiting… {remaining:.0f}s left "
                f"complete={saw_complete} fail={saw_fail} "
                f"artifacts={artifacts_ready()} "
                f"model_hits={buf.model_hits} "
                f"ram_max={max(buf.ram_samples) if buf.ram_samples else 'n/a'}",
                flush=True,
            )
            last_progress = now
        time.sleep(0.5)

    if artifacts_ready():
        return "timeout_with_artifacts"
    if saw_complete:
        return "timeout_with_signature"
    if saw_fail:
        return "timeout_with_fail"
    return "timeout"


def audit_popup() -> dict[str, Any]:
    path = _find_file("popup_animation.py")
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": bool(path and path.is_file()),
        "compile_ok": False,
        "has_tkinter": False,
        "has_canvas": False,
        "run_status": "not_run",
        "run_stdout": "",
        "run_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    if not path:
        result["detail"] = "missing popup_animation.py"
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
    result["compile_ok"] = compile_proc.returncode == 0
    if compile_proc.returncode != 0:
        result["detail"] = (compile_proc.stderr or "")[:500]
        return result
    src = path.read_text(encoding="utf-8", errors="replace")
    result["has_tkinter"] = bool(
        re.search(r"(?m)^\s*(import\s+tkinter|from\s+tkinter\s+import)", src)
    )
    result["has_canvas"] = "Canvas" in src or "canvas" in src.lower()

    # Execute briefly — mainloop scripts should be killed after a short window.
    env = os.environ.copy()
    env["TK_SILENCE_DEPRECATION"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4.0,
            env=env,
            check=False,
        )
        result["run_status"] = f"exited:{proc.returncode}"
        result["run_stdout"] = (proc.stdout or "")[-800:]
        result["run_stderr"] = (proc.stderr or "")[-800:]
        if proc.returncode != 0:
            result["detail"] = f"script exited {proc.returncode}: {result['run_stderr'][:300]}"
            return result
    except subprocess.TimeoutExpired as exc:
        # Expected for animation mainloop — process was running without immediate crash.
        result["run_status"] = "timeout_running_ok"
        result["run_stdout"] = ((exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))  # type: ignore[assignment]
        if isinstance(result["run_stdout"], bytes):
            result["run_stdout"] = result["run_stdout"].decode("utf-8", errors="replace")[-800:]
        err = exc.stderr or b""
        if isinstance(err, bytes):
            result["run_stderr"] = err.decode("utf-8", errors="replace")[-800:]
        else:
            result["run_stderr"] = str(err)[-800:]
    except Exception as exc:  # noqa: BLE001
        result["run_status"] = f"error:{type(exc).__name__}"
        result["detail"] = str(exc)
        return result

    if result["has_tkinter"] and result["has_canvas"] and result["compile_ok"]:
        if result["run_status"] in {"timeout_running_ok", "exited:0"}:
            result["verdict"] = "PASS"
            result["detail"] = f"compile+content ok; run={result['run_status']}"
        else:
            result["detail"] = f"run_status={result['run_status']}"
    else:
        gaps = []
        if not result["has_tkinter"]:
            gaps.append("no tkinter import")
        if not result["has_canvas"]:
            gaps.append("no Canvas")
        result["detail"] = "; ".join(gaps) or "content gate fail"
    return result


def audit_maze() -> dict[str, Any]:
    maze = _find_file("maze_solver.py")
    test = _find_file("tests/test_maze.py") or _find_file("test_maze.py")
    result: dict[str, Any] = {
        "maze_path": str(maze) if maze else None,
        "test_path": str(test) if test else None,
        "maze_exists": bool(maze),
        "test_exists": bool(test),
        "pytest_exit": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    if not maze or not test:
        missing = []
        if not maze:
            missing.append("maze_solver.py")
        if not test:
            missing.append("tests/test_maze.py")
        result["detail"] = "missing: " + ", ".join(missing)
        return result

    env = os.environ.copy()
    paths = [str(ROOT), str(maze.parent)]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([prev] if prev else []))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test), "-q", "--tb=short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result["pytest_exit"] = -1
        result["detail"] = "pytest timeout"
        return result
    result["pytest_exit"] = int(proc.returncode)
    result["pytest_stdout"] = (proc.stdout or "")[-2500:]
    result["pytest_stderr"] = (proc.stderr or "")[-2500:]
    if proc.returncode == 0:
        result["verdict"] = "PASS"
        result["detail"] = "pytest exit_code=0"
    else:
        result["detail"] = (
            f"pytest exit_code={proc.returncode}; "
            f"{(proc.stderr or proc.stdout or '')[:600]}"
        )
    return result


def extract_epic_status(log_text: str) -> dict[str, str]:
    status = {"epic1": "unknown", "epic2": "unknown", "epic3": "unknown", "overall": "unknown"}
    if re.search(r"All\s+3\s+epic\(s\)\s+completed", log_text, re.I):
        status.update(
            epic1="completed",
            epic2="completed",
            epic3="completed",
            overall="completed",
        )
        return status
    for i in (1, 2, 3):
        if re.search(rf"epic\s+{i}\s+validated\s+OK", log_text, re.I):
            status[f"epic{i}"] = "completed"
        elif re.search(rf"epic\s+{i}\s+failed", log_text, re.I):
            status[f"epic{i}"] = "failed"
        elif re.search(rf"dispatch epic\s+{i}", log_text, re.I):
            if status[f"epic{i}"] == "unknown":
                status[f"epic{i}"] = "started"
    if re.search(r"status=['\"]failed['\"]|broker_failed|exhausted repairs", log_text, re.I):
        status["overall"] = "failed"
    elif re.search(r"status=['\"]completed['\"]|All\s+\d+\s+epic", log_text, re.I):
        status["overall"] = "completed"
    return status


def append_psol(report: dict[str, Any]) -> None:
    PSOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "",
        SUITE_MARKER,
        "",
        f"_Updated: {now}_",
        f"_Inject status: `{report.get('inject_status')}`_",
        f"_Engine exit: `{report.get('engine_exit')}`_",
        "",
        f"- Epic1: `{report['epics'].get('epic1')}`",
        f"- Epic2: `{report['epics'].get('epic2')}`",
        f"- Epic3: `{report['epics'].get('epic3')}`",
        f"- Overall: `{report['epics'].get('overall')}`",
        f"- Popup verdict: `{report['popup'].get('verdict')}` ({report['popup'].get('detail')})",
        f"- Maze/pytest verdict: `{report['maze'].get('verdict')}` ({report['maze'].get('detail')})",
        f"- Model hits (qwen2.5-coder:7b): `{report.get('model_hits')}`",
        f"- RAM samples max%: `{report.get('ram_max')}`",
        "",
        "```text",
        (report.get("stream_tail") or "")[-4000:],
        "```",
        "",
    ]
    with PSOL_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    SUITE_LOG.parent.mkdir(parents=True, exist_ok=True)
    _clear_trigger()

    env = os.environ.copy()
    env.setdefault("DONNA_OLLAMA_MODEL", "qwen2.5-coder:7b")
    env.setdefault("OLLAMA_MODEL", "qwen2.5-coder:7b")
    env.setdefault("DONNA_LOCAL_MODEL", "qwen2.5-coder:7b")
    env.setdefault("DONNA_SKIP_BOOT_READY", "1")
    # Host often sits near/above 92% with Chrome; skip abort so epics can run,
    # while suite still samples RAM from MetaBroker health lines + psutil.
    env["DONNA_SKIP_RAM_BREAKER"] = "1"
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    mirror = SUITE_LOG.open("w", encoding="utf-8", errors="replace")
    mirror.write(f"# Animation & BFS suite start {datetime.now().isoformat()}\n")
    mirror.write(f"# prompt={BROKER_PROMPT}\n")
    mirror.flush()

    cmd = [sys.executable, str(ROOT / "run.py"), "--no-gui"]
    print(f"[suite] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    buf = StreamBuffer()
    t_out = threading.Thread(
        target=_reader_bytes, args=(proc.stdout, buf, mirror, "OUT"), daemon=True
    )
    t_err = threading.Thread(
        target=_reader_bytes, args=(proc.stderr, buf, mirror, "ERR"), daemon=True
    )
    t_out.start()
    t_err.start()

    boot_ok = _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S)
    print(f"[suite] boot_ok={boot_ok}", flush=True)
    if not boot_ok:
        try:
            proc.terminate()
        except Exception:
            pass
        mirror.write("\n# BOOT FAILED\n")
        mirror.write(buf.snapshot(tail=12_000))
        mirror.close()
        print("[suite] BOOT FAILED", flush=True)
        print(buf.snapshot(tail=4000), flush=True)
        return 2

    prior_chars = buf.chars()
    print("[suite] injecting broker prompt via .trigger_ask", flush=True)
    _write_trigger(BROKER_PROMPT)

    status = _wait_prompt_resolved(
        proc, buf, budget_s=PROMPT_BUDGET_S, prior_chars=prior_chars
    )
    print(f"[suite] inject_status={status}", flush=True)

    # Stop engine.
    try:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    except Exception:
        pass
    t_out.join(timeout=3)
    t_err.join(timeout=3)
    _clear_trigger()

    epics = extract_epic_status(buf.full())
    # Infer from artifacts when log signals are weak.
    popup = audit_popup()
    maze = audit_maze()
    if popup["verdict"] == "PASS" and epics["epic1"] in {"unknown", "started"}:
        epics["epic1"] = "completed"
    if maze["maze_exists"] and epics["epic2"] in {"unknown", "started"}:
        epics["epic2"] = "completed" if maze["verdict"] == "PASS" else "partial"
    if maze["test_exists"] and epics["epic3"] in {"unknown", "started"}:
        epics["epic3"] = "completed" if maze["verdict"] == "PASS" else "partial"
    if all(epics[k] == "completed" for k in ("epic1", "epic2", "epic3")):
        epics["overall"] = "completed"

    ram_max = max(buf.ram_samples) if buf.ram_samples else None
    report = {
        "inject_status": status,
        "engine_exit": proc.returncode,
        "epics": epics,
        "popup": popup,
        "maze": maze,
        "model_hits": buf.model_hits,
        "ram_max": ram_max,
        "ram_samples": buf.ram_samples[-20:],
        "stream_tail": buf.snapshot(tail=8000),
    }
    append_psol(report)

    import json

    summary_path = ROOT / "logs" / "animation_bfs_report.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    mirror.write("\n# REPORT\n")
    mirror.write(json.dumps(report, indent=2, ensure_ascii=False))
    mirror.write("\n")
    mirror.close()

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    ok = (
        popup["verdict"] == "PASS"
        and maze["verdict"] == "PASS"
        and (ram_max is None or ram_max <= 92.0)
        and buf.model_hits >= 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
