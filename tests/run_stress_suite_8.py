"""Diagnostic Suite 8 — Complex Scaffolding & Bidirectional Repair.

Headless Hybrid Meta-Broker run for a 3-file Event Bus scaffold, then quality
gates + PSOL append.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_suite_8.py
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
SUITE_MARKER = "### Diagnostic Suite 8: Complex Scaffolding & Bidirectional Repair"

PROMPT = (
    "/broker Build an Event Bus system. "
    "Epic 1: Write core interface in core_bus.py. "
    "Epic 2: Write a stdout logging plugin in logger_plugin.py that imports the core. "
    "Epic 3: Write tests/test_event_bus.py to verify the plugin registers to the core. "
    "Ensure tests pass."
)

BOOT_DEADLINE_S = 120.0
PROMPT_BUDGET_S = 900.0
QUIET_AFTER_COMPLETE_S = 8.0
TOTAL_DEADLINE_S = 1800.0

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
    r"|Bidirectional Repair Triage"
    r")",
    re.I,
)
_BUSY_RE = re.compile(
    r"("
    r"\[MetaBroker\]"
    r"|\[DagSupervisor\]"
    r"|\[DagWorker\]"
    r"|\[RuntimeHarness\]"
    r"|Bidirectional Repair Triage"
    r"|Loading weights|ChatOllama"
    r")",
    re.I,
)
_TRIAGE_RE = re.compile(
    r"Bidirectional Repair Triage\s*(?:→|=)\s*(TEST|CODE)",
    re.I,
)


class StreamBuffer:
    def __init__(self, *, maxlen: int = 500_000) -> None:
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


def _force_hybrid_planner() -> bool:
    from dana.graph.cloud_planner import (
        cloud_planner_key_present,
        ensure_dotenv_loaded,
        planner_mode_label,
    )
    from dana.settings import is_hybrid_planner_enabled, set_hybrid_planner_enabled

    ensure_dotenv_loaded()
    previous = bool(is_hybrid_planner_enabled())
    set_hybrid_planner_enabled(True)
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
        payload["hybrid_planner_enabled"] = True
        SETTINGS_PATH.write_text(
            _json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[qa8] WARNING: settings mirror failed ({exc})", flush=True)
    print(
        f"[qa8] hybrid_planner_enabled=True (was {previous}); "
        f"api_key_present={cloud_planner_key_present()}; "
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
        print(f"[qa8] restored hybrid_planner_enabled={bool(previous)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa8] WARNING: hybrid restore failed ({exc})", flush=True)


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
    return (
        (ROOT / "core_bus.py").is_file()
        and (ROOT / "logger_plugin.py").is_file()
        and (ROOT / "tests" / "test_event_bus.py").is_file()
    )


def _wait_prompt_resolved(
    proc: subprocess.Popen,
    buf: StreamBuffer,
    *,
    budget_s: float,
    prior_chars: int,
    label: str,
) -> str:
    deadline = time.time() + budget_s
    quiet_since: float | None = None
    last_chars = prior_chars
    while time.time() < deadline:
        if proc.poll() is not None:
            return "proc_exited"
        snap = buf.snapshot(tail=40_000)
        chars = buf.chars()
        settled = bool(_SETTLED_RE.search(snap))
        if _artifacts_ready() and settled:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                print(f"[qa8] {label}: artifacts+broker settled — resolving", flush=True)
                return "artifacts_ready"
        elif settled and _COMPLETION_RE.search(snap):
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "complete_sig"
        else:
            quiet_since = None
        if chars != last_chars:
            last_chars = chars
        left = int(deadline - time.time())
        if left % 30 < 2:
            print(
                f"[qa8] {label}: waiting… {left}s left "
                f"settled={settled} artifacts={_artifacts_ready()}",
                flush=True,
            )
        time.sleep(1.0)
    if _artifacts_ready():
        return "timeout_with_artifacts"
    if _COMPLETION_RE.search(buf.snapshot(tail=40_000)):
        return "timeout_with_signature"
    return "timeout"


def _scan_triage(stream_text: str, runtime_tail: str) -> dict[str, Any]:
    blob = f"{stream_text}\n{runtime_tail}"
    hits = _TRIAGE_RE.findall(blob)
    test_hits = sum(1 for h in hits if str(h).upper() == "TEST")
    code_hits = sum(1 for h in hits if str(h).upper() == "CODE")
    return {
        "triggered": bool(hits),
        "count": len(hits),
        "test_blame_count": test_hits,
        "code_blame_count": code_hits,
        "raw_hits": hits[:20],
    }


def audit_suite8() -> dict[str, Any]:
    core = ROOT / "core_bus.py"
    plugin = ROOT / "logger_plugin.py"
    test_path = ROOT / "tests" / "test_event_bus.py"
    result: dict[str, Any] = {
        "core_exists": core.is_file(),
        "plugin_exists": plugin.is_file(),
        "test_exists": test_path.is_file(),
        "core_path": str(core),
        "plugin_path": str(plugin),
        "test_path": str(test_path),
        "pytest_exit_code": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    missing = []
    if not result["core_exists"]:
        missing.append("core_bus.py")
    if not result["plugin_exists"]:
        missing.append("logger_plugin.py")
    if not result["test_exists"]:
        missing.append("tests/test_event_bus.py")
    if missing:
        result["detail"] = f"missing files: {', '.join(missing)}"
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)]
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
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
        result["pytest_exit_code"] = -1
        result["detail"] = "TIMEOUT: pytest exceeded 90s"
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
    triage: dict[str, Any],
    inject_status: str,
    stream_tail: str,
) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v = audit.get("verdict") or "FAIL"
    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Generated: {now}_",
            f"_Overall: `{v}`_",
            "",
            "Headless Hybrid Meta-Broker stress for 3-file Event Bus scaffolding "
            "with Bidirectional Repair Triage.",
            "",
            "| Check | Verdict | Detail |",
            "|---|---|---|",
            (
                f"| `8.1` core_bus.py | "
                f"`{'PASS' if audit.get('core_exists') else 'FAIL'}` | "
                f"`{audit.get('core_path')}` |"
            ),
            (
                f"| `8.2` logger_plugin.py | "
                f"`{'PASS' if audit.get('plugin_exists') else 'FAIL'}` | "
                f"`{audit.get('plugin_path')}` |"
            ),
            (
                f"| `8.3` tests/test_event_bus.py | "
                f"`{'PASS' if audit.get('test_exists') else 'FAIL'}` | "
                f"`{audit.get('test_path')}` |"
            ),
            (
                f"| `8.4` pytest | `{v}` | "
                f"exit_code=`{audit.get('pytest_exit_code')}`; {audit.get('detail')} |"
            ),
            (
                f"| `8.5` Bidirectional Repair Triage | "
                f"`{'TRIGGERED' if triage.get('triggered') else 'NOT_SEEN'}` | "
                f"count={triage.get('count')}; "
                f"TEST_blame={triage.get('test_blame_count')}; "
                f"CODE_blame={triage.get('code_blame_count')}; "
                f"hits={triage.get('raw_hits')} |"
            ),
            "",
            "#### Prompt",
            "",
            f"`{PROMPT}`",
            "",
            f"- **Inject status:** `{inject_status}`",
            f"- **pytest exit_code:** `{audit.get('pytest_exit_code')}`",
            f"- **Triage caught a bad test?** "
            f"`{'YES' if int(triage.get('test_blame_count') or 0) > 0 else 'NO'}`",
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


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print("=== Dana Diagnostic Suite 8 — Complex Scaffolding ===", flush=True)
    print(f"python={py}", flush=True)
    print(f"launch={run_py} --no-gui", flush=True)

    # Clean prior suite artifacts so existence checks are meaningful.
    for rel in ("core_bus.py", "logger_plugin.py", "tests/test_event_bus.py"):
        p = ROOT / rel
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass

    previous_hybrid: bool | None = None
    try:
        previous_hybrid = _force_hybrid_planner()
    except Exception as exc:  # noqa: BLE001
        print(f"[qa8] WARNING: hybrid setup failed ({exc})", flush=True)

    buf = StreamBuffer()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
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
    try:
        if not _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S):
            print("[qa8] BOOT FAILED — skipping inject", flush=True)
        else:
            print("[qa8] boot OK — injecting Event Bus Meta-Broker prompt", flush=True)
            prior_chars = buf.chars()
            _write_trigger(PROMPT)
            inject_status = _wait_prompt_resolved(
                proc,
                buf,
                budget_s=PROMPT_BUDGET_S,
                prior_chars=prior_chars,
                label="suite8",
            )
            print(f"[qa8] inject status={inject_status}", flush=True)
    finally:
        print("[qa8] terminating headless agent...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _restore_hybrid(previous_hybrid)

    print("[qa8] running quality audits…", flush=True)
    audit = audit_suite8()
    triage = _scan_triage(buf.snapshot(tail=80_000), _tail_runtime_log(200))
    print(
        f"[qa8] files core={audit['core_exists']} plugin={audit['plugin_exists']} "
        f"test={audit['test_exists']} pytest={audit['pytest_exit_code']} "
        f"verdict={audit['verdict']}",
        flush=True,
    )
    print(
        f"[qa8] triage triggered={triage['triggered']} "
        f"TEST={triage['test_blame_count']} CODE={triage['code_blame_count']}",
        flush=True,
    )
    psol = append_psol(
        audit=audit,
        triage=triage,
        inject_status=inject_status,
        stream_tail=buf.snapshot(tail=6000),
    )
    print(f"[qa8] PSOL appended → {psol}", flush=True)
    overall = audit.get("verdict") == "PASS"
    print(f"[qa8] OVERALL {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
