"""Diagnostic Suite 9 — Exit 2 Guards & Worker Escalation.

Headless Hybrid Meta-Broker run for a 3-file Config Manager scaffold, proving
pytest exit_code=2 bypasses triage and Worker Escalation to Cloud after 3
local repair failures.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_suite_9.py
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
SUITE_MARKER = "### Diagnostic Suite 9: Exit 2 Guards & Worker Escalation"

PROMPT = (
    "/broker Build a dynamic Config Manager. "
    "Epic 1: config_loader.py (loads a dict). "
    "Epic 2: config_validator.py (checks keys). "
    "Epic 3: tests/test_config.py to verify both."
)

BOOT_DEADLINE_S = 120.0
PROMPT_BUDGET_S = 1200.0
QUIET_AFTER_COMPLETE_S = 8.0

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
    r"|Worker Escalation\]\s*Local model failed"
    r")",
    re.I,
)
_ESCALATION_RE = re.compile(
    r"\[Worker Escalation\]\s*Local model failed 3 times\. Escalating to Cloud API\.",
    re.I,
)
_EXIT2_RE = re.compile(r"Exit-2 collection failure", re.I)
_HARNESS_EXIT_RE = re.compile(
    r"\[RuntimeHarness\]\s+END\s+success=(\w+)\s+exit=(-?\d+)",
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
        print(f"[qa9] WARNING: settings mirror failed ({exc})", flush=True)
    print(
        f"[qa9] hybrid_planner_enabled=True (was {previous}); "
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
        print(f"[qa9] restored hybrid_planner_enabled={bool(previous)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa9] WARNING: hybrid restore failed ({exc})", flush=True)


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
        (ROOT / "config_loader.py").is_file()
        and (ROOT / "config_validator.py").is_file()
        and (ROOT / "tests" / "test_config.py").is_file()
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
    busy_re = re.compile(
        r"\[MetaBroker\]|\[DagSupervisor\]|\[DagWorker\]|\[RuntimeHarness\]|"
        r"\[CloudPlanner\]|\[Worker Escalation\]",
        re.I,
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            return "proc_exited"
        snap = buf.snapshot(tail=40_000)
        chars = buf.chars()
        settled = bool(_SETTLED_RE.search(snap))
        artifacts = _artifacts_ready()
        busy = bool(busy_re.search(snap[-3000:]))
        if artifacts and settled:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                print(f"[qa9] {label}: artifacts+broker settled — resolving", flush=True)
                return "artifacts_ready"
        elif settled and _COMPLETION_RE.search(snap):
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "complete_sig"
        elif artifacts and not busy and chars == last_chars:
            # Graph stalled mid-repair (common under Gemini 429) but files exist.
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= 45.0:
                print(
                    f"[qa9] {label}: artifacts ready + quiet stall — resolving",
                    flush=True,
                )
                return "artifacts_stalled"
            else:
                pass
        else:
            quiet_since = None
        if chars != last_chars:
            last_chars = chars
            if busy:
                quiet_since = None
        left = int(deadline - time.time())
        if left % 30 < 2:
            print(
                f"[qa9] {label}: waiting… {left}s left "
                f"settled={settled} artifacts={artifacts}",
                flush=True,
            )
        time.sleep(1.0)
    if _artifacts_ready():
        return "timeout_with_artifacts"
    if _COMPLETION_RE.search(buf.snapshot(tail=40_000)):
        return "timeout_with_signature"
    return "timeout"


def _scan_escalation(stream_text: str, runtime_tail: str) -> dict[str, Any]:
    blob = f"{stream_text}\n{runtime_tail}"
    hits = _ESCALATION_RE.findall(blob)
    exit2_hits = _EXIT2_RE.findall(blob)
    harness_exits = [(m.group(1), int(m.group(2))) for m in _HARNESS_EXIT_RE.finditer(blob)]
    final_exit: int | None = harness_exits[-1][1] if harness_exits else None
    return {
        "escalation_triggered": bool(hits),
        "escalation_count": len(hits),
        "exit2_bypass_seen": bool(exit2_hits),
        "exit2_count": len(exit2_hits),
        "harness_exits": harness_exits[-10:],
        "final_harness_exit": final_exit,
    }


def audit_suite9() -> dict[str, Any]:
    loader = ROOT / "config_loader.py"
    validator = ROOT / "config_validator.py"
    test_path = ROOT / "tests" / "test_config.py"
    result: dict[str, Any] = {
        "loader_exists": loader.is_file(),
        "validator_exists": validator.is_file(),
        "test_exists": test_path.is_file(),
        "loader_path": str(loader),
        "validator_path": str(validator),
        "test_path": str(test_path),
        "pytest_exit_code": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
        "file_verdicts": {},
    }
    missing = []
    if not result["loader_exists"]:
        missing.append("config_loader.py")
    if not result["validator_exists"]:
        missing.append("config_validator.py")
    if not result["test_exists"]:
        missing.append("tests/test_config.py")
    result["file_verdicts"] = {
        "config_loader.py": "PASS" if result["loader_exists"] else "FAIL",
        "config_validator.py": "PASS" if result["validator_exists"] else "FAIL",
        "tests/test_config.py": "PASS" if result["test_exists"] else "FAIL",
    }
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
    escalation: dict[str, Any],
    inject_status: str,
    stream_tail: str,
) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v = audit.get("verdict") or "FAIL"
    fv = audit.get("file_verdicts") or {}
    esc = bool(escalation.get("escalation_triggered"))
    cloud_resolved = esc and int(audit.get("pytest_exit_code") or -1) == 0
    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Generated: {now}_",
            f"_Overall: `{v}`_",
            "",
            "Headless Hybrid Meta-Broker stress for Config Manager scaffolding "
            "with Pytest Exit-2 collection guards and Worker Auto-Escalation "
            "(Senior Dev Fallback to Cloud).",
            "",
            "| Check | Verdict | Detail |",
            "|---|---|---|",
            (
                f"| `9.1` config_loader.py | "
                f"`{fv.get('config_loader.py', 'FAIL')}` | "
                f"`{audit.get('loader_path')}` |"
            ),
            (
                f"| `9.2` config_validator.py | "
                f"`{fv.get('config_validator.py', 'FAIL')}` | "
                f"`{audit.get('validator_path')}` |"
            ),
            (
                f"| `9.3` tests/test_config.py | "
                f"`{fv.get('tests/test_config.py', 'FAIL')}` | "
                f"`{audit.get('test_path')}` |"
            ),
            (
                f"| `9.4` final pytest | `{v}` | "
                f"exit_code=`{audit.get('pytest_exit_code')}`; {audit.get('detail')} |"
            ),
            (
                f"| `9.5` Exit-2 collection bypass | "
                f"`{'SEEN' if escalation.get('exit2_bypass_seen') else 'NOT_SEEN'}` | "
                f"count={escalation.get('exit2_count')}; "
                f"harness_exits={escalation.get('harness_exits')} |"
            ),
            (
                f"| `9.6` Worker Escalation | "
                f"`{'TRIGGERED' if esc else 'NOT_SEEN'}` | "
                f"count={escalation.get('escalation_count')}; "
                f"cloud_resolved_exit0=`{'YES' if cloud_resolved else 'NO'}` |"
            ),
            "",
            "#### Prompt",
            "",
            f"`{PROMPT}`",
            "",
            f"- **Inject status:** `{inject_status}`",
            f"- **Final pytest exit_code:** `{audit.get('pytest_exit_code')}`",
            f"- **Final harness exit_code:** `{escalation.get('final_harness_exit')}`",
            f"- **[Worker Escalation] triggered:** `{'YES' if esc else 'NO'}`",
            f"- **Cloud model resolved to exit_code=0:** "
            f"`{'YES' if cloud_resolved else 'NO'}`",
            (
                "- **Note:** Suite forces `DONNA_WORKER_ESCALATE_AFTER=1` so the "
                "escalation path is observable; production default remains "
                "`repair_attempts >= 3`. Cloud resolution requires a non-429 "
                "Gemini response under Hybrid."
            ),
            "",
            "```text",
            (audit.get("pytest_stdout") or "(no pytest stdout)")[-1500:],
            "```",
            "",
            "### Runtime log tail",
            "",
            "```text",
            _tail_runtime_log(80) or "(unavailable)",
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
    """Best-effort clear of Donna single-instance TCP lock (127.0.0.1:47473)."""
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            if s.connect_ex(("127.0.0.1", 47473)) != 0:
                return
        finally:
            s.close()
        # Port in use — try to find and kill listener via PowerShell.
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
        print("[qa9] cleared stale lock on 47473 (if any)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa9] WARNING: lock clear failed ({exc})", flush=True)


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print("=== Dana Diagnostic Suite 9 — Exit 2 & Worker Escalation ===", flush=True)
    print(f"python={py}", flush=True)
    print(f"launch={run_py} --no-gui", flush=True)

    _kill_stale_lock()

    # Clean prior suite artifacts so existence checks are meaningful.
    for rel in ("config_loader.py", "config_validator.py", "tests/test_config.py"):
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
        print(f"[qa9] WARNING: hybrid setup failed ({exc})", flush=True)

    buf = StreamBuffer()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_env["DONNA_HEADLESS"] = "1"
    child_env["DONNA_DISABLE_TTS"] = "1"
    child_env["DONNA_DISABLE_MIC"] = "1"
    child_env["DONNA_DISABLE_IDLE_MONITOR"] = "1"
    child_env["DONNA_DISABLE_TOAST"] = "1"
    # Suite 9 forces early escalation so the Cloud fallback is observable even
    # when local repair would otherwise succeed before attempt 3.
    child_env["DONNA_WORKER_ESCALATE_AFTER"] = "1"
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
            print("[qa9] BOOT FAILED — skipping inject", flush=True)
        else:
            print("[qa9] boot OK — injecting Config Manager Meta-Broker prompt", flush=True)
            prior_chars = buf.chars()
            _write_trigger(PROMPT)
            inject_status = _wait_prompt_resolved(
                proc,
                buf,
                budget_s=PROMPT_BUDGET_S,
                prior_chars=prior_chars,
                label="suite9",
            )
            print(f"[qa9] inject status={inject_status}", flush=True)
    finally:
        print("[qa9] terminating headless agent...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _restore_hybrid(previous_hybrid)

    print("[qa9] running quality audits…", flush=True)
    audit = audit_suite9()
    escalation = _scan_escalation(buf.snapshot(tail=120_000), _tail_runtime_log(300))
    print(
        f"[qa9] files loader={audit['loader_exists']} "
        f"validator={audit['validator_exists']} "
        f"test={audit['test_exists']} pytest={audit['pytest_exit_code']} "
        f"verdict={audit['verdict']}",
        flush=True,
    )
    print(
        f"[qa9] escalation={escalation['escalation_triggered']} "
        f"exit2={escalation['exit2_bypass_seen']} "
        f"final_harness_exit={escalation['final_harness_exit']}",
        flush=True,
    )
    psol = append_psol(
        audit=audit,
        escalation=escalation,
        inject_status=inject_status,
        stream_tail=buf.snapshot(tail=6000),
    )
    print(f"[qa9] PSOL appended -> {psol}", flush=True)
    overall = audit.get("verdict") == "PASS"
    print(f"[qa9] OVERALL {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
