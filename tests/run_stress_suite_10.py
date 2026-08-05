"""Diagnostic Suite 10 — API Resilience & True Cloud Escalation.

4-Epic Key-Value + TTL Meta-Broker stress under Hybrid Cloud, proving Gemini
429/503 exponential backoff and Worker Escalation without premature Ollama
fallback.

Usage::

    .venv\\Scripts\\python.exe tests/run_stress_suite_10.py
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
SUITE_MARKER = "### Diagnostic Suite 10: API Resilience & True Cloud Escalation"

PROMPT = (
    "/broker Build an in-memory Key-Value store with TTL. "
    "Epic 1: kv_store.py (core logic). "
    "Epic 2: tests/test_kv_store.py (TDD for set/get). "
    "Epic 3: tests/test_ttl.py (TDD for expiration). "
    "Epic 4: kv_cli.py (a simple CLI wrapper)."
)

ARTIFACTS = (
    "kv_store.py",
    "tests/test_kv_store.py",
    "tests/test_ttl.py",
    "kv_cli.py",
)

BOOT_DEADLINE_S = 120.0
PROMPT_BUDGET_S = 1800.0
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
    r"|Worker Escalation\]\s*Local model failed"
    r")",
    re.I,
)
_ESCALATION_RE = re.compile(
    r"\[Worker Escalation\]\s*Local model failed 3 times\. Escalating to Cloud API\.",
    re.I,
)
_THROTTLE_RE = re.compile(
    r"\[Cloud API Throttled\]\s*Retrying in\s+([\d.]+)s",
    re.I,
)
_THROTTLE_OK_RE = re.compile(
    r"\[Cloud API Throttled\]\s*recovered after\s+(\d+)\s+retry",
    re.I,
)
_FALLBACK_RE = re.compile(
    r"\[Worker Escalation\].*falling back to local Ollama",
    re.I,
)
_HARNESS_EXIT_RE = re.compile(
    r"\[RuntimeHarness\]\s+END\s+success=(\w+)\s+exit=(-?\d+)",
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
        reset_cloud_throttle_stats,
    )
    from dana.settings import is_hybrid_planner_enabled, set_hybrid_planner_enabled

    ensure_dotenv_loaded()
    reset_cloud_throttle_stats()
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
        print(f"[qa10] WARNING: settings mirror failed ({exc})", flush=True)
    print(
        f"[qa10] hybrid_planner_enabled=True (was {previous}); "
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
        print(f"[qa10] restored hybrid_planner_enabled={bool(previous)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa10] WARNING: hybrid restore failed ({exc})", flush=True)


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
) -> str:
    deadline = time.time() + budget_s
    quiet_since: float | None = None
    last_chars = prior_chars
    busy_re = re.compile(
        r"\[MetaBroker\]|\[DagSupervisor\]|\[DagWorker\]|\[RuntimeHarness\]|"
        r"\[CloudPlanner\]|\[Worker Escalation\]|\[Cloud API Throttled\]",
        re.I,
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            return "proc_exited"
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
                    f"[qa10] {label}: artifacts+broker settled — resolving",
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
                    f"[qa10] {label}: artifacts ready + quiet stall — resolving",
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
                f"[qa10] {label}: waiting… {left}s left "
                f"settled={settled} artifacts={artifacts}",
                flush=True,
            )
        time.sleep(1.0)
    if _artifacts_ready():
        return "timeout_with_artifacts"
    if _COMPLETION_RE.search(buf.snapshot(tail=40_000)):
        return "timeout_with_signature"
    return "timeout"


def _scan_resilience(stream_text: str, runtime_tail: str) -> dict[str, Any]:
    blob = f"{stream_text}\n{runtime_tail}"
    throttle_waits = _THROTTLE_RE.findall(blob)
    throttle_ok = _THROTTLE_OK_RE.findall(blob)
    esc = _ESCALATION_RE.findall(blob)
    fallbacks = _FALLBACK_RE.findall(blob)
    harness_exits = [
        (m.group(1), int(m.group(2))) for m in _HARNESS_EXIT_RE.finditer(blob)
    ]
    return {
        "escalation_triggered": bool(esc),
        "escalation_count": len(esc),
        "throttle_hits": len(throttle_waits),
        "throttle_waits_s": [float(x) for x in throttle_waits[:20]],
        "throttle_retries_ok": len(throttle_ok),
        "ollama_fallback_count": len(fallbacks),
        "harness_exits": harness_exits[-12:],
        "final_harness_exit": harness_exits[-1][1] if harness_exits else None,
    }


def audit_suite10() -> dict[str, Any]:
    paths = {rel: ROOT / rel for rel in ARTIFACTS}
    result: dict[str, Any] = {
        "file_verdicts": {
            rel: ("PASS" if paths[rel].is_file() else "FAIL") for rel in ARTIFACTS
        },
        "paths": {rel: str(paths[rel]) for rel in ARTIFACTS},
        "pytest_exit_code": None,
        "pytest_stdout": "",
        "pytest_stderr": "",
        "verdict": "FAIL",
        "detail": "",
    }
    missing = [rel for rel in ARTIFACTS if not paths[rel].is_file()]
    if missing:
        result["detail"] = f"missing files: {', '.join(missing)}"
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(ROOT)]
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                str(ROOT / "tests" / "test_kv_store.py"),
                str(ROOT / "tests" / "test_ttl.py"),
            ],
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
    result["pytest_stdout"] = (proc.stdout or "")[-2500:]
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


def _tail_runtime_log(n: int = 100) -> str:
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
    resilience: dict[str, Any],
    inject_status: str,
    stream_tail: str,
) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    v = audit.get("verdict") or "FAIL"
    fv = audit.get("file_verdicts") or {}
    esc = bool(resilience.get("escalation_triggered"))
    throttle_hits = int(resilience.get("throttle_hits") or 0)
    throttle_ok = int(resilience.get("throttle_retries_ok") or 0)
    cloud_resolved = (
        esc
        and int(audit.get("pytest_exit_code") or -1) == 0
        and int(resilience.get("ollama_fallback_count") or 0) == 0
    )
    section = "\n".join(
        [
            "",
            SUITE_MARKER,
            "",
            f"_Generated: {now}_",
            f"_Overall: `{v}`_",
            "",
            "4-Epic Hybrid Meta-Broker stress for in-memory KV+TTL with Gemini "
            "exponential backoff (429/503) and true Worker Escalation.",
            "",
            "| Check | Verdict | Detail |",
            "|---|---|---|",
            (
                f"| `10.1` kv_store.py | "
                f"`{fv.get('kv_store.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('kv_store.py')}` |"
            ),
            (
                f"| `10.2` tests/test_kv_store.py | "
                f"`{fv.get('tests/test_kv_store.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('tests/test_kv_store.py')}` |"
            ),
            (
                f"| `10.3` tests/test_ttl.py | "
                f"`{fv.get('tests/test_ttl.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('tests/test_ttl.py')}` |"
            ),
            (
                f"| `10.4` kv_cli.py | "
                f"`{fv.get('kv_cli.py', 'FAIL')}` | "
                f"`{audit.get('paths', {}).get('kv_cli.py')}` |"
            ),
            (
                f"| `10.5` pytest (kv+ttl) | `{v}` | "
                f"exit_code=`{audit.get('pytest_exit_code')}`; {audit.get('detail')} |"
            ),
            (
                f"| `10.6` Cloud API Throttle (429/503) | "
                f"`{'SEEN' if throttle_hits else 'NOT_SEEN'}` | "
                f"hits={throttle_hits}; retries_ok={throttle_ok}; "
                f"waits_s={resilience.get('throttle_waits_s')} |"
            ),
            (
                f"| `10.7` Worker Escalation | "
                f"`{'TRIGGERED' if esc else 'NOT_SEEN'}` | "
                f"count={resilience.get('escalation_count')}; "
                f"ollama_fallback={resilience.get('ollama_fallback_count')}; "
                f"cloud_resolved_exit0=`{'YES' if cloud_resolved else 'NO'}` |"
            ),
            "",
            "#### Prompt",
            "",
            f"`{PROMPT}`",
            "",
            f"- **Inject status:** `{inject_status}`",
            f"- **Final pytest exit_code:** `{audit.get('pytest_exit_code')}`",
            f"- **Final harness exit_code:** `{resilience.get('final_harness_exit')}`",
            f"- **429/503 throttle hits (retried):** `{throttle_hits}`",
            f"- **Throttle recoveries (request succeeded after backoff):** `{throttle_ok}`",
            f"- **[Worker Escalation] triggered:** `{'YES' if esc else 'NO'}`",
            f"- **Premature Ollama fallback count:** "
            f"`{resilience.get('ollama_fallback_count')}`",
            f"- **Cloud model resolved to exit_code=0 (no Ollama fallback):** "
            f"`{'YES' if cloud_resolved else 'NO'}`",
            "",
            "```text",
            (audit.get("pytest_stdout") or "(no pytest stdout)")[-1800:],
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
    # Redact any accidental API keys in stream tails.
    section = re.sub(r"([?&]key=)[^&\s\"')]+", r"\1***", section, flags=re.I)
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
        print("[qa10] cleared stale lock on 47473 (if any)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[qa10] WARNING: lock clear failed ({exc})", flush=True)


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    run_py = ROOT / "run.py"
    print(
        "=== Dana Diagnostic Suite 10 — API Resilience & Cloud Escalation ===",
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

    previous_hybrid: bool | None = None
    try:
        previous_hybrid = _force_hybrid_planner()
    except Exception as exc:  # noqa: BLE001
        print(f"[qa10] WARNING: hybrid setup failed ({exc})", flush=True)

    buf = StreamBuffer()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_env["DONNA_HEADLESS"] = "1"
    child_env["DONNA_DISABLE_TTS"] = "1"
    child_env["DONNA_DISABLE_MIC"] = "1"
    child_env["DONNA_DISABLE_IDLE_MONITOR"] = "1"
    child_env["DONNA_DISABLE_TOAST"] = "1"
    # Suite must exercise Cloud backoff even when host RAM is already elevated.
    child_env["DONNA_SKIP_RAM_BREAKER"] = "1"
    # Force early Worker Escalation so Gemini path + 429 backoff are observable.
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
            print("[qa10] BOOT FAILED — skipping inject", flush=True)
        else:
            print(
                "[qa10] boot OK — injecting KV+TTL Meta-Broker prompt",
                flush=True,
            )
            prior_chars = buf.chars()
            _write_trigger(PROMPT)
            inject_status = _wait_prompt_resolved(
                proc,
                buf,
                budget_s=PROMPT_BUDGET_S,
                prior_chars=prior_chars,
                label="suite10",
            )
            print(f"[qa10] inject status={inject_status}", flush=True)
    finally:
        print("[qa10] terminating headless agent...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _restore_hybrid(previous_hybrid)

    print("[qa10] running quality audits…", flush=True)
    audit = audit_suite10()
    resilience = _scan_resilience(
        buf.snapshot(tail=160_000), _tail_runtime_log(400)
    )
    print(
        f"[qa10] files={audit.get('file_verdicts')} "
        f"pytest={audit['pytest_exit_code']} verdict={audit['verdict']}",
        flush=True,
    )
    print(
        f"[qa10] throttle_hits={resilience['throttle_hits']} "
        f"retries_ok={resilience['throttle_retries_ok']} "
        f"escalation={resilience['escalation_triggered']} "
        f"ollama_fallback={resilience['ollama_fallback_count']}",
        flush=True,
    )
    psol = append_psol(
        audit=audit,
        resilience=resilience,
        inject_status=inject_status,
        stream_tail=buf.snapshot(tail=6000),
    )
    print(f"[qa10] PSOL appended -> {psol}", flush=True)
    overall = audit.get("verdict") == "PASS"
    print(f"[qa10] OVERALL {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
