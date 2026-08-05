"""LRU Cache Meta-Broker suite — force-local, keep_alive=0, isolated process.

Usage::

    .venv\\Scripts\\python.exe tests/run_lru_cache_suite.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRIGGER = ROOT / ".trigger_ask"
SUITE_LOG = ROOT / "logs" / "lru_cache_suite.log"
MANIFEST = ROOT / ".dana_scratch" / "manifest.json"

BROKER_PROMPT = (
    "/broker Epic 1: Write an LRU Cache implementation in lru_cache.py with a "
    "class LRUCache supporting get(key) and put(key, value) operations. "
    "Epic 2: Write cache_simulator.py that imports LRUCache from lru_cache, "
    "runs a simulation of 100 cache reads/writes, and outputs a summary dict "
    "of hit/miss stats. "
    "Epic 3: Write tests/test_cache.py using Pytest to verify capacity eviction, "
    "key updates, and non-existent key returns."
)

BOOT_DEADLINE_S = 150.0
PROMPT_BUDGET_S = 900.0
QUIET_AFTER_COMPLETE_S = 8.0

_COMPLETION_RE = re.compile(
    r"("
    r"All\s+\d+\s+epic\(s\)\s+completed"
    r"|OK:\s*meta_broker status=completed"
    r"|graph\.invoke END status='completed'"
    r"|Meta-Broker finished status=completed"
    r")",
    re.I,
)
_FAIL_RE = re.compile(
    r"("
    r"graph\.invoke END status='failed'"
    r"|OK:\s*meta_broker status=failed"
    r"|ERROR:\s*meta_broker"
    r"|epic\s+\d+\s+exhausted repairs"
    r"|CRITICAL:\s*RAM usage"
    r")",
    re.I,
)
_BUSY_RE = re.compile(
    r"\[MetaBroker\]|\[DagSupervisor\]|\[DagWorker\]|RuntimeHarness|qwen2\.5-coder",
    re.I,
)
_GEMINI_429_RE = re.compile(r"429|Cloud API Throttled|generativelanguage\.googleapis", re.I)
_RAM_RE = re.compile(
    r"(?:ram[=:\s]+|health ram=|RAM breaker skipped.*?ram=)(\d+(?:\.\d+)?)\s*%?",
    re.I,
)


class StreamBuffer:
    def __init__(self, *, maxlen: int = 800_000) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self.text = ""
        self.total_chars = 0
        self.ram_samples: list[float] = []
        self.gemini_429 = 0
        self._maxlen = maxlen

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
            self.gemini_429 += len(_GEMINI_429_RE.findall(data))
            for m in _RAM_RE.finditer(data):
                try:
                    self.ram_samples.append(float(m.group(1)))
                except ValueError:
                    pass

    def snapshot(self, *, tail: int = 8000) -> str:
        with self._lock:
            return self.text[-tail:]

    def full(self) -> str:
        with self._lock:
            return self.text

    def chars(self) -> int:
        with self._lock:
            return self.total_chars


def _reader(pipe, buf: StreamBuffer, mirror, label: str) -> None:
    pending = ""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            text = (
                chunk.decode("utf-8", errors="replace")
                if isinstance(chunk, bytes)
                else str(chunk)
            )
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
                mirror.write(f"[{label}] {pending}\n")
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


def _artifacts_ready() -> bool:
    return all(
        (ROOT / name).is_file() and (ROOT / name).stat().st_size > 40
        for name in ("lru_cache.py", "cache_simulator.py", "tests/test_cache.py")
    )


def _wait_boot(proc: subprocess.Popen, buf: StreamBuffer, deadline: float) -> bool:
    needles = (
        "Headless mode: engine auto-ENGAGED",
        "Headless mode (--no-gui)",
        "Dana is ready",
        "Noise floor calibrated",
    )
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        snap = buf.snapshot(tail=30_000)
        if any(n in snap for n in needles):
            time.sleep(1.5)
            return True
        time.sleep(0.25)
    return False


def _wait_done(proc: subprocess.Popen, buf: StreamBuffer, budget_s: float) -> str:
    deadline = time.time() + budget_s
    quiet_since: float | None = None
    last_chars = buf.chars()
    last_log = 0.0
    saw_complete = False
    saw_fail = False
    while time.time() < deadline:
        if proc.poll() is not None:
            return "process_exited" if not _artifacts_ready() else "artifacts_ready"
        snap = buf.full()
        chars = buf.chars()
        growing = chars > last_chars
        if growing:
            last_chars = chars
            quiet_since = None
        if _COMPLETION_RE.search(snap):
            saw_complete = True
        if _FAIL_RE.search(snap[-12000:]):
            saw_fail = True
        if _artifacts_ready() and (saw_complete or saw_fail):
            if quiet_since is None and not growing:
                quiet_since = time.time()
            elif quiet_since and not growing and time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "artifacts_ready" if saw_complete else "broker_failed_with_artifacts"
        if saw_complete and quiet_since is None and not growing:
            quiet_since = time.time()
        elif saw_complete and quiet_since and not growing:
            if time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "completion_signature"
        if saw_fail and not growing:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= QUIET_AFTER_COMPLETE_S:
                return "broker_failed"
        if _BUSY_RE.search(snap[-2500:]):
            quiet_since = None
        now = time.time()
        if now - last_log >= 30.0:
            try:
                import psutil

                buf.ram_samples.append(float(psutil.virtual_memory().percent))
            except Exception:
                pass
            print(
                f"[lru] waiting… {max(0, deadline-now):.0f}s "
                f"complete={saw_complete} fail={saw_fail} "
                f"artifacts={_artifacts_ready()} "
                f"ram_max={max(buf.ram_samples) if buf.ram_samples else 'n/a'} "
                f"gemini429={buf.gemini_429}",
                flush=True,
            )
            last_log = now
        time.sleep(0.5)
    if _artifacts_ready():
        return "timeout_with_artifacts"
    return "timeout"


def main() -> int:
    SUITE_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        if TRIGGER.is_file():
            TRIGGER.unlink()
    except OSError:
        pass

    env = os.environ.copy()
    env["DONNA_FORCE_LOCAL"] = "1"
    env["DONNA_OLLAMA_KEEP_ALIVE"] = "0"
    env["DONNA_OLLAMA_MODEL"] = "qwen2.5-coder:7b"
    env["OLLAMA_MODEL"] = "qwen2.5-coder:7b"
    env["DONNA_LOCAL_MODEL"] = "qwen2.5-coder:7b"
    env["DONNA_SKIP_BOOT_READY"] = "1"
    env["DONNA_SKIP_RAM_BREAKER"] = "1"
    env["DONNA_NO_GUI"] = "1"
    env["DONNA_HEADLESS"] = "1"
    env["DONNA_META_BROKER_LOG"] = str(SUITE_LOG)
    env["DONNA_META_BROKER_TIMEOUT_S"] = "300"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    mirror = SUITE_LOG.open("w", encoding="utf-8", errors="replace")
    mirror.write(f"# LRU suite {datetime.now().isoformat()}\n# prompt={BROKER_PROMPT}\n")
    mirror.flush()

    cmd = [sys.executable, str(ROOT / "run.py"), "--no-gui"]
    print(f"[lru] launching {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    buf = StreamBuffer()
    t_out = threading.Thread(target=_reader, args=(proc.stdout, buf, mirror, "OUT"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, buf, mirror, "ERR"), daemon=True)
    t_out.start()
    t_err.start()

    if not _wait_boot(proc, buf, time.time() + BOOT_DEADLINE_S):
        print("[lru] BOOT FAILED", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
        mirror.close()
        return 2

    print("[lru] injecting broker prompt", flush=True)
    TRIGGER.write_text(BROKER_PROMPT + "\n", encoding="utf-8")
    status = _wait_done(proc, buf, PROMPT_BUDGET_S)
    print(f"[lru] status={status}", flush=True)

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
    try:
        if TRIGGER.is_file():
            TRIGGER.unlink()
    except OSError:
        pass

    report = {
        "status": status,
        "artifacts": _artifacts_ready(),
        "ram_max": max(buf.ram_samples) if buf.ram_samples else None,
        "gemini_429_hits": buf.gemini_429,
        "manifest_exists": MANIFEST.is_file(),
    }
    mirror.write("\n# REPORT\n" + json.dumps(report, indent=2) + "\n")
    mirror.close()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if _artifacts_ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
