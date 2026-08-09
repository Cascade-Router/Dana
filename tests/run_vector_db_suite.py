"""Suite 12 — Vector DB Engine Meta-Broker harness.

Launches an isolated Meta-Broker process (force-local qwen2.5-coder:7b),
logs telemetry to ``logs/suite_12.log``, then runs pytest remediation.

Usage::

    .venv\\Scripts\\python.exe tests/run_vector_db_suite.py
"""

from __future__ import annotations

import json
import os
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

SUITE_LOG = ROOT / "logs" / "suite_12.log"
MANIFEST = ROOT / ".dana_scratch" / "manifest.json"
ARTIFACTS = (
    ROOT / "vector_math.py",
    ROOT / "local_vector_db.py",
    ROOT / "tests" / "test_vector_db.py",
)

BROKER_PROMPT = (
    "/broker Epic 1: Write vector_math.py containing a class VectorDocument and "
    "mathematical functions for cosine_similarity and euclidean_distance between "
    "two lists of floats. Epic 2: Write local_vector_db.py that imports "
    "VectorDocument and the math functions from vector_math. Implement a "
    "LocalVectorDB class that supports add_document(doc), save_to_disk(filepath), "
    "load_from_disk(filepath), and search_top_k(query_vector, k). Epic 3: Write "
    "tests/test_vector_db.py using Pytest to verify adding documents, "
    "saving/loading the JSON state to a temporary directory, and accurately "
    "retrieving the top-K closest vectors using cosine similarity."
)


def _artifacts_ready() -> bool:
    return all(p.is_file() and p.stat().st_size > 40 for p in ARTIFACTS)


def _configure_env() -> dict[str, str]:
    env = os.environ.copy()
    env["DANA_FORCE_LOCAL"] = "1"
    env["DANA_OLLAMA_KEEP_ALIVE"] = "0"
    env["DANA_OLLAMA_MODEL"] = "qwen2.5-coder:7b"
    env["OLLAMA_MODEL"] = "qwen2.5-coder:7b"
    env["DANA_LOCAL_MODEL"] = "qwen2.5-coder:7b"
    env["DANA_SKIP_BOOT_READY"] = "1"
    env["DANA_SKIP_RAM_BREAKER"] = "1"
    env["DANA_NO_GUI"] = "1"
    env["DANA_HEADLESS"] = "1"
    env["DANA_META_BROKER_LOG"] = str(SUITE_LOG)
    env["DANA_META_BROKER_TIMEOUT_S"] = "900"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _ram_pct() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        return None


def _run_broker(log_fh: Any) -> dict[str, Any]:
    """Invoke isolated Meta-Broker; mirror telemetry into suite_12.log."""
    for k, v in _configure_env().items():
        os.environ[k] = v

    from dana.graph.meta_broker_process import (
        run_meta_broker_isolated,
        start_headless_telemetry_drainer,
    )

    start_headless_telemetry_drainer(log_path=SUITE_LOG)

    events: list[dict[str, Any]] = []
    tts_notes: list[str] = []
    ram_samples: list[float] = []

    def _on_event(ev: dict[str, Any]) -> None:
        events.append(dict(ev))
        line = json.dumps(ev, default=str)
        try:
            log_fh.write(f"[TELEMETRY] {line}\n")
            log_fh.flush()
        except Exception:
            pass
        msg = str(ev.get("message") or "")
        low = msg.lower()
        if "starting epic" in low or "dispatch epic" in low or "task complete" in low:
            tts_notes.append(msg)
            try:
                from dana.audio.tts_manager import get_tts_manager

                get_tts_manager().notify(msg.split(":")[0].strip() + ".")
                log_fh.write(f"[TTS] queued: {msg[:120]}\n")
                log_fh.flush()
            except Exception as exc:  # noqa: BLE001
                log_fh.write(f"[TTS] enqueue failed: {exc}\n")
                log_fh.flush()
        r = _ram_pct()
        if r is not None:
            ram_samples.append(r)

    stop_ram = threading.Event()

    def _ram_poll() -> None:
        while not stop_ram.wait(5.0):
            r = _ram_pct()
            if r is not None:
                ram_samples.append(r)
                try:
                    log_fh.write(f"[RAM] {r:.1f}%\n")
                    log_fh.flush()
                except Exception:
                    pass

    poller = threading.Thread(target=_ram_poll, name="Suite12Ram", daemon=True)
    poller.start()
    t0 = time.time()
    log_fh.write(f"[SUITE12] broker start t={datetime.now().isoformat()}\n")
    log_fh.write(f"[SUITE12] prompt={BROKER_PROMPT}\n")
    log_fh.flush()
    print("[suite12] launching isolated Meta-Broker…", flush=True)
    try:
        result = run_meta_broker_isolated(
            BROKER_PROMPT,
            on_event=_on_event,
            timeout_s=900.0,
            env_extra=_configure_env(),
        )
    finally:
        stop_ram.set()
    elapsed = time.time() - t0
    report = {
        "elapsed_s": round(elapsed, 1),
        "status": str((result or {}).get("status") or ""),
        "error": str((result or {}).get("error") or "")[:500],
        "artifacts_ready": _artifacts_ready(),
        "ram_peak": max(ram_samples) if ram_samples else None,
        "ram_samples": len(ram_samples),
        "telemetry_events": len(events),
        "tts_notifications": tts_notes,
        "manifest_exists": MANIFEST.is_file(),
        "epic_log": list((result or {}).get("epic_log") or [])[-12:],
    }
    log_fh.write("\n# BROKER_REPORT\n" + json.dumps(report, indent=2) + "\n")
    log_fh.flush()
    print(json.dumps(report, indent=2), flush=True)
    return {"result": result or {}, "report": report}


def _run_pytest() -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / "tests" / "test_vector_db.py"),
        "-q",
        "--tb=short",
    ]
    print(f"[suite12] pytest: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_configure_env(),
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return int(proc.returncode), out


def main() -> int:
    SUITE_LOG.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".dana_scratch").mkdir(parents=True, exist_ok=True)
    with SUITE_LOG.open("w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"# Suite 12 Vector DB {datetime.now().isoformat()}\n")
        log_fh.flush()
        broker = _run_broker(log_fh)
        report = broker["report"]

        if not _artifacts_ready():
            log_fh.write("[SUITE12] artifacts missing after broker — exit 1\n")
            print("[suite12] ARTIFACTS MISSING", flush=True)
            return 1

        code, pytest_out = _run_pytest()
        log_fh.write("\n# PYTEST\n" + pytest_out + "\n")
        log_fh.write(f"# pytest_exit={code}\n")
        log_fh.flush()
        print(pytest_out, flush=True)
        report["pytest_exit"] = code
        log_fh.write("\n# FINAL\n" + json.dumps(report, indent=2) + "\n")
        return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
