"""Suite 13 — Vision math + ROS2/MCP swarm brief (Jarvis test).

Usage::

    .venv\\Scripts\\python.exe tests/run_suite_13.py
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

SUITE_LOG = ROOT / "logs" / "suite_13.log"
MANIFEST = ROOT / ".dana_scratch" / "manifest.json"
ARTIFACTS = (
    ROOT / "vision_math.py",
    ROOT / "swarm_mcp.md",
    ROOT / "tests" / "test_vision_math.py",
)

BROKER_PROMPT = (
    "/broker Epic 1: Activate vision grounding to analyze a test image "
    "(generate a dummy image containing the equation \"f(x) = x^3 - 4x + 1\" "
    "if a camera is unavailable) and write vision_math.py with a function that "
    "calculates the roots of that equation. Epic 2: Generate a highly technical "
    "research brief in 'swarm_mcp.md' detailing a decentralized swarm architecture "
    "that integrates ROS2/Nav2 with LangGraph and Model Context Protocol (MCP) "
    "servers, allowing distributed edge-agents to dynamically call tools across "
    "the swarm. Epic 3: Write tests/test_vision_math.py to validate the "
    "root-finding math logic generated in Epic 1."
)


def _artifacts_ready() -> bool:
    return all(p.is_file() and p.stat().st_size > 40 for p in ARTIFACTS)


def _configure_env() -> dict[str, str]:
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
    env["DONNA_DEBUG_VISION"] = "1"
    env["DONNA_META_BROKER_LOG"] = str(SUITE_LOG)
    env["DONNA_META_BROKER_TIMEOUT_S"] = "900"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _ram_pct() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:
        return None


def _reset_manifest_for_suite() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps({"version": 1, "artifacts": []}, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_broker(log_fh: Any) -> dict[str, Any]:
    for k, v in _configure_env().items():
        os.environ[k] = v
    _reset_manifest_for_suite()

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
        try:
            log_fh.write(f"[TELEMETRY] {json.dumps(ev, default=str)}\n")
            log_fh.flush()
        except Exception:
            pass
        msg = str(ev.get("message") or "")
        low = msg.lower()
        if "starting epic" in low or "dispatch epic" in low:
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

    poller = threading.Thread(target=_ram_poll, name="Suite13Ram", daemon=True)
    poller.start()
    t0 = time.time()
    log_fh.write(f"[SUITE13] broker start t={datetime.now().isoformat()}\n")
    log_fh.write(f"[SUITE13] prompt={BROKER_PROMPT}\n")
    log_fh.flush()
    print("[suite13] launching isolated Meta-Broker…", flush=True)
    try:
        result = run_meta_broker_isolated(
            BROKER_PROMPT,
            on_event=_on_event,
            timeout_s=900.0,
            env_extra=_configure_env(),
        )
    finally:
        stop_ram.set()
    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "status": str((result or {}).get("status") or ""),
        "error": str((result or {}).get("error") or "")[:500],
        "artifacts_ready": _artifacts_ready(),
        "ram_peak": max(ram_samples) if ram_samples else None,
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
        str(ROOT / "tests" / "test_vision_math.py"),
        "-q",
        "--tb=short",
    ]
    print(f"[suite13] pytest: {' '.join(cmd)}", flush=True)
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
    with SUITE_LOG.open("w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"# Suite 13 Jarvis {datetime.now().isoformat()}\n")
        log_fh.flush()
        broker = _run_broker(log_fh)
        report = broker["report"]
        if not _artifacts_ready():
            log_fh.write("[SUITE13] artifacts missing after broker\n")
            print("[suite13] ARTIFACTS MISSING", flush=True)
            return 1
        code, pytest_out = _run_pytest()
        log_fh.write("\n# PYTEST\n" + pytest_out + "\n")
        log_fh.write(f"# pytest_exit={code}\n")
        report["pytest_exit"] = code
        log_fh.write("\n# FINAL\n" + json.dumps(report, indent=2) + "\n")
        print(pytest_out, flush=True)
        return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
