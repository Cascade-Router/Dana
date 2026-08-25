#!/usr/bin/env python3
"""Unified dev orchestrator for Dānā.

Starts, in order:
  1. The Cascade-Router gateway — a Docker Compose project that lives as a
     sibling of this repo (``../cascade-router``, i.e. a directory named
     ``cascade-router`` next to this ``DANA`` checkout). Launched with a
     ``DYNAMIC_ENV_PATH`` environment variable pointing at this repo's own
     ``.env``, which its ``docker-compose.yml`` interpolates into its
     service's ``env_file: - ${DYNAMIC_ENV_PATH}`` entry — the container
     gets this project's provider keys without either repo hardcoding the
     other's path. Waits for ``GET http://localhost:8080/health`` to return
     200 before continuing, since the FastAPI backend routes every cloud LLM
     call through it (see ``dana.core.model_provider.gateway_base_url``).
  2. The FastAPI backend, via ``scripts/launchers/launch_api_server.py``
     (``dana/api/server.py`` is just an importable ``app`` object with no
     ``__main__`` of its own — this launcher is the real entry point).
  3. The React/Vite frontend (``npm run dev`` in ``frontend/``).

Backend and frontend stdout/stderr are streamed to this terminal, each line
prefixed with its source. Ctrl+C tears everything down: both process TREES
are killed (not just the immediate child — on Windows, ``npm run dev``
spawns a cmd.exe -> npm.cmd -> node(vite) chain, and killing only the
top of that chain would orphan the actual dev server), then the gateway is
stopped with ``docker compose down``.

Usage:
    python start_dana.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CASCADE_ROUTER_DIR = (REPO_ROOT.parent / "cascade-router").resolve()
FRONTEND_DIR = REPO_ROOT / "frontend"
BACKEND_LAUNCHER = REPO_ROOT / "scripts" / "launchers" / "launch_api_server.py"
DANA_ENV_FILE = REPO_ROOT / ".env"

GATEWAY_HEALTH_URL = "http://localhost:8080/health"
GATEWAY_HEALTH_TIMEOUT_SEC = 60.0
GATEWAY_HEALTH_POLL_INTERVAL_SEC = 1.5

CHILD_STOP_GRACE_SEC = 10.0

IS_WINDOWS = sys.platform.startswith("win")


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def _stream_output(process: subprocess.Popen, tag: str) -> None:
    """Pipe a child process's combined stdout/stderr to ours, line-prefixed."""
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{tag}] {line.rstrip()}", flush=True)


def _start_streaming(process: subprocess.Popen, tag: str) -> threading.Thread:
    thread = threading.Thread(target=_stream_output, args=(process, tag), daemon=True)
    thread.start()
    return thread


# --- Step 1: Cascade-Router gateway (Docker) --------------------------------


def _require_dana_env_file() -> None:
    if not DANA_ENV_FILE.is_file():
        raise SystemExit(
            f"{DANA_ENV_FILE} not found — Cascade-Router needs it (via DYNAMIC_ENV_PATH) for "
            "provider API keys (GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ...)."
        )


def _compose_env() -> dict[str, str]:
    """Env for the `docker compose` subprocess itself: the parent's env plus
    DYNAMIC_ENV_PATH, which cascade-router's docker-compose.yml interpolates
    into its service's own `env_file:` entry (`env_file: - ${DYNAMIC_ENV_PATH}`).
    This is Compose's YAML-substitution mechanism, NOT `--env-file` — that
    flag only affects substitution too, it does not inject vars into a
    container, so it's not used here."""
    return {**os.environ, "DYNAMIC_ENV_PATH": str(DANA_ENV_FILE)}


def start_gateway() -> None:
    if not CASCADE_ROUTER_DIR.is_dir():
        raise SystemExit(
            f"Cascade-Router project not found at {CASCADE_ROUTER_DIR} — expected as a "
            "sibling directory of this repo (../cascade-router)."
        )
    _require_dana_env_file()
    _log("gateway", f"docker compose up -d  (in {CASCADE_ROUTER_DIR}, DYNAMIC_ENV_PATH={DANA_ENV_FILE})")
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=CASCADE_ROUTER_DIR,
            env=_compose_env(),
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("`docker` was not found on PATH — is Docker Desktop installed and running?") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"`docker compose up -d` failed (exit {exc.returncode}).") from exc
    _wait_for_gateway_health()


def _wait_for_gateway_health() -> None:
    _log("gateway", f"waiting for {GATEWAY_HEALTH_URL} ...")
    deadline = time.monotonic() + GATEWAY_HEALTH_TIMEOUT_SEC
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(GATEWAY_HEALTH_URL, timeout=3) as resp:
                if resp.status == 200:
                    _log("gateway", "healthy.")
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        time.sleep(GATEWAY_HEALTH_POLL_INTERVAL_SEC)
    raise SystemExit(
        f"Cascade-Router did not become healthy within {GATEWAY_HEALTH_TIMEOUT_SEC:.0f}s "
        f"(last error: {last_error}). Check `docker compose logs` in {CASCADE_ROUTER_DIR}."
    )


def stop_gateway() -> None:
    if not CASCADE_ROUTER_DIR.is_dir():
        return
    _log("gateway", "docker compose down ...")
    subprocess.run(["docker", "compose", "down"], cwd=CASCADE_ROUTER_DIR, env=_compose_env())


# --- Steps 2 & 3: backend + frontend subprocesses ---------------------------


def _popen_kwargs() -> dict:
    """Extra Popen kwargs so each child becomes the root of its own killable
    tree. POSIX: a new session/process group, so ``os.killpg`` below can stop
    the whole tree without touching this orchestrator's own group. Windows
    has no equivalent flag needed here — ``taskkill /T`` walks the OS's own
    parent/child PID records instead, see ``_kill_process_tree``."""
    if IS_WINDOWS:
        return {}
    return {"start_new_session": True}


def start_backend() -> subprocess.Popen:
    _log("backend", f"starting FastAPI/uvicorn via {BACKEND_LAUNCHER.name} ...")
    process = subprocess.Popen(
        [sys.executable, str(BACKEND_LAUNCHER)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **_popen_kwargs(),
    )
    _start_streaming(process, "backend")
    return process


def start_frontend() -> subprocess.Popen:
    _log("frontend", "starting npm run dev ...")
    if IS_WINDOWS:
        # npm ships as npm.cmd on Windows, which CreateProcess can't launch
        # directly without a command interpreter — shell=True routes it
        # through cmd.exe. Safe here: the command is a fixed literal, never
        # built from user input.
        command: str | list[str] = "npm run dev"
        shell = True
    else:
        command = ["npm", "run", "dev"]
        shell = False
    process = subprocess.Popen(
        command,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=shell,
        **_popen_kwargs(),
    )
    _start_streaming(process, "frontend")
    return process


def _kill_process_tree(process: subprocess.Popen, tag: str) -> None:
    """Terminate `process` and every descendant it spawned.

    `Popen.terminate()` alone only signals the immediate child — on Windows
    that's cmd.exe (or python.exe), not the node/vite grandchild actually
    doing the work, which would otherwise be left running as a zombie.
    """
    if process.poll() is not None:
        return  # already exited
    _log(tag, "stopping...")
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=CHILD_STOP_GRACE_SEC)
    except subprocess.TimeoutExpired:
        process.kill()


# --- Orchestration -----------------------------------------------------------


def main() -> int:
    backend: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None
    try:
        start_gateway()
        backend = start_backend()
        frontend = start_frontend()
        _log("orchestrator", "all services up — press Ctrl+C to stop.")
        while True:
            for name, proc in (("backend", backend), ("frontend", frontend)):
                code = proc.poll()
                if code is not None:
                    _log("orchestrator", f"{name} exited unexpectedly (code {code}) — shutting down.")
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        _log("orchestrator", "Ctrl+C received — shutting down...")
        return 0
    finally:
        if frontend is not None:
            _kill_process_tree(frontend, "frontend")
        if backend is not None:
            _kill_process_tree(backend, "backend")
        stop_gateway()


if __name__ == "__main__":
    raise SystemExit(main())
