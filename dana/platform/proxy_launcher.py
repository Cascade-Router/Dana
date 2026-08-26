"""Native subprocess launcher for the Cascade-Router LLM gateway.

Used by every entry point that needs the gateway running — the production
targets (``app.py``, ``dana/api/server.py``) and the dev orchestrator
(``start_dana.py``, which used to shell out to Docker Compose for this).

Cascade-Router (``../cascade-router``, a sibling checkout of this repo) is
a compiled C++ proxy (``cpp_core/src/proxy_server.cpp``), built via CMake —
either ``cpp_core/build_and_run.bat`` on Windows or ``cpp_core/Dockerfile``'s
build stage on Linux — NOT a Python service, despite that sibling repo also
holding Python scripts under ``src/`` (``train_router.py``, ``ingest.py``,
``quantize_onnx.py``, ...); those are offline training/data-pipeline tooling
that produces ``router_weights.json``/ONNX artifacts, not a server. This
module launches whichever platform build of the compiled binary is already
present under ``cpp_core/build/`` — it does not build it.

``proxy_server.cpp`` hardcodes its listen port at 8000 (``kListenPort``) —
there is no CLI flag or env var to change it, and ``argv[1]`` is already
reserved for the ONNX model path. Docker Compose's old ``8080:8000`` port
mapping is what previously made 8080 the externally visible port; running
the binary natively means that mapping is gone, so this launcher's default
``port`` is 8000 to match the binary's real listen port.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CASCADE_ROUTER_DIR = (REPO_ROOT.parent / "cascade-router").resolve()
CPP_CORE_DIR = CASCADE_ROUTER_DIR / "cpp_core"

# Matches proxy_server.cpp's own kDefaultModelPath — passed explicitly
# anyway (as build_and_run.bat and the Dockerfile's ENTRYPOINT both do) so
# a future default change there doesn't silently change what gets loaded
# here. Relative to CPP_CORE_DIR, the cwd the binary is launched from.
_MODEL_ARG = "../models/all-MiniLM-L6-v2-int8.onnx"

_HEALTH_TIMEOUT_SEC = 30.0
_HEALTH_POLL_INTERVAL_SEC = 0.5


def _binary_path() -> Path:
    """Where CMake actually puts ``proxy_server`` for each platform's build
    (see ``cpp_core/build_and_run.bat`` and ``cpp_core/Dockerfile``):
    Windows's Visual Studio generator is multi-config
    (``build/Release/...``); Linux's Makefile/Ninja build is single-config
    and puts the binary straight in ``build/``."""
    system = platform.system()
    if system == "Windows":
        return CPP_CORE_DIR / "build" / "Release" / "proxy_server.exe"
    if system == "Linux":
        return CPP_CORE_DIR / "build" / "proxy_server"
    raise RuntimeError(f"no Cascade-Router build convention known for platform {system!r}")


def _wait_for_health(port: int, process: subprocess.Popen) -> bool:
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + _HEALTH_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False  # exited early — bad model path, missing deps, etc.
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_SEC)
    return False


def start_cascade_proxy(port: int = 8000) -> subprocess.Popen | None:
    """Launch the Cascade-Router gateway (compiled ``proxy_server`` binary)
    as a background subprocess and block until it reports healthy on
    ``/health`` (or up to ``_HEALTH_TIMEOUT_SEC``).

    ``port`` only controls where THIS function health-checks — the binary
    itself always binds 8000 (see module docstring), so passing anything
    else here just makes the health check look in the wrong place.

    Returns the ``Popen`` handle — pass it to ``stop_cascade_proxy`` on
    shutdown — or ``None`` if ``../cascade-router`` isn't checked out as a
    sibling repo, hasn't been built yet, or never became healthy. A missing
    gateway deliberately doesn't crash the whole app at startup:
    ``tool_calling_provider()`` still works with ``"ollama"`` or an explicit
    ``DANA_CLOUD_PROVIDER`` even without the gateway running.
    """
    try:
        binary = _binary_path()
    except RuntimeError as exc:
        print(f"[proxy_launcher] {exc} — Cascade-Router not started.", flush=True)
        return None

    if not binary.is_file():
        print(
            f"[proxy_launcher] {binary} not found — build it first "
            "(cpp_core/build_and_run.bat on Windows, or cpp_core/Dockerfile's "
            "build stage on Linux). Cascade-Router not started.",
            flush=True,
        )
        return None

    if platform.system() != "Windows":
        # A locally-built Linux binary usually keeps its executable bit,
        # but one copied out of an image layer (e.g. into an HF Space)
        # doesn't always.
        try:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        except OSError:
            pass

    print(f"[proxy_launcher] starting {binary} ...", flush=True)
    try:
        process = subprocess.Popen([str(binary), _MODEL_ARG], cwd=CPP_CORE_DIR)
    except OSError as exc:
        print(f"[proxy_launcher] failed to launch {binary}: {exc}", flush=True)
        return None

    if _wait_for_health(port, process):
        print(f"[proxy_launcher] Cascade-Router healthy on port {port}.", flush=True)
        return process

    print(f"[proxy_launcher] Cascade-Router did not become healthy within {_HEALTH_TIMEOUT_SEC:.0f}s.", flush=True)
    stop_cascade_proxy(process)
    return None


def stop_cascade_proxy(process: subprocess.Popen | None) -> None:
    """Terminate a handle returned by ``start_cascade_proxy``.

    Safe to call with ``None`` (nothing was started) or an already-exited
    process.
    """
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()


__all__ = ("start_cascade_proxy", "stop_cascade_proxy")
