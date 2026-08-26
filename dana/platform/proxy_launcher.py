"""Native subprocess launcher for the Cascade-Router LLM gateway.

Used by every entry point that needs the gateway running — the production
targets (``app.py``, ``dana/api/server.py``) and the dev orchestrator
(``start_dana.py``, which used to shell out to Docker Compose for this).
Both the Windows desktop build and the Linux Hugging Face Space ship a
prebuilt Cascade-Router binary under ``bin/`` instead of requiring a Docker
daemon, so the same launcher works unmodified on every target.

``dana.core.model_provider.gateway_base_url()`` already defaults to
``http://localhost:8080/v1`` and the ``"gateway"`` provider target already
sends real chat-completion traffic there (see
``ModelProvider._resolve_openai_endpoint``'s ``"gateway"`` branch) — this
module only owns getting that process running and healthy before the rest
of the app starts routing calls to it.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = REPO_ROOT / "bin"

_HEALTH_TIMEOUT_SEC = 30.0
_HEALTH_POLL_INTERVAL_SEC = 0.5


def _binary_path() -> Path:
    system = platform.system()
    if system == "Windows":
        return BIN_DIR / "cascade-router.exe"
    if system == "Linux":
        return BIN_DIR / "cascade-router-linux"
    raise RuntimeError(f"no Cascade-Router binary available for platform {system!r}")


def _wait_for_health(port: int, process: subprocess.Popen) -> bool:
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + _HEALTH_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False  # exited early — bad port/config, missing deps, etc.
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_SEC)
    return False


def start_cascade_proxy(port: int = 8080) -> subprocess.Popen | None:
    """Launch the Cascade-Router gateway as a background subprocess and
    block until it reports healthy on ``/health`` (or up to
    ``_HEALTH_TIMEOUT_SEC``).

    Returns the ``Popen`` handle — pass it to ``stop_cascade_proxy`` on
    shutdown — or ``None`` if the platform binary isn't present, isn't
    launchable, or never became healthy. A missing gateway deliberately
    doesn't crash the whole app at startup: ``tool_calling_provider()``
    still works with ``"ollama"`` or an explicit ``DANA_CLOUD_PROVIDER``
    even without the gateway running.
    """
    try:
        binary = _binary_path()
    except RuntimeError as exc:
        print(f"[proxy_launcher] {exc} — Cascade-Router not started.", flush=True)
        return None

    if not binary.is_file():
        print(f"[proxy_launcher] {binary} not found — Cascade-Router not started.", flush=True)
        return None

    if platform.system() != "Windows":
        # Prebuilt Linux artifacts (e.g. unpacked from a release tarball into
        # the HF Space image) don't always keep the executable bit set.
        try:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        except OSError:
            pass

    print(f"[proxy_launcher] starting {binary} on port {port} ...", flush=True)
    try:
        process = subprocess.Popen([str(binary), "--port", str(port)], cwd=REPO_ROOT)
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
