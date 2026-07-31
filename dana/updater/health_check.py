"""Health-check & rollback gate for blue-green slot promotion.

``run_slot_health_check(candidate_slot_path)``:
  Pass → atomically switch active pointer; send ``hot_restart`` to sidecar IPC.
  Fail → abort, wipe candidate, append ``~/.dana/update_failures.log``, leave
  active slot untouched.

``verify_fn`` and ``ipc_client`` are injectable for offline unit tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dana.updater.slot_manager import SlotManager, dana_home, get_slot_manager

VerifyFn = Callable[[Path], bool]
HotRestartFn = Callable[[], dict[str, Any]]


def default_failure_log() -> Path:
    return dana_home() / "update_failures.log"


def default_verify_fn(candidate_slot_path: Path) -> bool:
    """Production verify: require VERSION marker + optional complex-task script.

    Prefer a lightweight local check so GPU / network is not required. When
    ``DANA_SLOT_FULL_VERIFY=1``, shell out to ``scripts/verify_complex_tasks.py``.
    """
    path = Path(candidate_slot_path)
    if not path.is_dir():
        return False
    version_file = path / "VERSION"
    if not version_file.is_file():
        return False
    ver = version_file.read_text(encoding="utf-8", errors="replace").strip()
    if not ver:
        return False
    # Sentinel for intentionally corrupt packages in tests / staging.
    if (path / "CORRUPT").is_file() or (path / "FAILED").is_file():
        return False

    if (os.environ.get("DANA_SLOT_FULL_VERIFY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _shell_verify_complex_tasks(path)
    return True


def _shell_verify_complex_tasks(candidate_slot_path: Path) -> bool:
    """Shell to ``scripts/verify_complex_tasks.py`` (production path)."""
    try:
        root = Path(__file__).resolve().parents[2]
    except Exception:  # noqa: BLE001
        root = Path.cwd()
    script = root / "scripts" / "verify_complex_tasks.py"
    if not script.is_file():
        return False
    env = os.environ.copy()
    env["DANA_CANDIDATE_SLOT"] = str(candidate_slot_path)
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@dataclass
class HealthCheckResult:
    ok: bool
    candidate_slot: str = ""
    active_slot: str = ""
    version: str = ""
    switched: bool = False
    wiped: bool = False
    hot_restart: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    status: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "candidate_slot": self.candidate_slot,
            "active_slot": self.active_slot,
            "version": self.version,
            "switched": self.switched,
            "wiped": self.wiped,
            "hot_restart": self.hot_restart,
            "error": self.error,
            "status": self.status,
        }


def append_update_failure(
    message: str,
    *,
    log_path: Path | None = None,
    candidate: Path | str | None = None,
) -> None:
    path = Path(log_path) if log_path else default_failure_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    if candidate is not None:
        line += f" candidate={candidate}"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _resolve_slot_name(manager: SlotManager, candidate: Path) -> str:
    cand = Path(candidate).resolve()
    for name in ("slot_a", "slot_b"):
        if manager.slot_path(name).resolve() == cand:
            return name
    # Fallback: directory name.
    return cand.name


def _invoke_hot_restart(ipc_client: Any | None) -> dict[str, Any]:
    if ipc_client is None:
        return {"ok": True, "skipped": True, "reason": "no ipc client"}
    if callable(ipc_client) and not hasattr(ipc_client, "hot_restart"):
        try:
            result = ipc_client()
            return dict(result) if isinstance(result, dict) else {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        result = ipc_client.hot_restart()
        return dict(result) if isinstance(result, dict) else {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: slot already switched; daemon may be offline in headless.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "degraded": True}


def run_slot_health_check(
    candidate_slot_path: Path | str,
    *,
    verify_fn: VerifyFn | None = None,
    ipc_client: Any | None = None,
    slot_manager: SlotManager | None = None,
    failure_log: Path | str | None = None,
    version: str | None = None,
) -> HealthCheckResult:
    """Verify candidate slot then promote or roll back.

    Parameters
    ----------
    candidate_slot_path:
        Path to the inactive slot that received the OTA unpack.
    verify_fn:
        Injectable predicate. Defaults to :func:`default_verify_fn` (may shell
        to ``scripts/verify_complex_tasks.py`` when ``DANA_SLOT_FULL_VERIFY=1``).
    ipc_client:
        Object with ``hot_restart()`` or a zero-arg callable. Optional.
    slot_manager:
        Injectable :class:`SlotManager` (tests).
    failure_log:
        Override path for ``update_failures.log``.
    version:
        Version string to record on successful switch.
    """
    candidate = Path(candidate_slot_path)
    manager = slot_manager or get_slot_manager()
    verify = verify_fn or default_verify_fn
    fail_log = Path(failure_log) if failure_log else default_failure_log()
    active_before = manager.active_slot_name()
    candidate_name = _resolve_slot_name(manager, candidate)
    ver = (version or "").strip().lstrip("vV")
    if not ver:
        ver_file = candidate / "VERSION"
        if ver_file.is_file():
            ver = ver_file.read_text(encoding="utf-8", errors="replace").strip().lstrip("vV")

    result = HealthCheckResult(
        ok=False,
        candidate_slot=candidate_name,
        active_slot=active_before,
        version=ver,
        status="checking",
    )

    # Never promote the already-active slot via this gate.
    if candidate_name == active_before:
        result.error = "candidate is the active slot — refusing self-promote"
        result.status = "failed"
        append_update_failure(result.error, log_path=fail_log, candidate=candidate)
        return result

    healthy = False
    verify_error = ""
    try:
        healthy = bool(verify(candidate))
    except Exception as exc:  # noqa: BLE001
        healthy = False
        verify_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    if not healthy:
        result.status = "failed"
        result.error = verify_error or "health check failed"
        try:
            manager.wipe_slot(candidate_name)
            result.wiped = True
        except Exception as exc:  # noqa: BLE001
            result.error += f"; wipe failed: {type(exc).__name__}: {exc}"
        append_update_failure(
            f"ROLLBACK {result.error}",
            log_path=fail_log,
            candidate=candidate,
        )
        # Active pointer must remain untouched.
        result.active_slot = manager.active_slot_name()
        return result

    # Promote.
    try:
        manager.switch_active(candidate_name, version=ver or None)
        result.switched = True
        result.active_slot = manager.active_slot_name()
        result.ok = True
        result.status = "healthy"
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = f"switch failed: {type(exc).__name__}: {exc}"
        try:
            manager.wipe_slot(candidate_name)
            result.wiped = True
        except Exception:  # noqa: BLE001
            pass
        append_update_failure(result.error, log_path=fail_log, candidate=candidate)
        result.active_slot = manager.active_slot_name()
        return result

    result.hot_restart = _invoke_hot_restart(ipc_client)
    return result
