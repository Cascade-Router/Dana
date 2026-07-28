"""Stage 9.3 — Git-based auto-updater for the native Dānā GUI.

Runs ``git fetch`` / ``git pull`` and ``pip install -r requirements.txt``
in ``PROJECT_ROOT``, then restarts via ``sys.executable -m donna.core_agent``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _repo_root() -> Path:
    try:
        from donna.paths import PROJECT_ROOT

        return Path(PROJECT_ROOT).resolve()
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Public alias — always the CAMGRASPER checkout root."""
    return _repo_root()


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict = {
        "cwd": str(cwd),
        "check": True,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_s,
        "shell": False,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    return subprocess.run(
        ["git", *args],
        **kwargs,
    )


def _run_pip_install(*, cwd: Path, timeout_s: float = 600.0) -> subprocess.CompletedProcess[str]:
    req = cwd / "requirements.txt"
    if not req.is_file():
        raise FileNotFoundError(f"requirements.txt missing under {cwd}")
    req_files = [req]
    cuda_req = cwd / "requirements-cuda.txt"
    if cuda_req.is_file():
        req_files.append(cuda_req)
    kwargs: dict = {
        "cwd": str(cwd),
        "check": True,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_s,
        "shell": False,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    cmd = [sys.executable, "-m", "pip", "install"]
    for path in req_files:
        cmd.extend(["-r", str(path)])
    return subprocess.run(cmd, **kwargs)


def local_head(*, cwd: Path | None = None) -> str:
    root = Path(cwd) if cwd is not None else _repo_root()
    proc = _run_git(["rev-parse", "HEAD"], cwd=root, timeout_s=30.0)
    return (proc.stdout or "").strip()


def upstream_head(*, cwd: Path | None = None) -> str:
    root = Path(cwd) if cwd is not None else _repo_root()
    proc = _run_git(["rev-parse", "@{u}"], cwd=root, timeout_s=30.0)
    return (proc.stdout or "").strip()


def check_for_updates(*, cwd: Path | None = None) -> bool:
    """Return True when remote tracking branch is ahead of local ``HEAD``.

    Runs ``git fetch`` then compares ``git rev-parse HEAD`` to ``@{u}``.
    Returns False when already current, offline, or no upstream is configured.
    """
    root = Path(cwd) if cwd is not None else _repo_root()
    try:
        _run_git(["fetch", "--quiet"], cwd=root, timeout_s=120.0)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        _log(f"git fetch failed: {_fmt_exc(exc)}")
        return False

    try:
        local = local_head(cwd=root)
        remote = upstream_head(cwd=root)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        _log(f"rev-parse failed (no upstream?): {_fmt_exc(exc)}")
        return False

    if not local or not remote:
        return False
    available = local != remote
    _log(
        f"check_for_updates local={local[:10]} remote={remote[:10]} "
        f"available={available}"
    )
    return available


@dataclass
class UpdateApplyResult:
    ok: bool
    message: str = ""
    stderr: str = ""


def apply_update_and_restart(
    *,
    cwd: Path | None = None,
    restart: bool = True,
) -> UpdateApplyResult:
    """``git pull`` + ``pip install -r requirements.txt``, then restart.

    On merge/pip failure returns ``UpdateApplyResult(ok=False, ...)`` and does
    **not** restart. On success launches a new process and calls ``sys.exit(0)``
    when ``restart=True`` (never returns in that case).
    """
    root = Path(cwd) if cwd is not None else _repo_root()
    try:
        pull = _run_git(["pull", "--ff-only"], cwd=root, timeout_s=180.0)
        _log(f"git pull ok: {(pull.stdout or pull.stderr or '')[:240]}")
    except subprocess.CalledProcessError as exc:
        err = _combine_stdio(exc)
        _log(f"git pull FAILED: {err}")
        return UpdateApplyResult(
            ok=False,
            message=(
                "Update Failed: Check terminal logs or resolve merge "
                "conflicts manually."
            ),
            stderr=err,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        err = _fmt_exc(exc)
        _log(f"git pull FAILED: {err}")
        return UpdateApplyResult(
            ok=False,
            message="Update Failed: git pull could not complete.",
            stderr=err,
        )

    try:
        pip = _run_pip_install(cwd=root)
        _log(f"pip install ok: {(pip.stdout or '')[-240:]}")
    except subprocess.CalledProcessError as exc:
        err = _combine_stdio(exc)
        _log(f"pip install FAILED: {err}")
        return UpdateApplyResult(
            ok=False,
            message=(
                "Update Failed: Check terminal logs or resolve merge "
                "conflicts manually."
            ),
            stderr=err,
        )
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        err = _fmt_exc(exc)
        _log(f"pip install FAILED: {err}")
        return UpdateApplyResult(
            ok=False,
            message="Update Failed: dependency install could not complete.",
            stderr=err,
        )

    if not restart:
        return UpdateApplyResult(ok=True, message="Update applied (restart skipped).")

    try:
        _spawn_new_instance(cwd=root)
    except OSError as exc:
        err = _fmt_exc(exc)
        _log(f"restart spawn FAILED: {err}")
        return UpdateApplyResult(
            ok=False,
            message="Update applied but restart failed — relaunch Dānā manually.",
            stderr=err,
        )

    _log("Update applied — exiting for restart")
    sys.exit(0)


def _spawn_new_instance(*, cwd: Path) -> None:
    import subprocess as sp

    cmd = [sys.executable, "-m", "donna.core_agent"]
    sp.Popen(  # noqa: S603 — intentional relaunch of our entrypoint
        cmd,
        cwd=str(cwd),
        close_fds=False,
        start_new_session=True,
    )


def _combine_stdio(exc: subprocess.CalledProcessError) -> str:
    out = (exc.stdout or "").strip()
    err = (exc.stderr or "").strip()
    parts = [p for p in (err, out) if p]
    return "\n".join(parts)[:4000] or f"exit={exc.returncode}"


def _fmt_exc(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return _combine_stdio(exc)
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"timeout after {exc.timeout}s"
    return f"{type(exc).__name__}: {exc}"


def _log(msg: str) -> None:
    try:
        from donna.logging import log

        log("Updater", msg)
    except Exception:  # noqa: BLE001
        print(f"[Updater] {msg}", flush=True)
