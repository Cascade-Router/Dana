"""Process watchdog + session persistence for hot-update engine swaps.

Monitors the Agent Engine daemon PID. On hot-update restart:
  1. Serialize conversation memory + task state to session_state.json
  2. Gracefully terminate the current engine PID
  3. Launch the updated engine
  4. Restore session on the new process (engine loads the file at boot)

All paths / PIDs / launch hooks are injectable for offline tests.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_SESSION_NAME = "session_state.json"


def default_session_path() -> Path:
    """``~/.dana/session_state.json`` (overridable via ``DANA_SESSION_STATE``)."""
    env = os.environ.get("DANA_SESSION_STATE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".dana" / DEFAULT_SESSION_NAME


def default_pid_path() -> Path:
    env = os.environ.get("DANA_ENGINE_PID_FILE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".dana" / "engine.pid"


LaunchFn = Callable[[], int | subprocess.Popen[Any]]
KillFn = Callable[[int], None]


class ProcessWatchdog:
    """Monitor / restart the Agent Engine daemon with session continuity."""

    def __init__(
        self,
        *,
        session_path: Path | str | None = None,
        pid_path: Path | str | None = None,
        pid: int | None = None,
        launch_fn: LaunchFn | None = None,
        kill_fn: KillFn | None = None,
        poll_interval_s: float = 0.05,
        ready_timeout_s: float = 5.0,
    ) -> None:
        self.session_path = Path(session_path) if session_path else default_session_path()
        self.pid_path = Path(pid_path) if pid_path else default_pid_path()
        self._pid = int(pid) if pid is not None else self._read_pid_file()
        self._launch_fn = launch_fn
        self._kill_fn = kill_fn or self._default_kill
        self.poll_interval_s = float(poll_interval_s)
        self.ready_timeout_s = float(ready_timeout_s)
        self._child: subprocess.Popen[Any] | None = None

    @property
    def pid(self) -> int | None:
        return self._pid

    def set_pid(self, pid: int | None) -> None:
        self._pid = int(pid) if pid is not None else None
        if self._pid is not None:
            self._write_pid_file(self._pid)

    def is_alive(self, pid: int | None = None) -> bool:
        target = self._pid if pid is None else int(pid)
        if target is None or target <= 0:
            return False
        if os.name == "nt":
            try:
                import psutil  # type: ignore[import-untyped]

                return bool(psutil.pid_exists(target))
            except Exception:  # noqa: BLE001
                pass
        try:
            os.kill(target, 0)
            return True
        except (OSError, SystemError):
            return False
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def save_session(
        cls,
        state: dict[str, Any],
        path: Path | str | None = None,
    ) -> Path:
        dest = Path(path) if path else default_session_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "conversation": list(state.get("conversation") or []),
            "task_state": dict(state.get("task_state") or {}),
            "meta": dict(state.get("meta") or {}),
            "saved_at": time.time(),
        }
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(dest)
        return dest

    @classmethod
    def load_session(cls, path: Path | str | None = None) -> dict[str, Any]:
        src = Path(path) if path else default_session_path()
        if not src.is_file():
            return {}
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            "conversation": list(data.get("conversation") or []),
            "task_state": dict(data.get("task_state") or {}),
            "meta": dict(data.get("meta") or {}),
            "saved_at": data.get("saved_at"),
        }

    def flush_session(self, state: dict[str, Any]) -> Path:
        return self.save_session(state, self.session_path)

    def terminate_engine(self, pid: int | None = None, *, timeout_s: float = 2.0) -> bool:
        """Gracefully terminate engine PID. Returns True if process is gone."""
        target = self._pid if pid is None else int(pid)
        if target is None:
            return True
        if target == os.getpid():
            # Never suicide the watchdog / UI / test host process.
            self._pid = None
            return True
        if not self.is_alive(target):
            self._pid = None
            return True
        try:
            self._kill_fn(target)
        except Exception:  # noqa: BLE001
            pass
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if not self.is_alive(target):
                self._pid = None
                return True
            time.sleep(self.poll_interval_s)
        # Force kill fallback.
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(target), "/F"],
                    check=False,
                    capture_output=True,
                )
            else:
                os.kill(target, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        gone = not self.is_alive(target)
        if gone:
            self._pid = None
        return gone

    def launch_engine(self) -> int:
        """Launch updated engine via injectable ``launch_fn`` (or default module)."""
        if self._launch_fn is None:
            self._launch_fn = self._default_launch
        result = self._launch_fn()
        if isinstance(result, subprocess.Popen):
            self._child = result
            new_pid = int(result.pid)
        else:
            new_pid = int(result)
        self.set_pid(new_pid)
        return new_pid

    def hot_update_restart(
        self,
        state: dict[str, Any] | None = None,
        *,
        wait_dead: bool = True,
    ) -> dict[str, Any]:
        """Serialize state, kill current engine, launch replacement, restore path."""
        session = dict(state or self.load_session(self.session_path))
        path = self.flush_session(session)
        old_pid = self._pid
        if wait_dead and old_pid is not None:
            self.terminate_engine(old_pid)
        new_pid = self.launch_engine()
        restored = self.load_session(self.session_path)
        return {
            "ok": True,
            "session_path": str(path),
            "old_pid": old_pid,
            "new_pid": new_pid,
            "restored_turns": len(restored.get("conversation") or []),
            "session": restored,
        }

    def _read_pid_file(self) -> int | None:
        try:
            if not self.pid_path.is_file():
                return None
            text = self.pid_path.read_text(encoding="utf-8").strip()
            return int(text) if text else None
        except (OSError, ValueError):
            return None

    def _write_pid_file(self, pid: int) -> None:
        try:
            self.pid_path.parent.mkdir(parents=True, exist_ok=True)
            self.pid_path.write_text(f"{int(pid)}\n", encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _default_kill(pid: int) -> None:
        if os.name == "nt":
            # SIGTERM analogue — taskkill without /F first.
            subprocess.run(
                ["taskkill", "/PID", str(pid)],
                check=False,
                capture_output=True,
            )
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def _default_launch(self) -> subprocess.Popen[Any]:
        cmd = [
            sys.executable,
            "-m",
            "dana.daemon",
            "--session",
            str(self.session_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict[str, Any] = {
            "stdout": None,  # inherit — surface daemon logs to parent
            "stderr": None,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = int(
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )
        return subprocess.Popen(cmd, **kwargs)
