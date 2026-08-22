"""System actuators: write local files, host shell, and jailed Python sandbox jobs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from dana.logging import log_debug
from dana.tools.powershell import DANGEROUS_COMMANDS_RE, SECURITY_VIOLATION_MSG
from dana.vault_service import windows_no_window_creationflags

# CREATE_SUSPENDED — primary thread starts suspended until ResumeThread.
_CREATE_SUSPENDED = 0x00000004

DEFAULT_TIMEOUT_SEC = 15
DEFAULT_PYTHON_TIMEOUT_SEC = 300
SANDBOX_JOB_TTL_S = 3600

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def write_to_file(filepath: str, content: str) -> str:
    """Create parent dirs, write UTF-8 text, return absolute path + byte size."""
    log_debug("Actuator", "executing tool=write_to_file")

    raw = (filepath or "").strip()
    if not raw:
        return "ERROR: empty filepath"

    try:
        path = Path(raw).expanduser()
        data = content if isinstance(content, str) else str(content or "")
        if path.suffix.lower() in {".py", ".pyi"} and re.search(
            r"(?i)<(?:!DOCTYPE\s+html|html|script|style)\b|```(?:html|css|javascript)\b",
            data,
        ):
            return (
                "ERROR: refused HTML/CSS/JS content for Python path; "
                "write valid Python only"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = data.encode("utf-8")
        path.write_bytes(encoded)
        abs_path = str(path.resolve())
        return f"OK: wrote {abs_path} ({len(encoded)} bytes)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: write_to_file failed: {exc}"


def execute_command(command: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> str:
    """Run a host command; PowerShell on Windows with sandbox safety checks.

    Applies ``DANGEROUS_COMMANDS_RE`` before spawn. On Windows, uses
    ``CREATE_NO_WINDOW`` and, when Job APIs are available, ``WindowsJob``
    (CREATE_SUSPENDED → assign → resume) with a hard timeout.
    """
    log_debug("Actuator", "executing tool=execute_command")

    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: empty command"

    if DANGEROUS_COMMANDS_RE.search(cmd):
        return SECURITY_VIOLATION_MSG

    try:
        timeout_sec = max(1, int(timeout))
    except (TypeError, ValueError):
        timeout_sec = DEFAULT_TIMEOUT_SEC

    try:
        if os.name == "nt":
            argv = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                cmd,
            ]
            return _execute_windows(argv, timeout_sec)
        return _execute_run(["/bin/sh", "-c", cmd], timeout_sec, windows=False)
    except FileNotFoundError:
        return (
            "ERROR: execute_command failed: shell executable not found "
            "(host has no PowerShell/sh on PATH)."
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execute_command timed out after {timeout_sec}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: execute_command failed: {exc}"


def _format_observation(returncode: int, stdout: str, stderr: str) -> str:
    out = (stdout or "").rstrip()
    err = (stderr or "").rstrip()
    return (
        f"returncode={int(returncode)}\n"
        f"stdout:\n{out or '(empty)'}\n"
        f"stderr:\n{err or '(empty)'}"
    )


def _execute_windows(argv: list[str], timeout_sec: int) -> str:
    """Prefer WindowsJob sandbox; fall back to subprocess.run + CREATE_NO_WINDOW."""
    from dana.tools.win32_sandbox import (
        JOB_APIS_AVAILABLE,
        WindowsJob,
        resume_suspended_process,
    )

    if not JOB_APIS_AVAILABLE:
        return _execute_run(argv, timeout_sec, windows=True)

    creationflags = windows_no_window_creationflags(_CREATE_SUSPENDED)
    proc = subprocess.Popen(  # noqa: S603
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    try:
        with WindowsJob() as job:
            if job.active:
                job.assign_pid(proc.pid)
            resume_suspended_process(proc.pid)
            stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return f"ERROR: execute_command timed out after {timeout_sec}s"
    except Exception:
        try:
            resume_suspended_process(proc.pid)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        raise

    returncode = int(proc.returncode if proc.returncode is not None else 0)
    return _format_observation(returncode, stdout or "", stderr or "")


def _execute_run(argv: list[str], timeout_sec: int, *, windows: bool) -> str:
    """subprocess.run path (non-Windows or missing Job APIs)."""
    run_kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_sec,
        "check": False,
    }
    if windows or os.name == "nt":
        run_kwargs["creationflags"] = windows_no_window_creationflags()

    completed = subprocess.run(argv, **run_kwargs)  # noqa: S603
    return _format_observation(
        int(completed.returncode),
        completed.stdout or "",
        completed.stderr or "",
    )


def _execution_jail() -> Path:
    from dana.paths import EXECUTION_JAIL_DIR

    return Path(EXECUTION_JAIL_DIR).resolve()


def _resolve_jailed_script(script_path: str) -> Path:
    """Resolve ``script_path`` under EXECUTION_JAIL_DIR; raise ValueError on escape."""
    jail = _execution_jail()
    raw = Path(str(script_path or "").strip()).expanduser()
    if not str(script_path or "").strip():
        raise ValueError("empty script_path")
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        # Tolerate leading execution_jail/ prefix.
        parts = raw.parts
        if parts and parts[0].lower() in {"execution_jail", "sandbox"}:
            raw = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        candidate = (jail / raw).resolve()
    try:
        candidate.relative_to(jail)
    except ValueError as exc:
        raise ValueError(
            f"path traversal blocked: {script_path!r} is outside execution_jail"
        ) from exc
    return candidate


def _jobs_dir() -> Path:
    path = _execution_jail() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_args(args: Any) -> list[str]:
    if args is None:
        return []
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:  # noqa: BLE001
            pass
        return [text]
    if isinstance(args, (list, tuple)):
        return [str(x) for x in args]
    return [str(args)]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _persist_job_outcome(payload: dict[str, Any]) -> None:
    try:
        from dana.memory.store import get_episodic_store

        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return
        get_episodic_store().add_fact(
            "task_outcome",
            f"sandbox_job_{job_id}",
            payload,
            confidence_score=1.0,
            ttl_seconds=SANDBOX_JOB_TTL_S,
        )
    except Exception:  # noqa: BLE001
        pass

    # Phase 5 — compress sandbox job logs into dense idle_compressed vault memory.
    try:
        from dana.memory.compressor import ingest_idle_compressed

        job_id = str(payload.get("job_id") or "").strip()
        log_path = str(payload.get("log_path") or "").strip()
        raw_parts: list[str] = [
            f"sandbox_job={job_id}",
            f"status={payload.get('status')}",
            f"exit_code={payload.get('exit_code')}",
            f"script={payload.get('script')}",
        ]
        if log_path:
            try:
                raw_parts.append(Path(log_path).read_text(encoding="utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                raw_parts.append(_log_tail(log_path, lines=80))
        ingest_idle_compressed(
            "\n".join(raw_parts),
            source="sandbox_job",
            topic=f"sandbox_job_{job_id}",
        )
    except Exception:  # noqa: BLE001
        pass


def _log_tail(log_path: str | Path, *, lines: int = 12) -> str:
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return "(log unavailable)"
    parts = text.splitlines()
    if not parts:
        return "(empty)"
    return "\n".join(parts[-max(1, int(lines)) :])


def execute_python_script(
    script_path: str,
    args: list[str] | None = None,
    timeout: int = DEFAULT_PYTHON_TIMEOUT_SEC,
    background: bool = False,
) -> str:
    """Run a Python script jailed to ``EXECUTION_JAIL_DIR``.

    Sync mode returns structured exit/duration/stdout/stderr.
    ``background=True`` starts a daemon job, streams output to
    ``execution_jail/jobs/<job_id>.log``, and returns the job handle immediately.
    """
    log_debug("Actuator", "executing tool=execute_python_script")

    try:
        script = _resolve_jailed_script(script_path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not script.is_file():
        return f"ERROR: script not found: {script}"

    try:
        timeout_sec = max(1, int(timeout))
    except (TypeError, ValueError):
        timeout_sec = DEFAULT_PYTHON_TIMEOUT_SEC

    argv = [sys.executable, str(script), *_coerce_args(args)]
    jail = _execution_jail()
    run_kwargs: dict[str, Any] = {
        "cwd": str(jail),
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout_sec,
        "check": False,
    }
    popen_kwargs: dict[str, Any] = {
        "cwd": str(jail),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        flags = windows_no_window_creationflags()
        run_kwargs["creationflags"] = flags
        popen_kwargs["creationflags"] = flags

    if not background:
        t0 = time.perf_counter()
        try:
            completed = subprocess.run(argv, **run_kwargs)  # noqa: S603
        except subprocess.TimeoutExpired as exc:
            return json.dumps(
                {
                    "status": "timeout",
                    "exit_code": -1,
                    "duration_s": round(time.perf_counter() - t0, 3),
                    "stdout": ((exc.stdout or "") if isinstance(exc.stdout, str) else "")[
                        -2000:
                    ],
                    "stderr": (
                        f"ERROR: execute_python_script timed out after {timeout_sec}s\n"
                        + (
                            (exc.stderr or "")
                            if isinstance(exc.stderr, str)
                            else ""
                        )
                    )[-2000:],
                    "cwd": str(jail),
                    "script": str(script),
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: execute_python_script failed: {exc}"

        return json.dumps(
            {
                "status": "completed" if int(completed.returncode or 0) == 0 else "failed",
                "exit_code": int(completed.returncode or 0),
                "duration_s": round(time.perf_counter() - t0, 3),
                "stdout": (completed.stdout or "")[-2000:],
                "stderr": (completed.stderr or "")[-2000:],
                "cwd": str(jail),
                "script": str(script),
            },
            ensure_ascii=False,
        )

    job_id = uuid.uuid4().hex[:12]
    log_path = _jobs_dir() / f"{job_id}.log"
    started_at = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "exit_code": None,
            "duration_s": None,
            "log_path": str(log_path),
            "script": str(script),
            "cwd": str(jail),
            "started_at": started_at,
        }

    def _bg() -> None:
        t0 = time.perf_counter()
        status = "failed"
        code = -1
        child_pid: int | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log_fh:
                log_fh.write(f"[start] job_id={job_id} script={script.name}\n")
                log_fh.flush()
                proc = subprocess.Popen(argv, **popen_kwargs)  # noqa: S603
                assert proc.stdout is not None
                child_pid = int(proc.pid) if proc.pid else None
                # Uncap this job from IdleMonitor's USER_ACTIVE ~20% throttle.
                if child_pid:
                    try:
                        from dana.middleware.idle_monitor import (
                            register_priority_override,
                        )

                        register_priority_override(child_pid, job_id=job_id)
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    for line in proc.stdout:
                        log_fh.write(line)
                        log_fh.flush()
                    proc.wait(timeout=timeout_sec)
                    code = int(proc.returncode if proc.returncode is not None else 0)
                    status = "completed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        proc.wait(timeout=5)
                    except Exception:  # noqa: BLE001
                        pass
                    code = -1
                    status = "timeout"
                    log_fh.write(
                        f"[error] timed out after {timeout_sec}s and was killed\n"
                    )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            code = -1
            try:
                with log_path.open("a", encoding="utf-8") as log_fh:
                    log_fh.write(f"[error] {exc}\n")
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                from dana.middleware.idle_monitor import unregister_priority_override

                unregister_priority_override(child_pid, job_id=job_id)
            except Exception:  # noqa: BLE001
                pass

        dur = round(time.perf_counter() - t0, 3)
        try:
            with log_path.open("a", encoding="utf-8") as log_fh:
                log_fh.write(
                    f"[end] status={status} exit_code={code} duration_s={dur}\n"
                )
        except Exception:  # noqa: BLE001
            pass

        payload = {
            "job_id": job_id,
            "status": status,
            "exit_code": code,
            "duration_s": dur,
            "log_path": str(log_path),
            "script": str(script),
            "cwd": str(jail),
            "finished_at": time.time(),
        }
        with _JOBS_LOCK:
            _JOBS[job_id].update(payload)
        _persist_job_outcome(payload)
        try:
            from dana.middleware.idle_monitor import queue_if_user_away

            queue_if_user_away(
                job_id=str(payload.get("job_id") or job_id),
                status=str(payload.get("status") or "completed"),
                summary=(
                    f"Sandbox script finished with status "
                    f"{payload.get('status')} (exit {payload.get('exit_code')})."
                ),
                kind="sandbox_job",
            )
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(
        target=_bg,
        name=f"sandbox-job-{job_id}",
        daemon=True,
    ).start()

    return json.dumps(
        {
            "job_id": job_id,
            "status": "running",
            "log_path": str(log_path),
            "script": str(script),
            "cwd": str(jail),
            "message": f"Background sandbox job started: {job_id}",
        },
        ensure_ascii=False,
    )


def get_sandbox_job_status(job_id: str | None = None) -> str:
    """Return status / exit / duration / log tail for one or all sandbox jobs."""
    log_debug("Actuator", "executing tool=get_sandbox_job_status")

    key = str(job_id or "").strip()
    with _JOBS_LOCK:
        if key:
            job = _JOBS.get(key)
            jobs = [dict(job)] if job else []
        else:
            jobs = [dict(j) for j in _JOBS.values()]

    if not jobs:
        # Fall back to episodic memory for finished jobs after process churn.
        if key:
            try:
                from dana.memory.store import get_episodic_store

                for fact in get_episodic_store().list_facts(include_expired=False):
                    if fact.get("category") != "task_outcome":
                        continue
                    if fact.get("key") != f"sandbox_job_{key}":
                        continue
                    raw = fact.get("value")
                    if isinstance(raw, str):
                        try:
                            parsed = json.loads(raw)
                        except Exception:  # noqa: BLE001
                            parsed = {"value": raw}
                    else:
                        parsed = raw if isinstance(raw, dict) else {"value": raw}
                    if isinstance(parsed, dict):
                        parsed = dict(parsed)
                        parsed.setdefault("job_id", key)
                        parsed["log_tail"] = _log_tail(
                            str(parsed.get("log_path") or "")
                        )
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        return "ERROR: no matching sandbox job"

    out: list[dict[str, Any]] = []
    for job in jobs:
        item = dict(job)
        item["log_tail"] = _log_tail(str(item.get("log_path") or ""))
        out.append(item)
    return json.dumps(out if not key else out[0], ensure_ascii=False, indent=2)
