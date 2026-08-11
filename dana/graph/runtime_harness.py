"""Closed-loop runtime validation harness for Meta-Broker epics."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

from dana.system_health import check_system_health, kill_process_tree

# Prefer fast, targeted checks — never a repo-wide pytest by default.
DEFAULT_VALIDATION_COMMAND = "python -m compileall .dana_scratch"
DEFAULT_HARNESS_TIMEOUT_S = 15.0

_FILE_TOKEN_RE = re.compile(
    r"([\w./\\-]+\.(?:py|pyi|md|txt|json))\b",
    re.I,
)
_TRIAGE_TEST_RE = re.compile(r"\bTEST\b", re.I)
_TRIAGE_CODE_RE = re.compile(r"\bCODE\b", re.I)

# Paths that must never be deleted by fail-fast rollback.
_ROLLBACK_PROTECTED_PREFIXES = (
    "dana/",
    "dana_security/",
    "dana_security/",
    "website/",
    "legacy/",
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
)

# run_key → {path: {"existed": bool, "backup": str|None}}
_EPIC_FILE_TRACKER: dict[str, dict[str, dict[str, Any]]] = {}


def _norm_rel(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _is_protected_rel(rel: str) -> bool:
    low = _norm_rel(rel).lower()
    if not low or low.startswith(".."):
        return True
    return any(low.startswith(p) for p in _ROLLBACK_PROTECTED_PREFIXES)


def begin_epic_artifact_tracking(
    workspace_path: str,
    file_paths: list[str] | None,
    *,
    run_key: str = "default",
) -> list[str]:
    """Snapshot existence (and content backup) for epic target files."""
    root = Path(str(workspace_path or ".")).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass
    key = str(run_key or "default")
    bucket = _EPIC_FILE_TRACKER.setdefault(key, {})
    tracked: list[str] = []
    for raw in file_paths or []:
        rel = _norm_rel(raw)
        if not rel or _is_protected_rel(rel):
            continue
        path = root / rel
        existed = path.is_file()
        backup: str | None = None
        if existed:
            try:
                backup = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                backup = None
        bucket[rel] = {"existed": existed, "backup": backup}
        tracked.append(rel)
    return tracked


def register_generated_paths(
    file_paths: list[str] | None,
    *,
    run_key: str = "default",
    workspace_path: str | None = None,
) -> None:
    """Record additional paths produced during the epic (e.g. after staging)."""
    if workspace_path:
        begin_epic_artifact_tracking(
            workspace_path, list(file_paths or []), run_key=run_key
        )
        return
    key = str(run_key or "default")
    bucket = _EPIC_FILE_TRACKER.setdefault(key, {})
    for raw in file_paths or []:
        rel = _norm_rel(raw)
        if not rel or _is_protected_rel(rel):
            continue
        bucket.setdefault(rel, {"existed": False, "backup": None})


def commit_epic_artifact_tracking(run_key: str = "default") -> None:
    """Accept the current on-disk state for ``run_key`` — discard its rollback
    snapshot without touching any files. Mirrors what the runtime harness node
    does inline on validation success; exposed here so other sandboxed
    callers (e.g. MCP tool execution) can commit the same way.
    """
    _EPIC_FILE_TRACKER.pop(str(run_key or "default"), None)


def rollback_scratch_workspace(
    workspace_path: str,
    file_paths: list[str] | None = None,
    *,
    run_key: str = "default",
    clear_tracker: bool = True,
) -> dict[str, Any]:
    """Delete or restore unvalidated artifacts after an aborted epic.

    - Files that did not exist at tracking start are deleted.
    - Files that existed are restored from the text backup when available.
    - Protected package trees (``dana/``, ``website/``, …) are never touched.
    """
    root = Path(str(workspace_path or ".")).expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass
    key = str(run_key or "default")
    bucket = dict(_EPIC_FILE_TRACKER.get(key) or {})
    for raw in file_paths or []:
        rel = _norm_rel(raw)
        if rel and not _is_protected_rel(rel):
            bucket.setdefault(rel, {"existed": False, "backup": None})

    deleted: list[str] = []
    restored: list[str] = []
    skipped: list[str] = []
    for rel, meta in bucket.items():
        if _is_protected_rel(rel):
            skipped.append(rel)
            continue
        path = root / rel
        try:
            if meta.get("existed") and meta.get("backup") is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(meta.get("backup") or ""), encoding="utf-8")
                restored.append(rel)
            elif path.is_file():
                path.unlink()
                deleted.append(rel)
            # Also drop scratch mirrors when present.
            scratch = root / ".dana_scratch" / rel
            if scratch.is_file() and not meta.get("existed"):
                try:
                    scratch.unlink()
                except OSError:
                    pass
        except OSError as exc:
            skipped.append(f"{rel}({exc})")

    if clear_tracker:
        _EPIC_FILE_TRACKER.pop(key, None)

    summary = {
        "deleted": deleted,
        "restored": restored,
        "skipped": skipped,
        "workspace_path": str(root),
    }
    print(
        f"[RuntimeHarness] rollback_scratch_workspace "
        f"deleted={deleted} restored={restored} skipped={skipped[:5]}",
        flush=True,
    )
    return summary



def _scratch_pythonpath_entries(project_root: Path) -> list[str]:
    """Paths so validation can import staged drafts under ``.dana_scratch``."""
    entries: list[str] = [str(project_root)]
    scratch = project_root / ".dana_scratch"
    if scratch.is_dir():
        entries.append(str(scratch.resolve()))
        try:
            for child in scratch.iterdir():
                if child.is_dir():
                    entries.append(str(child.resolve()))
        except OSError:
            pass
    return entries


def _read_text_capped(path: Path, *, limit: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]..."
    return text


def _guess_test_and_impl_paths(
    *,
    command: str,
    epic_goal: str,
    workspace: Path,
) -> tuple[str | None, str | None]:
    """Best-effort pair of (test_filepath, impl_filepath) for repair triage."""
    mentioned = [
        m.group(1).replace("\\", "/")
        for m in _FILE_TOKEN_RE.finditer(f"{command}\n{epic_goal}")
    ]
    test_path: str | None = None
    impl_path: str | None = None
    for rel in mentioned:
        low = rel.lower()
        name = rel.rsplit("/", 1)[-1].lower()
        if "test_" in name or "/tests/" in f"/{low}" or low.startswith("tests/"):
            if test_path is None:
                test_path = rel
        elif low.endswith(".py") and impl_path is None:
            impl_path = rel
    m = re.search(
        r"(?:pytest\s+)(?P<path>[\w./\\-]*test_[\w./\\-]+\.py)",
        command or "",
        re.I,
    )
    if m and test_path is None:
        test_path = m.group("path").replace("\\", "/")
    if test_path and impl_path is None:
        stem = Path(test_path).name
        if stem.startswith("test_"):
            candidate = stem[len("test_") :]
            for rel in (candidate, f"dana/{candidate}"):
                if (workspace / rel).is_file():
                    impl_path = rel.replace("\\", "/")
                    break
    return test_path, impl_path


def triage_bidirectional_repair(
    *,
    test_code: str,
    impl_code: str,
    error: str,
    test_filepath: str | None = None,
    impl_filepath: str | None = None,
) -> dict[str, Any]:
    """Lightweight LLM triage: blame flawed TEST vs wrong CODE (implementation).

    Returns keys: ``repair_triage`` (TEST|CODE), ``repair_target_filepath``,
    ``repair_triage_raw``.
    """
    test_fp = (test_filepath or "").strip() or "tests/test_unknown.py"
    impl_fp = (impl_filepath or "").strip() or "implementation.py"
    prompt = (
        "The test failed. Here is the test code:\n"
        f"```python\n{(test_code or '')[:4000]}\n```\n"
        "Here is the implementation:\n"
        f"```python\n{(impl_code or '')[:4000]}\n```\n"
        f"Here is the error:\n```\n{(error or '')[:2000]}\n```\n"
        "Respond with ONLY the word 'TEST' if the test logic is flawed, or "
        "'CODE' if the implementation is wrong."
    )
    raw = ""
    try:
        from dana.core.constants import OLLAMA_MODEL
        from dana.core.agent_loop import ask_ollama_messages

        raw = ask_ollama_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a senior test engineer performing repair triage. "
                        "Reply with exactly one token: TEST or CODE."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=OLLAMA_MODEL,
            num_predict=16,
            temperature=0.0,
        )
        raw = raw if isinstance(raw, str) else str(raw or "")
    except Exception as exc:  # noqa: BLE001
        raw = f"CODE (triage_fallback:{type(exc).__name__})"

    decision: Literal["TEST", "CODE"] = "CODE"
    head = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    if _TRIAGE_TEST_RE.search(head) and not _TRIAGE_CODE_RE.search(head):
        decision = "TEST"
    elif _TRIAGE_CODE_RE.search(raw or ""):
        decision = "CODE"
    elif _TRIAGE_TEST_RE.search(raw or ""):
        decision = "TEST"

    target = test_fp if decision == "TEST" else impl_fp
    print(
        f"[RuntimeHarness] Bidirectional Repair Triage → {decision} "
        f"target={target!r} raw={(raw or '')[:80]!r}",
        flush=True,
    )
    return {
        "repair_triage": decision,
        "repair_target_filepath": target,
        "repair_triage_raw": (raw or "").strip()[:500],
        "repair_test_filepath": test_fp,
        "repair_impl_filepath": impl_fp,
    }


def run_validation_harness(
    workspace_path: str,
    command: str,
    *,
    timeout_s: float = DEFAULT_HARNESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute a build/test command and return a structured result dict.

    Parameters
    ----------
    workspace_path:
        Working directory for the subprocess (must exist).
    command:
        Shell-like command string, e.g. ``python -m pytest tests/test_x.py -q``.
    timeout_s:
        Soft wall-clock limit for the child process (default 15s).
    """
    cwd = Path(str(workspace_path or ".")).expanduser()
    try:
        cwd = cwd.resolve()
    except OSError:
        cwd = Path(str(workspace_path or "."))

    cmd = str(command or "").strip()
    if not cmd:
        return {
            "success": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": "ERROR: validation command is empty",
        }
    if not cwd.is_dir():
        return {
            "success": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": f"ERROR: workspace_path is not a directory: {cwd}",
        }

    try:
        import sys

        if sys.platform == "win32":
            popen_args: str | list[str] = cmd
            use_shell = True
        else:
            popen_args = shlex.split(cmd, posix=True)
            use_shell = False
            if not popen_args:
                return {
                    "success": False,
                    "exit_code": 2,
                    "stdout": "",
                    "stderr": "ERROR: validation command parsed empty",
                }
    except ValueError as exc:
        return {
            "success": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": f"ERROR: invalid command string: {exc}",
        }

    env = os.environ.copy()
    try:
        from dana.paths import PROJECT_ROOT

        root = Path(PROJECT_ROOT).resolve()
    except Exception:  # noqa: BLE001
        root = cwd
    path_bits = _scratch_pythonpath_entries(root)
    prev = (env.get("PYTHONPATH") or "").strip()
    if prev:
        path_bits.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(path_bits)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        check_system_health()
    except SystemError as health_exc:
        return {
            "success": False,
            "exit_code": 137,
            "stdout": "",
            "stderr": str(health_exc),
        }

    wall = max(1.0, float(timeout_s))
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            popen_args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=use_shell,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired as timed_out:
            # Nuke the whole tree — pytest/compile children must not linger.
            partial_out = ""
            partial_err = ""
            try:
                if timed_out.stdout:
                    partial_out = (
                        timed_out.stdout
                        if isinstance(timed_out.stdout, str)
                        else timed_out.stdout.decode("utf-8", errors="replace")
                    )
                if timed_out.stderr:
                    partial_err = (
                        timed_out.stderr
                        if isinstance(timed_out.stderr, str)
                        else timed_out.stderr.decode("utf-8", errors="replace")
                    )
            except Exception:  # noqa: BLE001
                pass
            if proc.pid:
                print(
                    f"[RuntimeHarness] TIMEOUT after {wall}s — "
                    f"kill_process_tree(pid={proc.pid})",
                    flush=True,
                )
                kill_process_tree(int(proc.pid))
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except Exception:  # noqa: BLE001
                stdout, stderr = "", ""
            stdout = (stdout or "") or partial_out
            stderr = (stderr or "") or partial_err
            timeout_note = (
                f"ERROR: validation timed out after {timeout_s}s "
                "(process tree killed). Self-remediate: avoid infinite loops "
                "(e.g. BFS must track a visited set); keep tests fast."
            )
            if stderr:
                combined_err = f"{timeout_note}\n--- partial stderr ---\n{stderr}"
            else:
                combined_err = timeout_note
            return {
                "success": False,
                "exit_code": 124,
                "stdout": stdout or "",
                "stderr": combined_err,
                "timed_out": True,
            }
        code = int(proc.returncode if proc.returncode is not None else 1)
        return {
            "success": code == 0,
            "exit_code": code,
            "stdout": stdout or "",
            "stderr": stderr or "",
        }
    except OSError as exc:
        return {
            "success": False,
            "exit_code": 127,
            "stdout": "",
            "stderr": f"ERROR: failed to spawn validation process: {exc}",
        }
    finally:
        if proc is not None and proc.poll() is None:
            try:
                kill_process_tree(int(proc.pid))
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass


def make_runtime_harness_node(
    *,
    default_command: str = DEFAULT_VALIDATION_COMMAND,
    timeout_s: float = DEFAULT_HARNESS_TIMEOUT_S,
    harness_fn: Any | None = None,
):
    """LangGraph node: run validation for the active epic and store feedback."""

    runner = harness_fn or run_validation_harness

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        from dana.paths import PROJECT_ROOT
        from dana.system_health import check_system_health

        try:
            health = check_system_health()
            print(
                f"[RuntimeHarness] health ram={health['ram_percent']:.1f}% "
                f"cpu={health['cpu_percent']:.1f}%",
                flush=True,
            )
        except SystemError as health_exc:
            print(f"[RuntimeHarness] {health_exc}", flush=True)
            return {
                "runtime_feedback": {
                    "success": False,
                    "exit_code": 137,
                    "stdout": "",
                    "stderr": str(health_exc),
                },
                "broker_phase": "feedback",
                "status": "failed",
                "error": str(health_exc),
            }

        epics = list(state.get("epics") or [])
        idx = int(state.get("active_epic_index") or 0)
        epic = epics[idx] if 0 <= idx < len(epics) else {}
        workspace = str(
            state.get("workspace_path")
            or (epic or {}).get("workspace_path")
            or PROJECT_ROOT
        )
        command = str((epic or {}).get("validation_command") or "").strip()
        if not command:
            command = str(state.get("validation_command") or "").strip()
        if not command:
            command = str(default_command or DEFAULT_VALIDATION_COMMAND).strip()
        lowered = command.lower().replace("\\", "/")
        if lowered in {"pytest", "pytest -q", "python -m pytest", "python -m pytest -q"}:
            print(
                f"[RuntimeHarness] refusing global pytest {command!r}; "
                f"using {DEFAULT_VALIDATION_COMMAND!r}",
                flush=True,
            )
            command = DEFAULT_VALIDATION_COMMAND

        # Track epic target files so fail-fast abort can roll them back.
        # Prefer the pre-dispatch snapshot; only seed if missing.
        epic_id = (epic or {}).get("epic_id")
        run_key = f"epic-{epic_id}"
        epic_files = [
            m.group(1).replace("\\", "/")
            for m in _FILE_TOKEN_RE.finditer(
                f"{(epic or {}).get('goal') or ''}\n"
                f"{(epic or {}).get('validation_command') or ''}\n{command}"
            )
        ]
        if run_key not in _EPIC_FILE_TRACKER:
            begin_epic_artifact_tracking(workspace, epic_files, run_key=run_key)

        print(
            f"[RuntimeHarness] BEGIN cwd={workspace!r} cmd={command!r} "
            f"timeout_s={timeout_s} epic_id={epic_id!r} "
            f"PYTHONPATH+=.dana_scratch",
            flush=True,
        )
        try:
            result = runner(workspace, command, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            from dana.graph.monitor_bus import publish_graph_error

            msg = f"runtime_harness raised {type(exc).__name__}: {exc}"
            print(f"[RuntimeHarness] CRASH: {msg}", flush=True)
            try:
                publish_graph_error(msg, exc=exc, node="runtime_harness", dump=True)
            except Exception:  # noqa: BLE001
                pass
            result = {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": msg,
            }
        if not isinstance(result, dict):
            result = {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"ERROR: harness returned non-dict: {type(result)!r}",
            }
        print(
            f"[RuntimeHarness] END success={result.get('success')} "
            f"exit={result.get('exit_code')} "
            f"cmd={command!r} "
            f"stderr={(str(result.get('stderr') or '')[:200])!r}",
            flush=True,
        )
        feedback = {
            "success": bool(result.get("success")),
            "exit_code": int(result.get("exit_code") or 0),
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
            "command": command,
            "workspace_path": workspace,
            "epic_id": (epic or {}).get("epic_id"),
            "tracked_files": list(epic_files),
            "run_key": run_key,
        }
        if feedback["success"]:
            # Validated — drop rollback snapshot for this epic.
            _EPIC_FILE_TRACKER.pop(run_key, None)

        epics_out = list(epics)
        # On validation failure: bump epic.repair_attempts (Worker Escalation counter).
        if not feedback["success"] and 0 <= idx < len(epics_out):
            epic_mut = dict(epics_out[idx] or {})
            attempts = int(epic_mut.get("repair_attempts") or 0) + 1
            epic_mut["repair_attempts"] = attempts
            epics_out[idx] = epic_mut
            feedback["repair_attempts"] = attempts
            print(
                f"[RuntimeHarness] repair_attempts → {attempts} "
                f"for epic {epic_mut.get('epic_id')!r}",
                flush=True,
            )

        # Failure routing: Exit-2 collection errors bypass triage LLM.
        # Single-pass (max_repair_attempts=0) also skips triage — broker will ABORT.
        max_repairs = int(state.get("max_repair_attempts") or 0)
        if not feedback["success"] and max_repairs > 0:
            exit_code = int(feedback.get("exit_code") or 0)
            ws = Path(workspace)
            test_rel, impl_rel = _guess_test_and_impl_paths(
                command=command,
                epic_goal=str((epic or {}).get("goal") or ""),
                workspace=ws,
            )
            # Prefer a file named in the current epic goal (modified this epic).
            epic_files_goal = [
                m.group(1).replace("\\", "/")
                for m in _FILE_TOKEN_RE.finditer(str((epic or {}).get("goal") or ""))
            ]
            epic_target = (
                epic_files_goal[0] if epic_files_goal else (test_rel or impl_rel or "")
            )

            if exit_code == 2:
                err_blob = (
                    f"{feedback.get('stderr') or ''}\n"
                    f"{feedback.get('stdout') or ''}"
                ).strip()
                if len(err_blob) > 3500:
                    err_blob = err_blob[:3500] + "\n...[truncated]..."
                critical = (
                    "CRITICAL: Pytest failed to collect tests. This is a syntax "
                    "or import error, NOT a logic failure. Fix imports, typos, or "
                    f"indentation. Traceback:\n{err_blob or '(no traceback)'}"
                )
                feedback["stderr"] = critical
                feedback["repair_triage"] = "COLLECTION"
                feedback["repair_target_filepath"] = epic_target or test_rel or ""
                feedback["repair_triage_raw"] = "exit_code=2 bypass triage"
                feedback["collection_failure"] = True
                print(
                    f"[RuntimeHarness] Exit-2 collection failure → repair "
                    f"target={feedback['repair_target_filepath']!r} "
                    "(triage LLM bypassed)",
                    flush=True,
                )
            else:
                # Bidirectional Repair Triage — before the broker re-dispatches.
                try:
                    test_code = _read_text_capped(ws / test_rel) if test_rel else ""
                    impl_code = _read_text_capped(ws / impl_rel) if impl_rel else ""
                    err_blob = (
                        f"{feedback.get('stderr') or ''}\n"
                        f"{feedback.get('stdout') or ''}"
                    ).strip()
                    if test_code or impl_code:
                        triage = triage_bidirectional_repair(
                            test_code=test_code or "(missing test file)",
                            impl_code=impl_code or "(missing implementation file)",
                            error=err_blob or f"exit_code={feedback.get('exit_code')}",
                            test_filepath=test_rel,
                            impl_filepath=impl_rel,
                        )
                        feedback.update(triage)
                except Exception as triage_exc:  # noqa: BLE001
                    print(
                        f"[RuntimeHarness] triage skipped: {triage_exc}",
                        flush=True,
                    )
                    feedback["repair_triage"] = "CODE"
                    feedback["repair_triage_raw"] = f"triage_error:{triage_exc}"
        elif not feedback["success"]:
            print(
                "[RuntimeHarness] single-pass fail-fast — triage LLM skipped",
                flush=True,
            )

        return {
            "runtime_feedback": feedback,
            "broker_phase": "feedback",
            "status": "evaluating",
            "epics": epics_out,
        }

    return _node


__all__ = (
    "DEFAULT_HARNESS_TIMEOUT_S",
    "DEFAULT_VALIDATION_COMMAND",
    "begin_epic_artifact_tracking",
    "commit_epic_artifact_tracking",
    "make_runtime_harness_node",
    "register_generated_paths",
    "rollback_scratch_workspace",
    "run_validation_harness",
    "triage_bidirectional_repair",
)
