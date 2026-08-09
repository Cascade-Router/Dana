"""Dana Watchdog — LangGraph swarm (coder → AST gate → Titan → REPL).

Architecture:
  1. ``dana_coder`` — Template Method: LLM fills ``run_self_test`` / ``monitor_loop``
     bodies (JSON preferred); assembler wraps them in ``BaseWatchdog``.
  2. ``ast_static_analyzer`` — deterministic AST gatekeeper (forbidden imports,
     required methods). Failures bypass the LLM supervisor entirely.
  3. ``titan_supervisor`` — subjective logic / task-fit review (LLM).
  4. ``repl_executor`` — sandboxed subprocess + TTS bridge.

Usage:
  from dana.swarm.watchdog_graph import build_watchdog_graph, run_watchdog
  result = run_watchdog("Capture the screen every 30s and alert if Notepad opens")
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from dana.paths import PROJECT_ROOT, EXECUTION_JAIL_DIR, EXECUTION_JAIL_LIBRARY_DIR
from dana.swarm.watchdog_template import (
    FORBIDDEN_IMPORT_ROOTS,
    REQUIRED_METHODS,
    assemble_watchdog_script,
    parse_coder_payload,
)

DEFAULT_MODEL = "qwen2.5-coder:7b"
MAX_REVISIONS = 3
DEFAULT_EXEC_TIMEOUT_S = 45.0
_TTS_MARKER = "__DANA_TTS__:"

_REPO_ROOT = PROJECT_ROOT
_EXECUTION_JAIL_DIR = EXECUTION_JAIL_DIR
_EXECUTION_JAIL_LIBRARY_DIR = EXECUTION_JAIL_LIBRARY_DIR

_FORBIDDEN_BUILTINS = frozenset(
    {"eval", "exec", "compile", "__import__", "globals", "locals", "memoryview"}
)


def ensure_watchdog_sandbox() -> Path:
    """Create ``execution_jail/`` (+ ``library/``) if missing.

    Always ``dana.paths.EXECUTION_JAIL_DIR`` — never process cwd / repo root.
    """
    sandbox = EXECUTION_JAIL_DIR.resolve()
    library = EXECUTION_JAIL_LIBRARY_DIR.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    library.mkdir(parents=True, exist_ok=True)
    return Path(os.path.abspath(str(sandbox)))


def preflight_watchdog_write(
    code: str = "",
    *,
    require_code: bool = False,
) -> Path:
    """Verify sandbox write permissions and optional script integrity before write/exec.

    Raises ``RuntimeError`` with an explicit reason (logged by callers) instead of
    silently handing a broken path or empty script to TTS.
    """
    try:
        sandbox = ensure_watchdog_sandbox()
    except OSError as exc:
        raise RuntimeError(
            f"Watchdog sandbox unavailable under {EXECUTION_JAIL_DIR}: {exc}"
        ) from exc

    if not os.access(str(sandbox), os.W_OK | os.X_OK):
        raise RuntimeError(
            f"Watchdog sandbox is not writable: {sandbox}"
        )

    probe = sandbox / f".dana_write_probe_{os.getpid()}.tmp"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Watchdog sandbox write probe failed ({sandbox}): {exc}"
        ) from exc
    finally:
        try:
            if probe.is_file():
                probe.unlink()
        except OSError:
            pass

    if require_code:
        script = (code or "").strip()
        if not script:
            raise RuntimeError("Watchdog script template is empty — refusing to write")
        if "\x00" in script:
            raise RuntimeError("Watchdog script template contains NUL bytes — malformed")
        looks_python = any(
            token in script
            for token in (
                "import ",
                "from ",
                "def ",
                "class ",
                "assert ",
                "print(",
                "__DANA_TTS__",
                "BaseWatchdog",
                "GeneratedWatchdog",
            )
        )
        if not looks_python:
            raise RuntimeError(
                "Watchdog script template is malformed "
                "(no recognizable Python statements / TTS marker)"
            )

    return sandbox


_FILE_MODIFICATION_TOOLS = frozenset({"edit_file", "write_file", "delete_file"})
_PATH_KWARG_KEYS = ("path", "file_path", "filepath", "file", "target", "filename")
_CODE_KWARG_KEYS = ("content", "code", "text", "contents", "new_str", "new_string")


def verify_payload(tool_name: str, kwargs: dict) -> bool:
    """Dry-run verification for file-modification tool payloads.

    Ensures the target path stays under the active workspace directory and that
    any written Python source passes a basic AST syntax check.
    """
    name = str(tool_name or "").strip()
    if name not in _FILE_MODIFICATION_TOOLS:
        return False

    args = dict(kwargs or {})
    path_raw: Any = None
    for key in _PATH_KWARG_KEYS:
        if key in args and args[key] not in (None, ""):
            path_raw = args[key]
            break
    if path_raw is None:
        return False

    try:
        from dana.paths import DANA_WORKSPACE

        workspace = Path(DANA_WORKSPACE).resolve()
        target = Path(str(path_raw)).expanduser()
        if not target.is_absolute():
            target = (workspace / target).resolve()
        else:
            target = target.resolve()
        target.relative_to(workspace)
    except (OSError, TypeError, ValueError):
        return False

    if name in ("edit_file", "write_file"):
        code: str | None = None
        for key in _CODE_KWARG_KEYS:
            val = args.get(key)
            if isinstance(val, str):
                code = val
                break
        if code is not None:
            looks_py = str(target).lower().endswith(".py") or any(
                token in code
                for token in ("import ", "from ", "def ", "class ", "print(")
            )
            if looks_py:
                try:
                    ast.parse(code)
                except SyntaxError:
                    return False
    return True


class WatchdogState(TypedDict):
    task: str
    code: str
    feedback: str  # subjective LLM supervisor feedback
    lint_errors: str  # deterministic AST / static analyzer errors
    status: str
    # pending | drafting | LINT_OK | LINT_FAIL | APPROVED | REJECTED:… | executed | error
    revisions: int
    history: list[dict[str, Any]]


DANA_CODER_SYSTEM = """
You are Dana's Watchdog coder. You do NOT write a full Python script.
You only fill the Template Method hooks on GeneratedWatchdog(BaseWatchdog).

Output ONLY a JSON object (no markdown fences, no commentary) with this schema:
{
  "extra_imports": ["time"],
  "run_self_test": "<python statements for self.run_self_test body>",
  "monitor_loop": "<python statements for self.monitor_loop body>"
}

Rules for method bodies:
- Pure Python statements only (they will be indented inside the methods).
- Prefer allow-listed imports via extra_imports: time, pathlib, mss, PIL, pyautogui,
  typing, json, re, math, collections, dataclasses, datetime, random, hashlib, numpy, cv2.
- NEVER import or reference: os, sys, subprocess, shutil, socket, pty, requests, urllib,
  ctypes, pickle, importlib, eval, exec.
- run_self_test MUST validate assumptions (assert / raise) before monitoring.
- monitor_loop MUST call self.alert("short phrase") when the watch condition is met.
  self.alert already prints '__DANA_TTS__: …' — do not invent other TTS APIs.
- Honor a single-pass style suitable for DANA_WATCHDOG_ONCE=1 (no infinite loops).
- Keep each method body short and self-contained.
""".strip()


def _llm_content(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    if content is None:
        return str(result or "").strip()
    return str(content).strip()


def _extract_python_source(raw: str) -> str:
    """Legacy helper kept for tests / fallbacks that still pass raw Python."""
    text = (raw or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        return fence.group(1).strip()
    if "import " in text or "def " in text or "from " in text or "{" in text:
        candidates = [
            i
            for i in (
                text.find("{"),
                text.find("import "),
                text.find("from "),
                text.find("def "),
            )
            if i >= 0
        ]
        idx = min(candidates) if candidates else 0
        return text[idx:].strip()
    return text


def _chat_ollama(model: str = DEFAULT_MODEL, temperature: float = 0.2):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)


def dana_coder(state: WatchdogState, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Draft Template Method bodies (JSON) and assemble a full BaseWatchdog script."""
    task = (state.get("task") or "").strip()
    lint_errors = (state.get("lint_errors") or "").strip()
    feedback = (state.get("feedback") or "").strip()
    revisions = int(state.get("revisions") or 0)

    user = f"MONITORING TASK:\n{task or '(empty task)'}\n"
    if lint_errors:
        user += (
            "\nFATAL AST LINT ERRORS (deterministic — fix these first):\n"
            f"{lint_errors}\n"
            "\nRevise the JSON method bodies accordingly.\n"
        )
    elif feedback:
        user += (
            "\nTITAN SUPERVISOR FEEDBACK (address every point):\n"
            f"{feedback}\n"
            "\nRevise the JSON method bodies accordingly.\n"
        )
    else:
        user += "\nEmit the JSON for run_self_test + monitor_loop now.\n"

    try:
        llm = _chat_ollama(model=model, temperature=0.2)
        raw = _llm_content(
            llm.invoke(
                [
                    {"role": "system", "content": DANA_CODER_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        )
        payload = parse_coder_payload(_extract_python_source(raw) or raw)
        self_test = str(payload.get("run_self_test") or "").strip()
        monitor = str(payload.get("monitor_loop") or "").strip()
        if not self_test and not monitor:
            return {
                "code": "",
                "lint_errors": "",
                "feedback": "Dana coder returned empty Template Method bodies",
                "status": "error",
                "revisions": revisions,
            }
        code = assemble_watchdog_script(
            run_self_test=self_test or "pass",
            monitor_loop=monitor or "pass",
            extra_imports=list(payload.get("extra_imports") or []),
        )
        return {
            "code": code,
            "lint_errors": "",
            "status": "drafting",
            "revisions": revisions,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "code": state.get("code") or "",
            "lint_errors": "",
            "feedback": f"Dana coder failed: {exc}",
            "status": "error",
            "revisions": revisions,
        }


def analyze_watchdog_ast(code: str) -> list[str]:
    """Deterministic AST gatekeeper. Returns FATAL error strings (empty = pass)."""
    blob = (code or "").strip()
    errors: list[str] = []
    if not blob:
        return ["FATAL: empty assembled script."]

    try:
        tree = ast.parse(blob)
    except SyntaxError as exc:
        return [f"FATAL: SyntaxError: {exc}"]

    class_methods: dict[str, set[str]] = {}
    generated_node: ast.ClassDef | None = None
    has_generated = False
    has_base = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    errors.append(
                        f"FATAL: ast.Import detected '{alias.name}'. Remove immediately."
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0] if mod else ""
            # ``from abc import …`` is part of the trusted BaseWatchdog embed.
            if root and root != "abc" and root in FORBIDDEN_IMPORT_ROOTS:
                errors.append(
                    f"FATAL: ast.ImportFrom detected '{mod}'. Remove immediately."
                )
        elif isinstance(node, ast.ClassDef):
            if node.name == "BaseWatchdog":
                has_base = True
            if node.name == "GeneratedWatchdog":
                has_generated = True
                generated_node = node
                methods = {
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                class_methods[node.name] = methods
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTINS:
                errors.append(
                    f"FATAL: forbidden built-in call '{func.id}()'. Remove immediately."
                )

    if not has_base:
        errors.append("FATAL: missing required class 'BaseWatchdog'.")
    if not has_generated:
        errors.append("FATAL: missing required class 'GeneratedWatchdog'.")
    else:
        methods = class_methods.get("GeneratedWatchdog") or set()
        for required in REQUIRED_METHODS:
            if required not in methods:
                errors.append(
                    f"FATAL: missing required method "
                    f"'GeneratedWatchdog.{required}'."
                )

    # TTS / alert must appear inside GeneratedWatchdog methods (not only BaseWatchdog).
    if generated_node is not None:
        gen_src = ast.get_source_segment(blob, generated_node) or ""
        has_alert_call = False
        for node in ast.walk(generated_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "alert":
                has_alert_call = True
                break
            if isinstance(func, ast.Name) and func.id == "alert":
                has_alert_call = True
                break
        has_tts_literal = "__DANA_TTS__" in gen_src
        if not has_alert_call and not has_tts_literal:
            errors.append(
                "FATAL: missing TTS alert path in GeneratedWatchdog "
                "(need self.alert(...) or print('__DANA_TTS__: …'))."
            )

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for err in errors:
        if err not in seen:
            seen.add(err)
            unique.append(err)
    return unique


def ast_static_analyzer(state: WatchdogState) -> dict[str, Any]:
    """LangGraph node: programmatic AST lint — bypasses Titan on failure."""
    code = state.get("code") or ""
    revisions = int(state.get("revisions") or 0)
    fatal = analyze_watchdog_ast(code)
    history = list(state.get("history") or [])

    if fatal:
        lint_blob = "\n".join(fatal)
        revisions = revisions + 1
        history.append(
            {
                "stage": "ast_lint",
                "revision": revisions,
                "code": code,
                "feedback": lint_blob,
                "status": "LINT_FAIL",
            }
        )
        return {
            "lint_errors": lint_blob,
            # Keep feedback untouched — lint is a separate channel.
            "status": "LINT_FAIL",
            "revisions": revisions,
            "history": history,
        }

    history.append(
        {
            "stage": "ast_lint",
            "revision": revisions,
            "code": code,
            "feedback": "",
            "status": "LINT_OK",
        }
    )
    return {
        "lint_errors": "",
        "status": "LINT_OK",
        "revisions": revisions,
        "history": history,
    }


def titan_supervisor(state: WatchdogState, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Titan supervises generated code — subjective logic / task-fit (LLM)."""
    from dana_jason_loop.jason_critic import review_watchdog_code

    task = state.get("task") or ""
    code = state.get("code") or ""
    revisions = int(state.get("revisions") or 0) + 1

    def _ask(system: str, user: str) -> str:
        llm = _chat_ollama(model=model, temperature=0.1)
        return _llm_content(
            llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        )

    def _with_history(feedback: str, status_out: str) -> dict[str, Any]:
        history = list(state.get("history") or [])
        history.append(
            {
                "stage": "titan_eval",
                "revision": revisions,
                "code": code,
                "feedback": feedback,
                "status": status_out,
            }
        )
        return {
            "feedback": feedback,
            "lint_errors": "",  # clear lint channel after subjective review
            "status": status_out,
            "revisions": revisions,
            "history": history,
        }

    try:
        verdict = review_watchdog_code(
            code,
            task=task,
            model=model,
            ask_fn=_ask,
        )
    except Exception as exc:  # noqa: BLE001
        status_out = f"REJECTED: Titan supervisor failed ({exc})"
        return _with_history(f"Titan supervisor failed ({exc})", status_out)

    if verdict.strip().upper().startswith("APPROVED"):
        return _with_history("APPROVED", "APPROVED")

    status = verdict.strip()
    if not status.upper().startswith("REJECTED"):
        status = f"REJECTED: {status}"
    reason = status.split(":", 1)[-1].strip() if ":" in status else status
    status_out = (
        status if status.upper().startswith("REJECTED") else f"REJECTED: {reason}"
    )
    return _with_history(reason, status_out)


# Backward-compatible alias (internal module name unchanged).
jason_supervisor = titan_supervisor


def repl_executor(
    state: WatchdogState,
    *,
    timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
    stop_event: threading.Event | None = None,
    on_process: Callable[[subprocess.Popen[str] | None], None] | None = None,
) -> dict[str, Any]:
    """Safely execute Titan-approved code in a timed subprocess + TTS bridge."""
    from dana_jason_loop.jason_critic import static_code_safety_reject

    code = (state.get("code") or "").strip()

    def _with_history(feedback: str, status: str) -> dict[str, Any]:
        history = list(state.get("history") or [])
        history.append(
            {
                "stage": "execution",
                "revision": state.get("revisions", 0),
                "code": code,
                "feedback": feedback,
                "status": status,
            }
        )
        return {"feedback": feedback, "status": status, "history": history}

    if stop_event is not None and stop_event.is_set():
        return _with_history("repl_executor cancelled", "error")

    if not code:
        return _with_history("repl_executor: empty code", "error")

    blocked = static_code_safety_reject(code)
    if blocked:
        return _with_history(
            f"repl_executor refused unsafe code: {blocked}",
            "error",
        )

    try:
        sandbox = preflight_watchdog_write(code, require_code=True)
    except RuntimeError as exc:
        try:
            from dana.logging import log_exception

            log_exception("Watchdog", "Watchdog script write preflight failed", exc=exc)
        except Exception:
            pass
        try:
            from dana.db_core import log_watchdog_event

            log_watchdog_event(
                f"preflight failed: {exc}",
                level="error",
                stage="repl_executor",
            )
        except Exception:
            pass
        return _with_history(f"repl_executor preflight failed: {exc}", "error")

    script = code + "\n"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DANA_WATCHDOG_ONCE"] = "1"
    env["DANA_SANDBOX"] = str(sandbox)
    env["PYTHONPATH"] = (
        str(_REPO_ROOT)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", "")

    path: str | None = None
    proc: subprocess.Popen[str] | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_dana_watchdog.py",
                delete=False,
                encoding="utf-8",
                dir=str(sandbox),
                prefix="run_",
            ) as fh:
                fh.write(script)
                path = fh.name
        except OSError as exc:
            err = RuntimeError(
                f"Failed to write monitor script into {sandbox}: {exc}"
            )
            try:
                from dana.logging import log_exception

                log_exception("Watchdog", "Watchdog script file write failed", exc=err)
            except Exception:
                pass
            raise err from exc

        proc = subprocess.Popen(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.path.abspath(str(sandbox)),
            env=env,
        )
        if on_process is not None:
            on_process(proc)

        stdout_lines: list[str] = []

        def _stream_reader() -> None:
            if not proc.stdout:
                return
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                stdout_lines.append(line)
                if line.startswith(_TTS_MARKER):
                    phrase = line[len(_TTS_MARKER) :].strip()
                    if phrase:
                        try:
                            from dana.core_agent import enqueue_speech as _live_enqueue

                            _live_enqueue(phrase)
                        except Exception:
                            pass

        reader_thread = threading.Thread(target=_stream_reader, daemon=True)
        reader_thread.start()

        deadline = time.monotonic() + float(timeout_s)
        while True:
            if stop_event is not None and stop_event.is_set():
                proc.kill()
                proc.wait(timeout=1.0)
                return _with_history("repl_executor cancelled", "error")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait(timeout=1.0)
                tip = f"repl_executor timed out after {timeout_s:.0f}s"
                return _with_history(tip, "error")

            try:
                proc.wait(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        reader_thread.join(timeout=0.5)

        stderr = proc.stderr.read() if proc.stderr else ""
        stdout_full = "".join(stdout_lines)

        clean_out = "\n".join(
            ln for ln in stdout_full.splitlines() if not ln.startswith(_TTS_MARKER)
        ).strip()

        notes = []
        if clean_out:
            notes.append(clean_out[:800])
        if stderr.strip():
            notes.append("stderr: " + stderr.strip()[:400])

        feedback = "\n".join(notes) if notes else "(no output)"

        if proc.returncode != 0:
            return _with_history(
                f"repl_executor exit {proc.returncode}: {feedback}",
                "error",
            )

        return _with_history(feedback, "executed")

    except Exception as exc:  # noqa: BLE001
        return _with_history(f"repl_executor failed: {exc}", "error")
    finally:
        if on_process is not None:
            try:
                on_process(None)
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def terminal_failure(state: WatchdogState) -> dict[str, Any]:
    """Fallback node when max revisions are hit. Logs root cause, then spoken abort."""
    history = list(state.get("history") or [])
    prior = [
        f"r{h.get('revision', '?')}:{h.get('status', '')} — {h.get('feedback', '')}"
        for h in history
        if isinstance(h, dict)
    ]
    lint_errors = (state.get("lint_errors") or "").strip()
    feedback = (state.get("feedback") or "").strip()
    status = (state.get("status") or "").strip()
    root = lint_errors or feedback or status or "Max revisions reached without APPROVED script"
    detail = (
        f"Watchdog aborted writing monitor script. root={root!r}; "
        f"revisions={state.get('revisions', 0)}; prior=[{'; '.join(prior[-3:])}]"
    )
    abort_exc = RuntimeError(detail)
    try:
        from dana.logging import log_exception

        log_exception(
            "Watchdog",
            "Monitor script write aborted (terminal_failure)",
            exc=abort_exc,
        )
    except Exception:
        print(f"[Watchdog] EXCEPTION: {detail}", flush=True)

    phrase = (
        "I encountered an error trying to write that monitor script and had to abort."
    )
    try:
        from dana.core_agent import enqueue_speech

        enqueue_speech(phrase)
    except Exception:
        print(f"{_TTS_MARKER} {phrase}", flush=True)

    history.append(
        {
            "stage": "terminal_failure",
            "revision": state.get("revisions", 0),
            "code": state.get("code", ""),
            "feedback": detail,
            "status": "error",
        }
    )
    return {
        "status": "error",
        "feedback": detail,
        "history": history,
    }


def _route_after_ast(
    state: WatchdogState,
) -> Literal["titan_supervisor", "dana_coder", "terminal_failure"]:
    """Coder → AST → (pass) Titan | (fail, retries left) Coder | terminal."""
    status = (state.get("status") or "").strip().upper()
    revisions = int(state.get("revisions") or 0)

    if status == "LINT_OK":
        return "titan_supervisor"
    if status == "LINT_FAIL" and revisions < MAX_REVISIONS:
        return "dana_coder"
    return "terminal_failure"


def _route_after_titan(
    state: WatchdogState,
) -> Literal["repl_executor", "dana_coder", "terminal_failure"]:
    status = (state.get("status") or "").strip()
    revisions = int(state.get("revisions") or 0)

    if status.upper() == "APPROVED":
        return "repl_executor"
    if status.upper().startswith("REJECTED") and revisions < MAX_REVISIONS:
        return "dana_coder"

    return "terminal_failure"


_route_after_jason = _route_after_titan


def log_episode(state: WatchdogState) -> dict[str, Any]:
    """Persist the final WatchdogState to SQLite episodic memory (best-effort)."""
    try:
        from dana.swarm.experience_logger import log_watchdog_episode

        log_watchdog_episode(state)
    except Exception:
        pass
    return {}


def build_watchdog_graph(
    *,
    model: str = DEFAULT_MODEL,
    exec_timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
    stop_event: threading.Event | None = None,
    on_process: Callable[[subprocess.Popen[str] | None], None] | None = None,
):
    """Compile START → coder → AST → Titan ⇢ REPL → log → END."""

    def _cancelled() -> dict[str, Any]:
        return {
            "feedback": "Watchdog cancelled by user",
            "status": "error",
        }

    def _coder(state: WatchdogState) -> dict[str, Any]:
        if stop_event is not None and stop_event.is_set():
            return _cancelled()
        return dana_coder(state, model=model)

    def _ast(state: WatchdogState) -> dict[str, Any]:
        if stop_event is not None and stop_event.is_set():
            return _cancelled()
        return ast_static_analyzer(state)

    def _titan(state: WatchdogState) -> dict[str, Any]:
        if stop_event is not None and stop_event.is_set():
            return _cancelled()
        return titan_supervisor(state, model=model)

    def _repl(state: WatchdogState) -> dict[str, Any]:
        return repl_executor(
            state,
            timeout_s=exec_timeout_s,
            stop_event=stop_event,
            on_process=on_process,
        )

    graph = StateGraph(WatchdogState)
    graph.add_node("dana_coder", _coder)
    graph.add_node("ast_static_analyzer", _ast)
    graph.add_node("titan_supervisor", _titan)
    graph.add_node("repl_executor", _repl)
    graph.add_node("terminal_failure", terminal_failure)
    graph.add_node("log_episode", log_episode)

    graph.add_edge(START, "dana_coder")
    graph.add_edge("dana_coder", "ast_static_analyzer")
    graph.add_conditional_edges(
        "ast_static_analyzer",
        _route_after_ast,
        {
            "titan_supervisor": "titan_supervisor",
            "dana_coder": "dana_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_conditional_edges(
        "titan_supervisor",
        _route_after_titan,
        {
            "repl_executor": "repl_executor",
            "dana_coder": "dana_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_edge("repl_executor", "log_episode")
    graph.add_edge("terminal_failure", "log_episode")
    graph.add_edge("log_episode", END)
    return graph.compile()


def _empty_state(**overrides: Any) -> WatchdogState:
    base: WatchdogState = {
        "task": "",
        "code": "",
        "feedback": "",
        "lint_errors": "",
        "status": "pending",
        "revisions": 0,
        "history": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def run_watchdog(
    task: str,
    *,
    model: str = DEFAULT_MODEL,
    initial_code: str = "",
    initial_feedback: str = "",
    exec_timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
) -> WatchdogState:
    """Invoke the Watchdog graph; returns the final state."""
    app = build_watchdog_graph(model=model, exec_timeout_s=exec_timeout_s)
    seed = _empty_state(
        task=(task or "").strip(),
        code=initial_code or "",
        feedback=initial_feedback or "",
    )
    result = app.invoke(seed)
    return WatchdogState(
        task=str(result.get("task") or seed["task"]),
        code=str(result.get("code") or ""),
        feedback=str(result.get("feedback") or ""),
        lint_errors=str(result.get("lint_errors") or ""),
        status=str(result.get("status") or "error"),
        revisions=int(result.get("revisions") or 0),
        history=list(result.get("history") or []),
    )


if __name__ == "__main__":
    import sys as _sys

    topic = " ".join(_sys.argv[1:]).strip() or (
        "Every 30 seconds, capture the primary screen and print "
        "'__DANA_TTS__: Notepad detected' if a window titled Notepad is visible; "
        "include a self-test before the loop."
    )
    print(f"Watchdog task: {topic}\n")
    final = run_watchdog(topic)
    print(f"status={final['status']} revisions={final['revisions']}")
    print(f"lint_errors={final.get('lint_errors')!r}")
    print(f"feedback={final['feedback']}\n")
    print(final["code"] or "(no code)")
