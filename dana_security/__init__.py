"""Isolated verification sandbox for LLM-generated tool code.

Security model:
  1. AST static analysis — block dangerous imports and builtins.
  2. Multiprocessing jail — hard TTL kill switch (default 2000 ms).
"""

from __future__ import annotations

import ast
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Windows: hide helper consoles (CREATE_NO_WINDOW = 0x08000000).
_CREATE_NO_WINDOW = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))

# Multiprocessing spawn bypasses Popen — force windowless interpreter for workers.
if os.name == "nt":
    _pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.isfile(_pythonw):
        mp.set_executable(_pythonw)


def _subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    return {"creationflags": _CREATE_NO_WINDOW}


def _nt_hide_worker_console() -> None:
    """Hide console for multiprocessing children only (never the MainProcess UI)."""
    if os.name != "nt":
        return
    try:
        if mp.current_process().name == "MainProcess":
            return
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:  # noqa: BLE001
        pass

# Host-takeover surface — strictly forbidden in generated tools.
BLOCKED_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "multiprocessing",
        "pathlib",
        "importlib",
        "builtins",
        "pickle",
        "marshal",
        "pty",
        "fcntl",
        "signal",
        "threading",
        "http",
        "urllib",
        "requests",
        "winreg",
        "msvcrt",
    }
)

BLOCKED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)

DEFAULT_TTL_MS = 2000

# Resolve via dana.paths when available; fall back to package-relative repo root.
try:
    from dana.paths import PROJECT_ROOT as _REPO_ROOT
    from dana.paths import TOOLS_JSON as _TOOLS_JSON
except ImportError:  # pragma: no cover - bootstrap / partial installs
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _TOOLS_JSON = _REPO_ROOT / "dana" / "tools" / "tools.json"

DYNAMIC_DIR = _REPO_ROOT / "dana" / "tools" / "dynamic"
GENERATED_TOOLS_PATH = DYNAMIC_DIR / "generated_tools.py"
TOOLS_JSON_PATH = _TOOLS_JSON


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    error: str = ""
    stdout: str = ""
    blocked_by_ast: bool = False
    timed_out: bool = False
    elapsed_ms: float = 0.0


class SandboxSecurityError(ValueError):
    pass


class _SecurityVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = (alias.name or "").split(".")[0]
            if root in BLOCKED_IMPORTS:
                raise SandboxSecurityError(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        root = mod.split(".")[0]
        if root in BLOCKED_IMPORTS:
            raise SandboxSecurityError(f"Blocked import from: {mod}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in BLOCKED_NAMES:
            raise SandboxSecurityError(f"Blocked call: {name}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block os.system-style attribute access on forbidden roots when loaded via alias tricks.
        if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_IMPORTS:
            raise SandboxSecurityError(f"Blocked attribute on {node.value.id}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Only block direct loads of dangerous callables (eval/exec/open/...).
        if (
            node.id in BLOCKED_NAMES
            and isinstance(node.ctx, ast.Load)
            and node.id
            in {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}
        ):
            raise SandboxSecurityError(f"Blocked name: {node.id}")
        self.generic_visit(node)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def validate_ast(python_code: str) -> None:
    """Raise SandboxSecurityError if the code violates sandbox policy."""
    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        raise SandboxSecurityError(f"Syntax error: {exc}") from exc
    _SecurityVisitor().visit(tree)


def _worker(code: str, entry: str, args: tuple, kwargs: dict, queue: mp.Queue) -> None:
    """Child process entry — restricted builtins, no blocked imports."""
    _nt_hide_worker_console()
    try:
        validate_ast(code)
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }
        env: dict[str, Any] = {"__builtins__": safe_builtins}
        compiled = compile(code, "<dana_security>", "exec")
        exec(compiled, env, env)  # noqa: S102 — intentional jail exec after AST gate
        fn = env.get(entry)
        if not callable(fn):
            queue.put({"ok": False, "error": f"Entry function '{entry}' not found or not callable"})
            return
        out = fn(*args, **kwargs)
        queue.put({"ok": True, "result": out})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def _worker_stdio() -> None:
    """stdin/stdout pickle worker for Windows CREATE_NO_WINDOW subprocess path."""
    _nt_hide_worker_console()
    job = pickle.load(sys.stdin.buffer)  # noqa: S301 — host-trusted sandbox job
    code = job["code"]
    entry = job["entry"]
    args = tuple(job.get("args") or ())
    kwargs = dict(job.get("kwargs") or {})
    queue: list[dict[str, Any]] = []

    class _Q:
        def put(self, item: dict[str, Any]) -> None:
            queue.append(item)

    _worker(code, entry, args, kwargs, _Q())  # type: ignore[arg-type]
    pickle.dump(queue[0] if queue else {"ok": False, "error": "empty"}, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def _run_in_sandbox_subprocess(
    python_code: str,
    *,
    entry: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ttl_ms: int,
    t0: float,
) -> SandboxResult:
    """Windows-safe worker: subprocess with CREATE_NO_WINDOW (no ghost console)."""
    import time

    job = {
        "code": python_code,
        "entry": entry,
        "args": list(args),
        "kwargs": dict(kwargs),
    }
    cmd = [
        sys.executable,
        "-c",
        "from dana_security import _worker_stdio; _worker_stdio()",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parent),
            **_subprocess_kwargs(),
        )
    except OSError as exc:
        return SandboxResult(
            ok=False,
            error=f"Sandbox spawn failed: {exc}",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    try:
        out, err = proc.communicate(
            pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL),
            timeout=max(0.05, ttl_ms / 1000.0),
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)
        return SandboxResult(
            ok=False,
            error=f"Sandbox TTL exceeded ({ttl_ms} ms)",
            timed_out=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not out:
        detail = (err or b"").decode("utf-8", errors="replace")[:400]
        return SandboxResult(
            ok=False,
            error=f"Sandbox worker exited without result: {detail}",
            elapsed_ms=elapsed_ms,
        )
    try:
        payload = pickle.loads(out)  # noqa: S301
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(
            ok=False,
            error=f"Sandbox result decode failed: {exc}",
            elapsed_ms=elapsed_ms,
        )
    if payload.get("ok"):
        return SandboxResult(ok=True, result=payload.get("result"), elapsed_ms=elapsed_ms)
    return SandboxResult(
        ok=False,
        error=str(payload.get("error") or "unknown"),
        elapsed_ms=elapsed_ms,
    )


def run_in_sandbox(
    python_code: str,
    *,
    entry: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    ttl_ms: int = DEFAULT_TTL_MS,
) -> SandboxResult:
    """AST-validate then execute ``entry(*args)`` in a killed-on-timeout worker."""
    import time

    t0 = time.perf_counter()
    try:
        validate_ast(python_code)
    except SandboxSecurityError as exc:
        return SandboxResult(
            ok=False,
            error=str(exc),
            blocked_by_ast=True,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # Windows: prefer CREATE_NO_WINDOW subprocess to avoid ghost consoles from mp.spawn.
    if os.name == "nt":
        return _run_in_sandbox_subprocess(
            python_code,
            entry=entry,
            args=args,
            kwargs=dict(kwargs or {}),
            ttl_ms=ttl_ms,
            t0=t0,
        )

    queue: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_worker,
        args=(python_code, entry, args, dict(kwargs or {}), queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=max(0.05, ttl_ms / 1000.0))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
        return SandboxResult(
            ok=False,
            error=f"Sandbox TTL exceeded ({ttl_ms} ms)",
            timed_out=True,
            elapsed_ms=elapsed_ms,
        )

    if queue.empty():
        return SandboxResult(
            ok=False,
            error="Sandbox worker exited without result",
            elapsed_ms=elapsed_ms,
        )
    payload = queue.get()
    if payload.get("ok"):
        return SandboxResult(ok=True, result=payload.get("result"), elapsed_ms=elapsed_ms)
    return SandboxResult(ok=False, error=str(payload.get("error") or "unknown"), elapsed_ms=elapsed_ms)


def _safe_tool_name(name: str) -> str:
    cleaned = re_sub_tool(name)
    if not cleaned or not cleaned[0].isalpha():
        raise ValueError(f"Invalid tool_name: {name!r}")
    return cleaned


def re_sub_tool(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_]", "", (name or "").strip())


def ensure_dynamic_package() -> None:
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    init_path = DYNAMIC_DIR / "__init__.py"
    if not init_path.is_file():
        init_path.write_text('"""Dynamically synthesized Donna tools."""\n', encoding="utf-8")
    if not GENERATED_TOOLS_PATH.is_file():
        GENERATED_TOOLS_PATH.write_text(
            '"""Auto-generated tools (architect_new_tool). Do not edit by hand."""\n\n',
            encoding="utf-8",
        )


def append_generated_function(tool_name: str, python_code: str) -> Path:
    """Serialize verified code into dana/tools/dynamic/generated_tools.py."""
    ensure_dynamic_package()
    name = _safe_tool_name(tool_name)
    block = textwrap.dedent(
        f"""

# --- begin dynamic tool: {name} ---
{python_code.rstrip()}
# --- end dynamic tool: {name} ---
"""
    )
    existing = GENERATED_TOOLS_PATH.read_text(encoding="utf-8") if GENERATED_TOOLS_PATH.is_file() else ""
    # Replace prior definition of the same tool if present.
    marker_start = f"# --- begin dynamic tool: {name} ---"
    marker_end = f"# --- end dynamic tool: {name} ---"
    if marker_start in existing:
        pre, rest = existing.split(marker_start, 1)
        if marker_end in rest:
            _, post = rest.split(marker_end, 1)
            existing = pre.rstrip() + post
        else:
            existing = pre
    GENERATED_TOOLS_PATH.write_text(existing.rstrip() + "\n" + block + "\n", encoding="utf-8")
    return GENERATED_TOOLS_PATH


def register_tool_schema(
    tool_name: str,
    *,
    description_en: str | None = None,
    param_name: str = "text",
) -> dict[str, Any]:
    """Append (or replace) a tool schema entry in tools.json and return the spec."""
    name = _safe_tool_name(tool_name)
    spec = {
        "id": name,
        "description_en": description_en
        or f"Dynamically synthesized tool `{name}` (sandbox-verified).",
        "description_fa": f"  `{name}` (  ).",
        "parameters": [
            {
                "name": param_name,
                "type": "string",
                "required": True,
                "description_en": "Primary text input.",
                "description_fa": "  .",
            }
        ],
        "aliases_en": {"_intent": [name.replace("_", " "), f"run {name}"]},
        "aliases_fa": {"_intent": [name]},
        "dynamic": True,
    }
    with open(TOOLS_JSON_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    tools = list(payload.get("tools") or [])
    tools = [t for t in tools if str(t.get("id")) != name]
    tools.append(spec)
    payload["tools"] = tools
    tmp = str(TOOLS_JSON_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, TOOLS_JSON_PATH)
    return spec


def architect_new_tool(
    tool_name: str,
    python_code: str,
    *,
    test_input: str = "hello",
    ttl_ms: int = DEFAULT_TTL_MS,
) -> dict[str, Any]:
    """Verify code in the sandbox; on success, persist + register the tool schema."""
    try:
        from dana.settings import is_dynamic_tool_synthesis_enabled, synthesis_locked_message

        if not is_dynamic_tool_synthesis_enabled():
            return {
                "ok": False,
                "locked": True,
                "error": synthesis_locked_message("en"),
            }
    except Exception:  # noqa: BLE001
        # Fail closed if settings cannot be read.
        return {
            "ok": False,
            "locked": True,
            "error": "Dynamic tool synthesis is locked (settings unavailable).",
        }

    name = _safe_tool_name(tool_name)
    code = textwrap.dedent(python_code or "").strip()
    if not code:
        return {"ok": False, "error": "python_code is empty"}

    # Phase 1: AST gate (also re-checked inside worker).
    try:
        validate_ast(code)
    except SandboxSecurityError as exc:
        return {"ok": False, "error": str(exc), "blocked_by_ast": True}

    # Phase 2: execute against test input.
    sand = run_in_sandbox(code, entry=name, args=(test_input,), ttl_ms=ttl_ms)
    if not sand.ok:
        return {
            "ok": False,
            "error": sand.error,
            "timed_out": sand.timed_out,
            "blocked_by_ast": sand.blocked_by_ast,
            "elapsed_ms": sand.elapsed_ms,
        }

    # Phase 3: persist + register.
    path = append_generated_function(name, code)
    schema = register_tool_schema(name)
    return {
        "ok": True,
        "tool_name": name,
        "test_result": sand.result,
        "path": str(path),
        "schema_id": schema["id"],
        "elapsed_ms": sand.elapsed_ms,
    }


def load_dynamic_source(tool_name: str) -> str | None:
    """Extract a single dynamic tool's source from generated_tools.py."""
    name = _safe_tool_name(tool_name)
    if not GENERATED_TOOLS_PATH.is_file():
        return None
    text = GENERATED_TOOLS_PATH.read_text(encoding="utf-8")
    marker_start = f"# --- begin dynamic tool: {name} ---"
    marker_end = f"# --- end dynamic tool: {name} ---"
    if marker_start not in text or marker_end not in text:
        return None
    body = text.split(marker_start, 1)[1].split(marker_end, 1)[0]
    return body.strip()


def execute_dynamic_tool(tool_name: str, text: str, *, ttl_ms: int = DEFAULT_TTL_MS) -> SandboxResult:
    """Re-run a registered dynamic tool inside the sandbox (never import into host)."""
    source = load_dynamic_source(tool_name)
    if not source:
        return SandboxResult(ok=False, error=f"Dynamic tool source not found: {tool_name}")
    return run_in_sandbox(source, entry=_safe_tool_name(tool_name), args=(text,), ttl_ms=ttl_ms)
