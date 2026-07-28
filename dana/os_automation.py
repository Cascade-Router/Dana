"""Safe OS productivity helpers: keystrokes, clipboard, shell, apps, and file reads.

Keystroke injection remains non-destructive (no privileged control chords).
Shell execution is hard-capped with a 10s timeout and truncated stdout/stderr.
Local file reads are truncated to protect the LLM context window.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

# Hard cap so a runaway LLM cannot flood the input queue.
MAX_INJECT_CHARS = 4_000
MAX_CLIPBOARD_CHARS = 8_000
MAX_TERMINAL_OUTPUT_CHARS = 8_000
MAX_LOCAL_FILE_CHARS = 3_000
TERMINAL_COMMAND_TIMEOUT_SEC = 10
TRUNCATION_SUFFIX = "... [TRUNCATED FOR LENGTH]"

# Control / format characters that must never be injected as "typing".
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# Sequences that look like OS-level secure-attention / cancel chords when
# expressed as text (pyautogui hotkey syntax or literal descriptions).
_FORBIDDEN_CHORD_RE = re.compile(
    r"(?ix)"
    r"("
    r"ctrl\s*[\+\-]\s*alt\s*[\+\-]\s*del"
    r"|ctrl\s*[\+\-]\s*alt\s*[\+\-]\s*delete"
    r"|control\s*[\+\-]\s*alt\s*[\+\-]\s*delete"
    r"|alt\s*[\+\-]\s*f4"
    r"|win\s*[\+\-]\s*l"
    r"|ctrl\s*[\+\-]\s*shift\s*[\+\-]\s*esc"
    r"|ctrl\s*[\+\-]\s*esc"
    r")"
)

# Naked shell / PowerShell launch patterns — blocked unless authenticated wrap.
_NAKED_SHELL_RE = re.compile(
    r"(?ix)"
    r"("
    r"^\s*(cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?|bash|sh|wsl)\b"
    r"|^\s*(?:Start-Process|Invoke-Expression|iex)\b"
    r"|^\s*(?:rm\s+-rf|del\s+/[fq]|format\s+[a-z]:)"
    r")"
)

_TRANSIENT_AUTH_ENV = "DONNA_OS_AUTH_CONTEXT"
# Production default: real keystroke injection (unset / "0" / "false").
# Debug re-enable: set DONNA_OS_DRY_RUN=1 in the environment to validate the
# inject_keystrokes pipeline without touching the OS input stack (used by E2E).
_DRY_RUN_ENV = "DONNA_OS_DRY_RUN"


@dataclass
class SanitizeResult:
    text: str
    stripped_controls: int = 0
    blocked: bool = False
    reason: str = ""


def sanitize_keystroke_text(text: str, *, authenticated: bool | None = None) -> SanitizeResult:
    """Strip hazardous controls and reject forbidden chords / naked shell lines."""
    if text is None:
        return SanitizeResult(text="", blocked=True, reason="empty text")
    raw = str(text)
    if not raw:
        return SanitizeResult(text="", blocked=True, reason="empty text")
    if len(raw) > MAX_INJECT_CHARS:
        return SanitizeResult(
            text="",
            blocked=True,
            reason=f"text exceeds {MAX_INJECT_CHARS} character limit",
        )

    if _FORBIDDEN_CHORD_RE.search(raw):
        return SanitizeResult(
            text="",
            blocked=True,
            reason="forbidden OS control chord (e.g. Ctrl+Alt+Del / Alt+F4)",
        )

    auth = (
        authenticated
        if authenticated is not None
        else bool(os.environ.get(_TRANSIENT_AUTH_ENV, "").strip())
    )
    if not auth and _NAKED_SHELL_RE.search(raw):
        return SanitizeResult(
            text="",
            blocked=True,
            reason=(
                "naked shell/command line blocked; "
                "requires authenticated context (DONNA_OS_AUTH_CONTEXT)"
            ),
        )

    cleaned, n_sub = _CONTROL_CHAR_RE.subn("", raw)
    # Normalize newlines to Enter-friendly \n; drop lone CR.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned:
        return SanitizeResult(
            text="",
            stripped_controls=n_sub,
            blocked=True,
            reason="text empty after control-character sanitization",
        )
    return SanitizeResult(text=cleaned, stripped_controls=n_sub)


def _dry_run_enabled() -> bool:
    return os.environ.get(_DRY_RUN_ENV, "").strip().lower() in ("1", "true", "yes")


def inject_keystrokes(
    text: str,
    *,
    interval: float = 0.02,
    authenticated: bool | None = None,
) -> dict[str, Any]:
    """Type sanitized plaintext into the focused window (pyautogui).

    Never sends hotkey chords — only printable / newline characters via typewrite.
    Set DONNA_OS_DRY_RUN=1 to validate without touching the OS input stack.
    """
    result = sanitize_keystroke_text(text, authenticated=authenticated)
    if result.blocked:
        return {
            "ok": False,
            "error": result.reason,
            "chars_typed": 0,
            "stripped_controls": result.stripped_controls,
        }

    if _dry_run_enabled():
        return {
            "ok": True,
            "dry_run": True,
            "chars_typed": len(result.text),
            "stripped_controls": result.stripped_controls,
            "preview": result.text[:80],
        }

    try:
        import pyautogui
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"pyautogui not installed: {exc}",
            "chars_typed": 0,
        }

    # Fail-safe: moving mouse to a corner aborts; keep off for agent typing.
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    # typewrite only emits key events for characters — no chord macros.
    pyautogui.typewrite(result.text, interval=max(0.0, float(interval)))
    return {
        "ok": True,
        "dry_run": False,
        "chars_typed": len(result.text),
        "stripped_controls": result.stripped_controls,
    }


def read_clipboard_context(*, max_chars: int = MAX_CLIPBOARD_CHARS) -> dict[str, Any]:
    """Fetch plaintext clipboard payload for immediate conversational context."""
    max_chars = max(1, min(int(max_chars), MAX_CLIPBOARD_CHARS))
    try:
        text = _read_clipboard_text()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "text": "", "truncated": False}

    if text is None:
        return {
            "ok": True,
            "text": "",
            "empty": True,
            "truncated": False,
            "note": "clipboard empty or non-text format",
        }

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    # Strip NULs / other controls from clipboard before handing to the LLM.
    text = _CONTROL_CHAR_RE.sub("", text)
    return {
        "ok": True,
        "text": text,
        "empty": not bool(text.strip()),
        "truncated": truncated,
        "chars": len(text),
    }


def run_terminal_command(command: str) -> str:
    """Execute a shell command on the host OS and return stdout (or an error string).

    Uses ``subprocess.run`` with ``shell=True``, ``capture_output=True``, and a
    strict ``timeout=10`` so hanging processes cannot stall the ReAct loop.
    Non-zero exits and timeouts return a clean error string for the LLM.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "ERROR: empty command"

    run_kwargs: dict[str, Any] = {
        "shell": True,
        "capture_output": True,
        "text": True,
        "timeout": TERMINAL_COMMAND_TIMEOUT_SEC,
        "encoding": "utf-8",
        "errors": "replace",
    }
    # Avoid flashing a console window when agent.py patches Popen on Windows.
    if os.name == "nt":
        run_kwargs["creationflags"] = int(
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )

    try:
        completed = subprocess.run(cmd, **run_kwargs)  # noqa: S602
    except subprocess.TimeoutExpired as exc:
        partial_err = ""
        if isinstance(exc.stderr, str) and exc.stderr.strip():
            partial_err = f" stderr={exc.stderr.strip()[:MAX_TERMINAL_OUTPUT_CHARS]!r}"
        elif isinstance(exc.stderr, bytes) and exc.stderr:
            partial_err = (
                f" stderr={exc.stderr.decode('utf-8', errors='replace').strip()[:MAX_TERMINAL_OUTPUT_CHARS]!r}"
            )
        return (
            f"ERROR: command timed out after {TERMINAL_COMMAND_TIMEOUT_SEC} seconds."
            f"{partial_err}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: run_terminal_command failed: {exc}"

    stdout = (completed.stdout or "").rstrip()
    stderr = (completed.stderr or "").rstrip()

    if completed.returncode != 0:
        detail = stderr or stdout or "(no output)"
        if len(detail) > MAX_TERMINAL_OUTPUT_CHARS:
            detail = detail[:MAX_TERMINAL_OUTPUT_CHARS] + "\n...[truncated]"
        return f"ERROR: command failed (exit {completed.returncode}): {detail}"

    if len(stdout) > MAX_TERMINAL_OUTPUT_CHARS:
        return stdout[:MAX_TERMINAL_OUTPUT_CHARS] + "\n...[truncated]"
    return stdout


# Spoken name → Windows launch target (PATH / App Paths / shell builtins).
_APP_LAUNCH_MAP: dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "code": "code",
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "firefox": "firefox.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "spotify": "spotify.exe",
    "discord": "discord.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "paint": "mspaint.exe",
}


def _resolve_app_executable(app_name: str) -> str | None:
    raw = (app_name or "").strip()
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    if key in _APP_LAUNCH_MAP:
        return _APP_LAUNCH_MAP[key]
    compact = key.replace(" ", "")
    for alias, exe in _APP_LAUNCH_MAP.items():
        if alias.replace(" ", "") == compact:
            return exe
    # Allow exact mapped executable names (chrome.exe, notepad.exe, code).
    for exe in _APP_LAUNCH_MAP.values():
        if key == exe.lower():
            return exe
    return None


def open_application(app_name: str) -> str:
    """Launch a known Windows application without blocking the agent thread.

    Uses ``subprocess.Popen`` (not ``run``) so GUI apps stay open independently.
    """
    display = (app_name or "").strip() or "(empty)"
    exe = _resolve_app_executable(app_name)
    if not exe:
        return f"ERROR: Unknown application {display}."

    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        # Detach from Donna's console; shell=True resolves App Paths (chrome.exe, code).
        creation = int(getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
        creation |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        popen_kwargs["creationflags"] = creation
        popen_kwargs["shell"] = True
        launch_target: str | list[str] = exe
    else:
        popen_kwargs["shell"] = False
        launch_target = [exe]

    try:
        subprocess.Popen(launch_target, **popen_kwargs)  # noqa: S602
    except FileNotFoundError:
        return f"ERROR: Unknown application {display}."
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Failed to launch {display}: {exc}"

    return f"OK: Launched {display}."


def _resolve_repo_file_candidates(filepath: str) -> list[str]:
    """Resolve a user/LLM filepath against the CAMGRASPER repo root.

    CAMGRASPER is the project root; ``dana/`` is a sub-module (never invent
    ``dana/core/...``). Rewrite that hallucination to ``dana/`` and try
    PROJECT_ROOT-relative candidates before failing.
    """
    from pathlib import Path

    from dana.paths import PROJECT_ROOT

    raw = (filepath or "").strip().strip("\"'")
    if not raw:
        return []

    normalized = raw.replace("\\", "/")
    # Common hallucination: dana/core/<module> instead of dana/<module>.
    normalized = re.sub(r"(?i)\bdonna/core/", "dana/", normalized)
    normalized = re.sub(r"(?i)^core/", "dana/", normalized)

    root = Path(PROJECT_ROOT).resolve()
    candidates: list[Path] = []
    p = Path(normalized)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((root / normalized).resolve())
        # Also try stripping a leading repo-folder name if present.
        parts = Path(normalized).parts
        if parts and parts[0].lower() in {"camgrasper", "dana"}:
            if parts[0].lower() == "camgrasper" and len(parts) > 1:
                candidates.append((root.joinpath(*parts[1:])).resolve())
        # Bare module names under dana/
        name = Path(normalized).name
        if name and not normalized.lower().startswith("dana/"):
            candidates.append((root / "dana" / name).resolve())

    # Dedupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    # Keep the normalized relative string last for clear error messages.
    if normalized not in seen:
        out.append(normalized)
    return out


def read_local_file(filepath: str) -> str:
    """Read a local text file, truncated to ``MAX_LOCAL_FILE_CHARS`` for the LLM.

    Relative paths are resolved from the CAMGRASPER ``PROJECT_ROOT`` first.
    """
    path = (filepath or "").strip()
    if not path:
        return "ERROR: missing filepath"

    candidates = _resolve_repo_file_candidates(path)
    last_error = f"ERROR: File not found: {path}"
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except FileNotFoundError:
            last_error = f"ERROR: File not found: {candidate}"
            continue
        except PermissionError:
            return f"ERROR: Permission denied: {candidate}"
        except IsADirectoryError:
            return f"ERROR: Path is a directory, not a file: {candidate}"
        except OSError as exc:
            last_error = f"ERROR: Failed to read {candidate}: {exc}"
            continue

        if len(text) > MAX_LOCAL_FILE_CHARS:
            return text[:MAX_LOCAL_FILE_CHARS] + TRUNCATION_SUFFIX
        return text

    tried = ", ".join(candidates[:4]) if candidates else path
    return f"{last_error} (resolved from PROJECT_ROOT; tried: {tried})"


def _read_clipboard_text() -> str | None:
    if sys.platform == "win32":
        return _read_clipboard_win32()
    # Cross-platform fallback via tkinter (stdlib).
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            data = root.clipboard_get()
        except tk.TclError:
            data = None
        finally:
            root.destroy()
        return str(data) if data is not None else None
    except Exception:
        return None


def _read_clipboard_win32() -> str | None:
    import win32clipboard
    import win32con

    win32clipboard.OpenClipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            return str(data) if data is not None else None
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
            data = win32clipboard.GetClipboardData(win32con.CF_TEXT)
            if data is None:
                return None
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        return None
    finally:
        win32clipboard.CloseClipboard()
