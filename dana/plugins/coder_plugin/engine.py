"""Software Engineering plugin — wraps ``aider`` (github.com/Aider-AI/aider)
as an isolated subprocess so Dana can read and safely modify its OWN
codebase. Loaded generically via ``dana.plugins.plugin_manager`` +
``dana.core.react_dispatch.refresh_plugin_tools`` — see this repo's
``manifest.json`` for the four tools' declared schemas/domain.

Four tools, two very different risk profiles:

- ``search_codebase``, ``analyze_codebase``, and ``run_verification_command`` are
  genuinely read-only: the first runs a FIXED, non-shell ``git grep -n -E``
  argv (a regex, never an arbitrary command string built from model input)
  to locate matching lines by line number WITHOUT paying to read whole
  files into context; the second reads specific files verbatim once
  ``search_codebase`` (or the user) has already narrowed down which ones
  matter; the third runs a WHITELISTED verification command (pytest,
  flake8, mypy, or ``black --check`` — see its own docstring) so an edit
  can be checked and self-corrected BEFORE it ever reaches a human for
  approval. All three declared ``read_only: true`` in the manifest, so
  they dispatch immediately, no human approval needed.
- ``execute_code_task`` hands a task to ``aider`` in headless one-shot mode
  (``--yes`` — real file edits + a real git commit). Declared
  ``read_only: false`` (the manifest default), so
  ``dana.core.react_dispatch``'s generic plugin wiring HITL-gates it
  exactly like ``execute_terminal_command`` (the DIFFERENT, native
  os_tools tool — see ``run_verification_command``'s own docstring for why
  this plugin's verification runner is named differently on purpose) —
  this function is never reached without an explicit human approval click
  first.

``analyze_codebase``/``execute_code_task`` (the two tools that take path
arguments) confine every one of them to ``PROJECT_ROOT`` (``dana.paths``)
and refuse a short denylist of maximally-sensitive paths (``.env``, ``.git``)
even under an approved call — an LLM-driven coding task must never be the
vector that edits or exfiltrates a real secret. ``search_codebase`` takes
no path argument at all (a repo-wide ``git grep``, optionally scoped by
file extension), so it has nothing to escape — its only guard is that
``regex_pattern``/``file_extension`` are always fixed, separate argv
elements, never a shell string. ``run_verification_command``'s safety boundary
is different again: a fixed ALLOWLIST of verification executables (see its
own docstring), never a path check, since it takes no path argument
either. This plugin deliberately operates on the WHOLE repository, not the
narrower ``AGENT_WORKSPACE_DIR`` sandbox ``dana.plugins.os.file_system``
confines its own tools to — that's the entire point (Dana editing its own
source) — so the path/denylist/allowlist checks here are this plugin's
actual safety boundary, not a formality.
"""

from __future__ import annotations

import difflib
import shlex
import subprocess
from pathlib import Path
from typing import Any

from dana.paths import PROJECT_ROOT

_SEARCH_TIMEOUT_S = 20.0
_ANALYZE_TIMEOUT_S = 20.0
# Aider's own repo-map scan plus a real Gemini round trip regularly takes
# 20-30s even for a trivial task (observed live during this plugin's own
# verification) — generous headroom over that, not a tight ceiling.
_EXECUTE_TIMEOUT_S = 180.0
# A real pytest run (even a narrow one) can take longer than a linter
# invocation — generous enough for a targeted test file/module, not
# intended to cover this repo's ENTIRE suite in one call.
_VERIFY_TIMEOUT_S = 120.0
_MAX_OUTPUT_CHARS = 8_000
_MAX_FILE_READ_CHARS = 20_000

# Real secrets / VCS internals — never reachable through this plugin
# regardless of what a task description or query asks for, even after
# human HITL approval (the approval dialog shows the task text, not an
# audit of every file aider might end up touching).
_DENYLISTED_RELATIVE_PARTS = frozenset({".env", ".git"})

# run_verification_command's ENTIRE safety boundary: the base executable of a
# parsed (never shell-interpreted) command must be one of these keys, or
# the call is refused outright — there is no escape hatch, no "allow this
# once" override, and no attempt to sanitize/allow anything else. A value
# of None means any arguments are fine; a non-None tuple means the FIRST
# argument must equal one of it, in order to rule out an otherwise-allowed
# executable's own mutating invocation.
_ALLOWED_VERIFY_COMMANDS: dict[str, tuple[str, ...] | None] = {
    "pytest": None,
    "flake8": None,
    "mypy": None,
    # Bare `black` REWRITES files in place — only the non-mutating
    # `--check` invocation may run through this read-only tool; an actual
    # reformat must go through execute_code_task (HITL-gated) instead.
    "black": ("--check",),
}


def _validate_verify_command(raw_command: str) -> tuple[list[str] | None, str | None]:
    """Parses ``raw_command`` (``shlex.split`` — never shell-interpreted) and
    checks its base executable against ``_ALLOWED_VERIFY_COMMANDS``. Returns
    ``(argv, None)`` on success or ``(None, error_message)`` on failure.
    Shared by ``run_verification_command`` and ``execute_code_task``'s
    optional ``test_command`` so both enforce the exact same allowlist —
    aider's own ``--test-cmd`` subprocess must never become a second, looser
    escape hatch around the one already established here.
    """
    try:
        argv = shlex.split(raw_command)
    except ValueError as exc:
        return None, f"could not parse command: {exc}"
    if not argv:
        return None, "command must be non-empty"

    base = argv[0]
    if base not in _ALLOWED_VERIFY_COMMANDS:
        return None, (
            f"{base!r} is not a whitelisted verification command — only "
            f"{sorted(_ALLOWED_VERIFY_COMMANDS)} are permitted"
        )
    required_prefix = _ALLOWED_VERIFY_COMMANDS[base]
    if required_prefix and argv[1:2] != list(required_prefix):
        return None, f"{base!r} may only be run as {base!r} followed by {' '.join(required_prefix)!r}"
    return argv, None


class PathEscapeError(ValueError):
    """A path argument resolved outside PROJECT_ROOT or hit the denylist."""


def _resolve_repo_path(rel_path: str) -> Path:
    """Resolves ``rel_path`` against ``PROJECT_ROOT`` and refuses anything
    that escapes it (``..`` traversal, an absolute path elsewhere) or names
    a denylisted secret/VCS-internal path — mirrors ``dana.plugins.os.
    file_system.resolve_sandboxed_path``'s contract, just rooted at the
    whole repo instead of ``AGENT_WORKSPACE_DIR``.
    """
    candidate = (PROJECT_ROOT / rel_path).resolve()
    try:
        rel = candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        raise PathEscapeError(f"path escapes the project root: {rel_path!r}") from None
    if rel.parts and rel.parts[0] in _DENYLISTED_RELATIVE_PARTS:
        raise PathEscapeError(f"path is denylisted (secrets/VCS-internal): {rel_path!r}")
    return candidate


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


# dana.core.react_dispatch.dispatch_tool_call only preserves whatever we
# put in THIS field for a failed tool call — "stdout"/"stderr"/"output"
# below are read by nothing downstream and simply dropped (raw_error =
# str(payload.get("error") or ...), then digest_error truncates THAT to
# its own _RAW_ERROR_MAX_CHARS=400 from the front). A pytest/flake8/mypy
# failure's actually useful line (the assertion, the traceback's raised
# exception, the specific lint violation) is almost always at the END of
# the captured output, not the start (mostly collection/progress noise) —
# so this takes the TAIL, sized to what survives digest_error's own clip
# mostly intact, instead of head-truncating into a hand-off point where
# nothing useful is left for the LLM to act on.
_ERROR_TAIL_CHARS = 400


def _tail_for_error(combined_output: str) -> str:
    if not combined_output:
        return ""
    return combined_output[-_ERROR_TAIL_CHARS:]


def search_codebase(args: dict[str, Any]) -> dict[str, Any]:
    """Read-only context compressor: runs a fixed ``git grep -n -E
    <regex_pattern>`` argv (optionally scoped to one file extension via a
    pathspec) to locate matching lines by line number — WITHOUT reading any
    whole file into context. Use this FIRST to find a function signature,
    class definition, or keyword; only fall back to ``analyze_codebase``
    for the specific files this turns up. ``regex_pattern``/
    ``file_extension`` are always passed as separate, fixed argv elements
    (``shell=False``) — never concatenated into a shell string, so this is
    safe to dispatch with no human approval (see manifest.json's
    ``read_only: true``).
    """
    regex_pattern = str(args.get("regex_pattern") or "").strip()
    if not regex_pattern:
        return {"ok": False, "error": "search_codebase requires a non-empty 'regex_pattern'"}

    # --untracked (still .gitignore-respecting) so a file the agent (or the
    # user) just created but hasn't committed/staged yet — a brand-new
    # plugin, a skill, a fresh module — is still found; a tracked-only grep
    # would silently miss exactly the code most likely to be mid-edit.
    command = ["git", "grep", "--untracked", "-n", "-E", regex_pattern]
    file_extension = str(args.get("file_extension") or "").strip().lstrip(".")
    if file_extension:
        command += ["--", f"*.{file_extension}"]

    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, shell=False; regex_pattern is one arg, never shell-interpreted
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT_S,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"git grep timed out after {_SEARCH_TIMEOUT_S}s"}
    except OSError as exc:
        return {"ok": False, "error": f"could not run git grep: {exc}"}

    # git grep exits 1 (not an error) when it simply finds no matches — a
    # genuine invocation error (e.g. a malformed -E pattern) exits 2+.
    if completed.returncode == 1:
        return {"ok": True, "regex_pattern": regex_pattern, "matches": "No matches found."}
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": _truncate(completed.stderr or f"git grep exited with code {completed.returncode}"),
        }
    return {
        "ok": True,
        "regex_pattern": regex_pattern,
        "matches": _truncate(completed.stdout) if completed.stdout else "No matches found.",
    }


def analyze_codebase(args: dict[str, Any]) -> dict[str, Any]:
    """Read-only reconnaissance: reads ``files`` verbatim. Use
    ``search_codebase`` FIRST to locate which files/lines actually matter
    by pattern — this keeps context usage bounded instead of reading whole
    files speculatively.
    """
    raw_files = args.get("files")
    file_list = [str(f) for f in raw_files] if isinstance(raw_files, list) else []
    if not file_list:
        return {
            "ok": False,
            "error": "analyze_codebase requires a non-empty 'files' list — use search_codebase to find them first",
        }

    contents: dict[str, str] = {}
    errors: dict[str, str] = {}
    for rel_path in file_list:
        try:
            path = _resolve_repo_path(rel_path)
        except PathEscapeError as exc:
            errors[rel_path] = str(exc)
            continue
        if not path.is_file():
            errors[rel_path] = "file does not exist"
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors[rel_path] = f"could not read file: {exc}"
            continue
        contents[rel_path] = _truncate(text, _MAX_FILE_READ_CHARS)
    if not contents and errors:
        return {"ok": False, "error": "; ".join(f"{k}: {v}" for k, v in errors.items())}
    return {"ok": True, "files": contents, "errors": errors or None}


def run_verification_command(args: dict[str, Any]) -> dict[str, Any]:
    """Read-only verification runner: parses ``command`` into a fixed argv
    (``shlex.split`` — never a shell string, ``shell=False``) and refuses
    anything whose base executable isn't in ``_ALLOWED_VERIFY_COMMANDS``.
    This exists ONLY so Dana can run its own tests/linters and read a real
    traceback to self-correct BEFORE ``execute_code_task``'s edit is ever
    handed to a human for approval — it must never become a general-purpose
    shell tool. Declared ``read_only: true`` in the manifest: none of
    pytest/flake8/mypy/``black --check`` mutate the project, so this
    dispatches immediately, no human approval needed.

    Deliberately named differently from the native, ``os_tools`` domain's
    ``execute_terminal_command`` (dana.plugins.os.process_manager) — that
    tool is genuinely arbitrary/mutating and HITL-gated; this one is a
    narrow, hardcoded allowlist with no such escape hatch, so collapsing
    them into one name/schema would blur a real safety distinction.
    """
    raw_command = str(args.get("command") or "").strip()
    if not raw_command:
        return {"ok": False, "error": "run_verification_command requires a non-empty 'command'"}

    argv, error = _validate_verify_command(raw_command)
    if error:
        return {"ok": False, "error": error}

    try:
        completed = subprocess.run(  # noqa: S603 — argv from shlex.split, shell=False; base already allowlist-checked above
            argv,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_VERIFY_TIMEOUT_S,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _truncate(exc.stdout) if isinstance(exc.stdout, str) else ""
        stderr = _truncate(exc.stderr) if isinstance(exc.stderr, str) else ""
        combined = (stdout + "\n" + stderr).strip()
        return {
            "ok": False,
            # Generic on purpose — NOT "FreeCAD subprocess" (that phrasing
            # belongs to dana.plugins.freecad.engine's own timeout, a
            # different plugin entirely). Whatever partial stdout/stderr
            # subprocess.run managed to capture before the kill rides along
            # in "error" too (see _tail_for_error's own docstring for why
            # dispatch_tool_call/digest_error need it there, not only in
            # "output").
            "error": _tail_for_error(combined) or f"Verification command timed out after {_VERIFY_TIMEOUT_S:.0f} seconds.",
            "stdout": stdout,
            "stderr": stderr,
            "output": combined,
        }
    except FileNotFoundError:
        return {"ok": False, "error": f"{base!r} is not installed / not on PATH"}
    except OSError as exc:
        return {"ok": False, "error": f"could not run {base!r}: {exc}"}

    stdout = _truncate(completed.stdout or "")
    stderr = _truncate(completed.stderr or "")
    combined = (stdout + "\n" + stderr).strip()
    if completed.returncode != 0:
        return {
            "ok": False,
            # The real traceback/assertion/lint-violation text lives in
            # "combined" (usually stdout for pytest, not stderr) — falling
            # back to stderr-or-a-bare-exit-code here used to swallow it
            # entirely from the ONE field (dana.core.react_dispatch.
            # dispatch_tool_call's digest_error) that actually survives
            # into what the model reads on a failed tool call; "stdout"/
            # "stderr"/"output" below are dropped on that path, so the
            # traceback the LLM needs to self-correct MUST be in "error".
            "error": _tail_for_error(combined) or f"{base} exited with code {completed.returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "output": combined,
            "returncode": completed.returncode,
        }
    return {"ok": True, "stdout": stdout, "stderr": stderr, "output": combined, "returncode": 0}


def generate_code_task_diff(args: dict[str, Any]) -> str | None:
    """Generate a unified diff preview for execute_code_task before HITL approval.
    
    This reads the current state of files that would be modified and generates
    a preview showing the existing content. Since we can't predict exactly what
    Aider will do, this provides a baseline diff showing what exists now.
    
    Returns None if diff generation fails.
    """
    try:
        raw_files = args.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            return None
            
        task_description = str(args.get("task_description") or "").strip()
        
        diff_lines = []
        diff_lines.append(f"Task: {task_description}")
        diff_lines.append("")
        
        for rel_path in raw_files[:10]:  # Limit to first 10 files to avoid huge diffs
            try:
                resolved_path = _resolve_repo_path(str(rel_path))
                
                if resolved_path.exists() and resolved_path.is_file():
                    # Read current content
                    try:
                        current_content = resolved_path.read_text(encoding='utf-8')
                        current_lines = current_content.splitlines(keepends=True)
                        
                        # Generate a preview showing current file state
                        diff_lines.extend([
                            f"--- {rel_path}",
                            f"+++ {rel_path} (will be modified)",
                            f"@@ -1,{len(current_lines)} +1,? @@",
                        ])
                        
                        # Show first few and last few lines of current content
                        preview_lines = min(10, len(current_lines))
                        for i, line in enumerate(current_lines[:preview_lines]):
                            diff_lines.append(f" {line.rstrip()}")
                            
                        if len(current_lines) > preview_lines:
                            diff_lines.append(f" ... ({len(current_lines) - preview_lines} more lines)")
                            
                        diff_lines.append("")
                        
                    except UnicodeDecodeError:
                        diff_lines.extend([
                            f"--- {rel_path}",
                            f"+++ {rel_path} (binary file, will be modified)",
                            f"@@ Binary file @@",
                            "",
                        ])
                else:
                    # File doesn't exist, will be created
                    diff_lines.extend([
                        f"--- /dev/null",
                        f"+++ {rel_path} (new file)",
                        f"@@ -0,0 +1,? @@",
                        f"+New file will be created",
                        "",
                    ])
                    
            except PathEscapeError:
                diff_lines.extend([
                    f"--- {rel_path} (path error)",
                    f"+++ {rel_path} (cannot access)",
                    f"@@ Path outside project root @@",
                    "",
                ])
            except Exception:
                diff_lines.extend([
                    f"--- {rel_path}",
                    f"+++ {rel_path} (will be modified)",
                    f"@@ Cannot preview this file @@",
                    "",
                ])
        
        if len(raw_files) > 10:
            diff_lines.append(f"... and {len(raw_files) - 10} more files")
            
        return "\n".join(diff_lines)
        
    except Exception:
        return None


def execute_code_task(args: dict[str, Any]) -> dict[str, Any]:
    """Hands ``task_description`` + ``files`` to aider in headless one-shot
    mode. MUTATING (manifest.json: ``read_only: false``) — dana.core.
    react_dispatch's generic plugin wiring HITL-gates this exactly like
    execute_terminal_command; this function only ever runs post-approval.

    ``test_command``, when given, is validated against the exact same
    ``_ALLOWED_VERIFY_COMMANDS`` allowlist ``run_verification_command`` uses
    (via ``_validate_verify_command``) and passed through to aider's own
    ``--test-cmd``/``--auto-test`` flags, so aider re-runs it and
    self-repairs any failing traceback inside its own subprocess loop before
    this call returns — collapsing the separate edit-verify-repair ReAct
    turns into this one invocation. An invalid ``test_command`` fails the
    call outright, before aider (or any subprocess) ever runs.
    """
    task_description = str(args.get("task_description") or "").strip()
    if not task_description:
        return {"ok": False, "error": "execute_code_task requires a non-empty task_description"}

    raw_files = args.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return {"ok": False, "error": "execute_code_task requires a non-empty 'files' list"}

    test_command = str(args.get("test_command") or "").strip()
    if test_command:
        _, error = _validate_verify_command(test_command)
        if error:
            return {"ok": False, "error": f"invalid test_command: {error}"}

    resolved_files: list[Path] = []
    for rel_path in raw_files:
        try:
            resolved_files.append(_resolve_repo_path(str(rel_path)))
        except PathEscapeError as exc:
            return {"ok": False, "error": str(exc)}

    command = [
        "aider",
        "--model", "gemini/gemini-3.6-flash",
        # SEARCH/REPLACE blocks, not a whole-file rewrite per edit — "udiff"
        # (unified diff) was tried first, but Gemini Flash's actual diff
        # output doesn't parse/apply reliably under that format: Aider would
        # propose the edit, then silently fail to apply it or commit,
        # reporting success with nothing actually changed. "diff" is Aider's
        # more robust, model-agnostic default and handles Gemini's output
        # correctly.
        "--edit-format", "diff",
        "--yes",  # auto-commit and accept all prompts — this IS headless mode
        "--no-stream",  # disable streaming output — see this plugin's own live verification
    ]
    if test_command:
        command += ["--test-cmd", test_command, "--auto-test"]
    command += ["--message", task_description] + [str(p) for p in resolved_files]

    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, shell=False; task_description is one arg
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_EXECUTE_TIMEOUT_S,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"aider timed out after {_EXECUTE_TIMEOUT_S}s",
            "stdout": _truncate(exc.stdout) if isinstance(exc.stdout, str) else "",
            "stderr": _truncate(exc.stderr) if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "aider is not installed / not on PATH (verified working via `uv tool install aider-chat`)",
        }
    except OSError as exc:
        return {"ok": False, "error": f"could not run aider: {exc}"}

    stdout = _truncate(completed.stdout or "")
    stderr = _truncate(completed.stderr or "")
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": stderr.strip() or f"aider exited with code {completed.returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": completed.returncode,
        }
    return {"ok": True, "stdout": stdout, "stderr": stderr, "returncode": 0}


__all__ = (
    "search_codebase",
    "analyze_codebase",
    "run_verification_command",
    "execute_code_task",
    "PathEscapeError",
)
