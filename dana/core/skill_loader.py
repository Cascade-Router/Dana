"""Autonomous Skill Acquisition — loads user-defined "skills" (plain Python
files under ``AGENT_WORKSPACE_DIR/skills/``) as native ReAct tools.

A skill file is a small Python module exposing exactly two things:

    TOOL_SCHEMA = {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

    def run(args: dict) -> dict:
        ...

``TOOL_SCHEMA`` is the FULL OpenAI tool-calling schema (not just the inner
``"function"`` object) — it drops straight into the ``tools=[...]`` array
``dana.core.react_dispatch._llm_tools_schema`` builds, no reshaping needed.
``run`` is called with exactly the arguments dict the model provided, same
as every other tool handler in ``dana.core.react_dispatch.TOOL_HANDLERS``
once wrapped to that uniform ``(arguments, engine, control_plane)`` calling
convention (see ``react_dispatch.refresh_user_skills``/``_wrap_skill_handler``
— this module has no knowledge of that registry at all, by design, so it
stays independently testable).

Security model — two layers, deliberately NOT three:
  1. Every skill file's path is resolved through
     ``dana.plugins.os.file_system.resolve_sandboxed_path`` — the EXACT
     same traversal-rejecting helper the "os_tools" ReAct tools already
     use, reused rather than reimplemented, so a skill can only ever be
     loaded from (or saved to) ``AGENT_WORKSPACE_DIR/skills/``, never an
     arbitrary path.
  2. Beyond that, a skill's ``run()`` is real, unsandboxed Python running
     IN-PROCESS (``compile()``+``exec()`` into a throwaway module
     namespace — see ``_import_skill_module``, deliberately NOT
     ``importlib``'s file-loader path, which caches bytecode in a way
     that goes stale across rapid edits) — no subprocess isolation, no
     timeout, unlike ``dana.plugins.os.process_manager.run_python_script``. The
     safety boundary here is NOT sandboxing the interpreter; it's that
     every skill tool_id is force-added to the HITL mutation gate (see
     ``react_dispatch.refresh_user_skills``) — a human must explicitly
     approve dispatching it, same policy ``run_python_script`` already
     uses for the same reason. This module does not scan skill source for
     "dangerous" imports/calls — that kind of content-blocklisting is weak
     and easy to bypass; the approval step is the real control.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

# Lowercase snake_case only — matches every existing tool_id's naming
# convention in dana/tools/tools.json (search_web, create_freecad_box, ...)
# and doubles as this module's path-safety net for the skill_name itself:
# no '.', '/', or '..' can ever appear in something this pattern accepts,
# so a crafted skill_name can't escape skills/ on its own (resolve_sandboxed_path
# below is still called regardless, as defense in depth, not a substitute).
_VALID_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Namespaces every dynamically-imported skill's synthetic module name so it
# can never collide with (or be confused for) a real installed package —
# these modules are deliberately never registered in sys.modules, so this
# prefix is cosmetic/diagnostic only, not a real isolation mechanism.
_SKILL_MODULE_PREFIX = "dana_user_skill__"


def is_valid_skill_name(skill_name: str) -> bool:
    return bool(_VALID_SKILL_NAME.fullmatch(skill_name or ""))


def validate_tool_schema(skill_name: str, schema: Any) -> str | None:
    """Returns an error string if ``schema`` isn't a valid, self-consistent
    OpenAI tool schema for ``skill_name``; ``None`` if it's fine. Used both
    when saving a brand-new skill (``save_skill``) and when loading an
    existing one back off disk (``load_user_skills``) — the same rules
    apply either way, so a hand-edited/stale file on disk can't sneak past
    validation just because it skipped ``save_skill``.
    """
    if not isinstance(schema, dict):
        return "schema must be an object"
    if schema.get("type") != "function":
        return "schema.type must be 'function'"
    fn = schema.get("function")
    if not isinstance(fn, dict):
        return "schema.function must be an object"
    if fn.get("name") != skill_name:
        return f"schema.function.name ({fn.get('name')!r}) must equal skill_name ({skill_name!r})"
    if not isinstance(fn.get("parameters"), dict):
        return "schema.function.parameters must be an object"
    return None


def save_skill(skill_name: str, python_code: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Writes a new skill to ``skills/<skill_name>.py`` — ``schema``
    becomes that file's ``TOOL_SCHEMA`` assignment (serialized as literal
    JSON, so it can never desync from the validated dict the caller
    passed), and ``python_code`` is appended verbatim below it (expected to
    define ``run(args)``, checked with a cheap substring probe here —
    ``load_user_skills`` below does the REAL verification, actually
    importing the file and checking ``run`` is callable).

    Does not itself refresh any tool registry — see
    ``dana.core.react_dispatch.refresh_user_skills``, called separately
    (by the ``save_new_skill`` tool) right after a successful save.
    """
    skill_name = (skill_name or "").strip()
    if not is_valid_skill_name(skill_name):
        return {
            "ok": False,
            "error": f"invalid skill_name {skill_name!r} — must be lowercase snake_case, e.g. 'convert_csv_to_json'",
        }
    schema_error = validate_tool_schema(skill_name, schema)
    if schema_error:
        return {"ok": False, "error": schema_error}
    if not python_code.strip():
        return {"ok": False, "error": "python_code must not be empty"}
    if "def run(" not in python_code:
        return {"ok": False, "error": "python_code must define a top-level 'def run(args):' function"}

    try:
        target = resolve_sandboxed_path(f"skills/{skill_name}.py")
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}

    file_contents = (
        f'"""Auto-generated user skill: {skill_name}. Written by the save_new_skill tool — see\n'
        f'dana.core.skill_loader for how TOOL_SCHEMA/run() are loaded back."""\n\n'
        f"TOOL_SCHEMA = {json.dumps(schema, indent=2)}\n\n\n"
        f"{python_code.rstrip()}\n"
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_contents, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"could not write skill file: {exc}"}
    return {"ok": True, "skill_name": skill_name, "path": f"skills/{skill_name}.py"}


def delete_skill(skill_name: str) -> dict[str, Any]:
    """Deletes ``skills/<skill_name>.py`` if present, through the SAME
    ``resolve_sandboxed_path`` traversal check ``save_skill``/
    ``load_user_skills`` use — no separate path logic to drift out of
    sync. Idempotent: deleting an already-absent skill is ``{"ok": True,
    "deleted": False}``, not an error (matches ``dana.api.sessions.
    delete_session``'s same idiom) — a double-click in the frontend's
    SkillsPlugin, or the agent's own ``delete_skill`` tool naming a skill
    that's already gone, must not surface as a failure.

    Does not itself refresh any tool registry — see
    ``dana.core.react_dispatch.refresh_user_skills``, called separately
    right after, by both the ``delete_skill`` tool and ``dana.api.skills``'s
    ``DELETE`` endpoint.
    """
    skill_name = (skill_name or "").strip()
    if not is_valid_skill_name(skill_name):
        return {"ok": False, "error": f"invalid skill_name {skill_name!r}"}
    try:
        target = resolve_sandboxed_path(f"skills/{skill_name}.py")
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        target.unlink()
        return {"ok": True, "skill_name": skill_name, "deleted": True}
    except FileNotFoundError:
        return {"ok": True, "skill_name": skill_name, "deleted": False}
    except OSError as exc:
        return {"ok": False, "error": f"could not delete skill file: {exc}"}


def read_skill_source(skill_name: str) -> str | None:
    """Returns the raw Python source of ``skills/<skill_name>.py``, or
    ``None`` if it's an invalid name, escapes the sandbox, or doesn't
    exist/can't be read — used by ``dana.api.skills``'s ``GET /api/skills``
    to show the frontend SkillsPlugin the actual on-disk source for each
    currently-loaded skill (never raises, so one unreadable file can't
    break the whole listing).
    """
    if not is_valid_skill_name(skill_name):
        return None
    try:
        target = resolve_sandboxed_path(f"skills/{skill_name}.py")
    except PathEscapeError:
        return None
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        return None


def write_skill_source(skill_name: str, raw_code: str) -> dict[str, Any]:
    """Overwrites ``skills/<skill_name>.py`` with ``raw_code`` VERBATIM —
    no ``TOOL_SCHEMA`` wrapping the way ``save_skill`` does for a
    brand-new skill the ``save_new_skill`` tool creates. This is the
    USER's manual-edit path (``dana.api.skills``'s ``PUT`` endpoint): the
    frontend SkillsPlugin edits the FULL file content it got back from
    ``GET /api/skills`` (already includes ``TOOL_SCHEMA``/``run()``), so
    writing it back verbatim is exactly what "save my edits" means here.

    Does not itself validate or refresh any tool registry — see
    ``validate_skill_file``/``dana.core.react_dispatch.refresh_user_skills``,
    called separately right after by the ``PUT`` endpoint.
    """
    skill_name = (skill_name or "").strip()
    if not is_valid_skill_name(skill_name):
        return {"ok": False, "error": f"invalid skill_name {skill_name!r}"}
    if not raw_code.strip():
        return {"ok": False, "error": "code must not be empty"}
    try:
        target = resolve_sandboxed_path(f"skills/{skill_name}.py")
    except PathEscapeError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw_code, encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"could not write skill file: {exc}"}
    return {"ok": True, "skill_name": skill_name, "path": f"skills/{skill_name}.py"}


def validate_skill_file(skill_name: str) -> str | None:
    """Attempts to (re)import ``skills/<skill_name>.py`` fresh and
    validate its ``TOOL_SCHEMA``/``run()`` — returns a SPECIFIC error
    string describing exactly what's wrong (a ``SyntaxError``'s own
    message, a missing ``run()``, a bad schema shape, ...), or ``None`` if
    it loads and validates cleanly. Used by ``dana.api.skills``'s ``PUT``
    endpoint right after ``write_skill_source`` to give the frontend a
    precise, actionable error instead of a generic "failed to import".
    """
    if not is_valid_skill_name(skill_name):
        return f"invalid skill_name {skill_name!r}"
    try:
        target = resolve_sandboxed_path(f"skills/{skill_name}.py")
    except PathEscapeError as exc:
        return str(exc)
    if not target.is_file():
        return f"no skill file found at skills/{skill_name}.py"

    module, import_error = _import_skill_module(skill_name, target)
    if module is None:
        return import_error
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return "missing a callable run(args) function"
    return validate_tool_schema(skill_name, getattr(module, "TOOL_SCHEMA", None))


def _import_skill_module(skill_name: str, path: Path) -> tuple[ModuleType | None, str | None]:
    """Compiles and executes ``path``'s source directly into a fresh,
    throwaway module namespace, never registered in ``sys.modules``.

    Deliberately NOT ``importlib.util.spec_from_file_location`` +
    ``exec_module`` (this module's original approach) — that path's
    default ``SourceFileLoader`` consults (and writes) a ``__pycache__``
    bytecode cache keyed on the source file's mtime. Two rapid overwrites
    of the SAME skill file — exactly the ``save_new_skill``/``PUT``-edit
    hot-reload flow this whole module exists for — can land within the
    same mtime tick on some filesystems, and that loader will then
    silently reuse STALE bytecode compiled from the PREVIOUS version
    (confirmed empirically: a skill edited and immediately re-dispatched
    kept running its old code). Reading the source text and
    ``compile()``+``exec()``-ing it fresh every call has no such cache to
    go stale — always reflects exactly what's on disk right now.

    Returns ``(module, None)`` on success, or ``(None, error)`` on any
    read/compile/exec-time failure — never raises, so a syntax error or
    raised exception in one skill file can't take down a whole reload.
    ``error`` is the ACTUAL exception's own message (e.g. a
    ``SyntaxError``'s text and line number), not a generic placeholder —
    ``validate_skill_file`` below (dana.api.skills's ``PUT`` endpoint)
    surfaces it directly to the frontend so a hand-edit's mistake is
    immediately actionable.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"

    module = ModuleType(f"{_SKILL_MODULE_PREFIX}{skill_name}")
    module.__file__ = str(path)
    try:
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)  # noqa: S102 — this IS the skill execution; see module docstring's security model
    except Exception as exc:  # noqa: BLE001 — the caller needs the SPECIFIC reason, not just "it failed"
        return None, f"{type(exc).__name__}: {exc}"
    return module, None


def load_user_skills() -> dict[str, Any]:
    """Scans ``skills/`` for ``.py`` files and safely imports each one.

    Returns ``{"skills": {tool_id: {"schema": ..., "handler": ...}}, "skipped": [{"file", "reason"}, ...]}``
    — every file that fails ANY validation step (bad filename, escapes the
    sandbox, fails to import, missing TOOL_SCHEMA/run, schema doesn't
    validate) is recorded in ``skipped`` with why, rather than raising —
    one bad skill file must never prevent every other already-working one
    from loading.
    """
    loaded: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    try:
        skill_dir = resolve_sandboxed_path("skills")
    except PathEscapeError:
        return {"skills": loaded, "skipped": skipped}
    if not skill_dir.is_dir():
        return {"skills": loaded, "skipped": skipped}

    for path in sorted(skill_dir.glob("*.py")):
        skill_name = path.stem
        if skill_name.startswith("_"):
            continue  # e.g. __init__.py / __pycache__ leftovers — never a real skill
        if not is_valid_skill_name(skill_name):
            skipped.append({"file": path.name, "reason": "filename is not a valid skill identifier"})
            continue
        # Re-resolve THIS specific file through the sandbox check again,
        # right before importing it — defense in depth against a symlink
        # swapped in between the glob() above and the import below.
        try:
            resolve_sandboxed_path(f"skills/{path.name}")
        except PathEscapeError:
            skipped.append({"file": path.name, "reason": "resolves outside the sandboxed skills directory"})
            continue

        module, import_error = _import_skill_module(skill_name, path)
        if module is None:
            skipped.append({"file": path.name, "reason": import_error or "failed to import"})
            continue

        schema = getattr(module, "TOOL_SCHEMA", None)
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            skipped.append({"file": path.name, "reason": "missing a callable run(args) function"})
            continue
        schema_error = validate_tool_schema(skill_name, schema)
        if schema_error:
            skipped.append({"file": path.name, "reason": schema_error})
            continue

        loaded[skill_name] = {"schema": schema, "handler": run_fn}

    return {"skills": loaded, "skipped": skipped}


__all__ = (
    "delete_skill",
    "is_valid_skill_name",
    "load_user_skills",
    "read_skill_source",
    "save_skill",
    "validate_skill_file",
    "validate_tool_schema",
    "write_skill_source",
)
