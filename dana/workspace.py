"""CAMGRASPER workspace bootstrap + one-shot legacy migration.

Ensures ``CAMGRASPER/{logs,tracker,sandbox,custom_tools,cursor_handoffs,captures}``
exist (``DONNA_WORKSPACE`` == repo root) and migrates prior artifacts without data loss.
"""



from __future__ import annotations



import os

import shutil

from pathlib import Path

from typing import Any



from dana.paths import (

    BUG_TRACKER_PATH,

    CAPTURES_DIR,

    CURSOR_HANDOFF_DIR,

    CURSOR_HANDOFF_MIRROR_DIR,

    CURSOR_HANDOFF_MIRROR_PATH,

    CURSOR_HANDOFF_PATH,

    CUSTOM_TOOLS_DIR,

    DONNA_WORKSPACE,

    DOCS_DIR,

    GENERAL_TOOLS_DIR,

    LEGACY_DESKTOP_GENERATED_TOOLS_DIR,

    LEGACY_GENERATED_TOOLS_DIR,

    LOGS_DIR,

    PENDING_PATCHES_DIR,

    PROJECT_ROOT,

    REPO_CUSTOM_TOOLS_DIR,

    DONNA_SECURITY_DIR,

    EXECUTION_JAIL_DIR,

    EXECUTION_JAIL_LIBRARY_DIR,

    TASK_QUEUE_PATH,
    TEXT_INJECTION_PATH,

    TRACKER_DIR,

    WORKSPACE_MIGRATION_MARKER,

    WORKSPACE_SUBDIRS,

    ensure_workspace_on_syspath,

)





def _log(msg: str) -> None:

    try:

        from dana.logging import log



        log("Workspace", msg)

    except Exception:

        try:
            from dana.stdio_boot import ensure_stdio

            ensure_stdio()
        except Exception:
            pass
        print(f"[Workspace] {msg}")





def _safe_move_file(src: Path, dst: Path) -> bool:

    """Move ``src`` → ``dst`` if src exists and dst does not. Returns True if moved."""

    if not src.is_file():

        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():

        return False

    try:

        shutil.move(str(src), str(dst))

        return True

    except OSError as exc:

        _log(f"WARNING: could not move {src} → {dst}: {exc}")

        try:

            shutil.copy2(str(src), str(dst))

            return True

        except OSError as exc2:

            _log(f"WARNING: copy fallback failed for {src}: {exc2}")

            return False





def _safe_copy_file(src: Path, dst: Path) -> bool:

    """Copy ``src`` → ``dst`` if src exists and dst does not."""

    if not src.is_file():

        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():

        return False

    try:

        shutil.copy2(str(src), str(dst))

        return True

    except OSError as exc:

        _log(f"WARNING: could not copy {src}: {exc}")

        return False





def _safe_move_tree_files(src_dir: Path, dst_dir: Path, *, pattern: str = "*") -> int:

    """Move files matching ``pattern`` from ``src_dir`` into ``dst_dir``."""

    if not src_dir.is_dir():

        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0

    for path in sorted(src_dir.glob(pattern)):

        if not path.is_file():

            continue

        if path.name.upper() == "README.MD" and dst_dir == PENDING_PATCHES_DIR:

            pass

        if _safe_move_file(path, dst_dir / path.name):

            moved += 1

    return moved





def write_workspace_readme() -> None:

    readme = DONNA_WORKSPACE / "README.md"

    body = """# Donna Workspace (CAMGRASPER)

Repo root is the runtime workspace (`DONNA_WORKSPACE == PROJECT_ROOT`).

| Folder | Contents |
|--------|----------|
| `logs/` | `donna_runtime.log`, `donna_conversation.log` |
| `tracker/` | `bug_tracker.json`, `pending_patches/` |
| `execution_jail/` | Filesystem jail (`task_queue.json`, `library/`, fixture copies). Not the Python package. |
| `dana_security/` | Importable security package + `patch_ledger.md` |
| `custom_tools/` | Sole Tool Forge write/load root (ephemeral; wiped on context reset) |
| `cursor_handoffs/` | `donna_handoff.md` (mirrored into `.cursor/instructions/`) |
| `captures/` | Screen captures from OS computer-use |
| `_archive/` | Unused legacy snapshots (not loaded at runtime) |

Promoted tools: `dana/tools/general/` (Git-tracked). Do not merge `execution_jail/` with `dana_security/`.

"""

    if not readme.is_file():

        readme.write_text(body, encoding="utf-8")

    else:

        # Refresh when still documenting the old generated_tools path.

        try:

            existing = readme.read_text(encoding="utf-8")

            if (
                ("generated_tools/" in existing and "custom_tools/" not in existing)
                or ("sandbox/" in existing and "execution_jail/" not in existing)
                or ("donna_sandbox/" in existing and "dana_security/" not in existing)
            ):

                readme.write_text(body, encoding="utf-8")

        except OSError:

            pass





def ensure_custom_tools_package() -> Path:

    """Create Desktop ``custom_tools/`` with ``__init__.py`` and put it on ``sys.path``."""

    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    init_path = CUSTOM_TOOLS_DIR / "__init__.py"

    if not init_path.is_file():

        init_path.write_text(

            '"""Hot-loaded Tool Forge modules (CAMGRASPER/custom_tools)."""\n',

            encoding="utf-8",

        )

    ensure_workspace_on_syspath()

    return CUSTOM_TOOLS_DIR





def ensure_generated_tools_package() -> Path:

    """Backward-compat alias → ``ensure_custom_tools_package``."""

    return ensure_custom_tools_package()





def ensure_general_tools_package() -> Path:

    """Ensure repo ``dana/tools/general/`` exists with ``__init__.py``."""

    GENERAL_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    init_path = GENERAL_TOOLS_DIR / "__init__.py"

    if not init_path.is_file():

        init_path.write_text(

            '"""Promoted general-purpose tools (Git-tracked)."""\n',

            encoding="utf-8",

        )

    return GENERAL_TOOLS_DIR





def ensure_repo_custom_tools_package() -> Path:

    """Ensure gitignored repo mirror ``dana/tools/custom/`` exists."""

    REPO_CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    init_path = REPO_CUSTOM_TOOLS_DIR / "__init__.py"

    if not init_path.is_file():

        init_path.write_text(

            '"""Repo-local custom tools mirror (gitignored contents)."""\n',

            encoding="utf-8",

        )

    return REPO_CUSTOM_TOOLS_DIR





def ensure_cursor_handoff_mirror() -> None:

    """Keep project ``.cursor/instructions/donna_handoff.md`` linked/copied to workspace."""

    CURSOR_HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    CURSOR_HANDOFF_MIRROR_DIR.mkdir(parents=True, exist_ok=True)

    if not CURSOR_HANDOFF_PATH.is_file():

        return

    mirror = CURSOR_HANDOFF_MIRROR_PATH

    try:

        if mirror.is_symlink() or mirror.exists():

            try:

                if mirror.resolve() == CURSOR_HANDOFF_PATH.resolve():

                    return

            except OSError:

                pass

            try:

                if mirror.is_file() or mirror.is_symlink():

                    mirror.unlink()

            except OSError:

                pass

        try:

            os.symlink(str(CURSOR_HANDOFF_PATH), str(mirror))

            return

        except OSError:

            shutil.copy2(str(CURSOR_HANDOFF_PATH), str(mirror))

    except OSError as exc:

        _log(f"WARNING: cursor handoff mirror failed: {exc}")





def _migrate_desktop_generated_to_custom() -> list[str]:

    """Move legacy ``generated_tools/*.py`` → ``custom_tools/``."""

    moved: list[str] = []

    src = LEGACY_DESKTOP_GENERATED_TOOLS_DIR

    if not src.is_dir():

        return moved

    # Same resolved path when GENERATED_TOOLS_DIR aliases CUSTOM_TOOLS_DIR — skip.

    try:

        if src.resolve() == CUSTOM_TOOLS_DIR.resolve():

            return moved

    except OSError:

        pass

    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.glob("*.py")):

        if path.name == "__init__.py":

            continue

        if _safe_move_file(path, CUSTOM_TOOLS_DIR / path.name):

            moved.append(f"custom_tools/{path.name} (from generated_tools)")

    archive_src = src / "_archive"

    if archive_src.is_dir():

        dest_arch = CUSTOM_TOOLS_DIR / "_archive"

        dest_arch.mkdir(parents=True, exist_ok=True)

        for path in sorted(archive_src.glob("*")):

            if path.is_file() and _safe_move_file(path, dest_arch / path.name):

                moved.append(f"custom_tools/_archive/{path.name}")

    return moved





def migrate_legacy_artifacts() -> dict[str, Any]:

    """One-shot move of prior CAMGRASPER artifacts into ``DONNA_WORKSPACE``.



    Idempotent: skips destinations that already exist. Safe to call every boot;

    marker file records that migration has been attempted.

    """

    report: dict[str, Any] = {"moved": [], "skipped": True}



    if WORKSPACE_MIGRATION_MARKER.is_file():

        pass

    else:

        report["skipped"] = False



    # Logs

    legacy_logs = PROJECT_ROOT / "logs"

    for name in ("donna_runtime.log", "donna_conversation.log"):

        if _safe_move_file(legacy_logs / name, LOGS_DIR / name):

            report["moved"].append(f"logs/{name}")



    # Bug tracker + pending patches

    if _safe_move_file(DOCS_DIR / "bug_tracker.json", BUG_TRACKER_PATH):

        report["moved"].append("tracker/bug_tracker.json")

    n_patches = _safe_move_tree_files(

        DOCS_DIR / "pending_patches", PENDING_PATCHES_DIR, pattern="*"

    )

    if n_patches:

        report["moved"].append(f"tracker/pending_patches/({n_patches} files)")



    # Cursor handoff

    legacy_handoff = PROJECT_ROOT / ".cursor" / "instructions" / "donna_handoff.md"

    if _safe_move_file(legacy_handoff, CURSOR_HANDOFF_PATH):

        report["moved"].append("cursor_handoffs/donna_handoff.md")



    # Desktop generated_tools → custom_tools

    report["moved"].extend(_migrate_desktop_generated_to_custom())



    # In-repo forge tools → Desktop custom_tools

    if LEGACY_GENERATED_TOOLS_DIR.is_dir():

        for path in sorted(LEGACY_GENERATED_TOOLS_DIR.glob("*.py")):

            if path.name == "__init__.py":

                continue

            if _safe_move_file(path, CUSTOM_TOOLS_DIR / path.name):

                report["moved"].append(f"custom_tools/{path.name}")

        stub = LEGACY_GENERATED_TOOLS_DIR / "__init__.py"

        stub_body = (

            '"""Relocated: forged tools now live under CAMGRASPER/custom_tools/.\n\n'

            "See dana.paths.CUSTOM_TOOLS_DIR.\n"

            '"""\n'

        )

        try:

            stub.write_text(stub_body, encoding="utf-8")

        except OSError:

            pass



    # Legacy jail library misplaced under the security package (do NOT move package code).

    legacy_lib = DONNA_SECURITY_DIR / "library"

    n_lib = _safe_move_tree_files(legacy_lib, EXECUTION_JAIL_LIBRARY_DIR, pattern="*.py")

    if n_lib:

        report["moved"].append(f"execution_jail/library/({n_lib} scripts)")



    for name in (

        "sample_notes.txt",

        "project_omega_status.txt",

        "latest_swarm_report.txt",

    ):

        src = DOCS_DIR / name

        if _safe_copy_file(src, EXECUTION_JAIL_DIR / name):

            report["moved"].append(f"execution_jail/{name} (copied)")



    legacy_cap = DOCS_DIR / "last_screen_capture.png"

    if _safe_move_file(legacy_cap, CAPTURES_DIR / "last_screen_capture.png"):

        report["moved"].append("captures/last_screen_capture.png")



    try:

        WORKSPACE_MIGRATION_MARKER.write_text("migrated\n", encoding="utf-8")

    except OSError:

        pass



    return report





def _migrate_renamed_workspace_roots() -> None:
    """Rename pre-collision roots if they still exist on disk."""
    pairs = (
        (PROJECT_ROOT / "sandbox", EXECUTION_JAIL_DIR),
        (PROJECT_ROOT / "donna_sandbox", DONNA_SECURITY_DIR),
    )
    for src, dst in pairs:
        try:
            if not src.is_dir() or src.resolve() == dst.resolve():
                continue
            if dst.exists():
                continue
            src.rename(dst)
        except OSError:
            pass


def ensure_donna_workspace(*, migrate: bool = True) -> Path:

    """Create CAMGRASPER workspace trees, migrate legacy artifacts, prepare forge imports."""

    DONNA_WORKSPACE.mkdir(parents=True, exist_ok=True)

    _migrate_renamed_workspace_roots()

    for sub in WORKSPACE_SUBDIRS:

        sub.mkdir(parents=True, exist_ok=True)

    write_workspace_readme()

    try:

        if not TASK_QUEUE_PATH.is_file():

            TASK_QUEUE_PATH.write_text("[]\n", encoding="utf-8")

        # Legacy flat interceptor — keep empty file for operators; migrate on boot.

        if not TEXT_INJECTION_PATH.is_file():

            TEXT_INJECTION_PATH.write_text("", encoding="utf-8")

        from dana.tools.task_queue import migrate_legacy_input_txt

        migrate_legacy_input_txt()

    except OSError:

        pass

    ensure_custom_tools_package()

    ensure_general_tools_package()

    ensure_repo_custom_tools_package()

    ensure_workspace_on_syspath()



    if migrate:

        report = migrate_legacy_artifacts()

        moved = report.get("moved") or []

        if moved:

            _log(f"Migrated {len(moved)} artifact group(s) → {DONNA_WORKSPACE}")

            for item in moved:

                _log(f"  + {item}")

        else:

            _log(f"Workspace ready at {DONNA_WORKSPACE}")

    else:

        _log(f"Workspace ready at {DONNA_WORKSPACE} (migration skipped)")



    ensure_cursor_handoff_mirror()



    try:

        import dana.tools.sandbox_io as sio



        sio._SANDBOX_READ_ROOT = EXECUTION_JAIL_DIR.resolve()

    except Exception:

        pass



    return DONNA_WORKSPACE


