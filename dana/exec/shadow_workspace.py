"""Transactional shadow workspaces for REPL / file-mutation staging.

Writes land under ``.dana_scratch/<session_id>/`` first. On success
(``exit_code == 0``) ``commit()`` copies staged files to destinations;
on failure ``rollback()`` discards scratch without touching targets.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from dana.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

SCRATCH_DIRNAME = ".dana_scratch"

_active_shadow: ContextVar["ShadowWorkspace | None"] = ContextVar(
    "dana_active_shadow",
    default=None,
)


def default_scratch_base() -> Path:
    """Repo-root ``.dana_scratch`` (override via ``base_dir`` in tests)."""
    return Path(PROJECT_ROOT).resolve() / SCRATCH_DIRNAME


def get_active_shadow() -> ShadowWorkspace | None:
    """Return the shadow workspace bound for this context (if any)."""
    return _active_shadow.get()


@contextmanager
def bind_shadow_workspace(workspace: ShadowWorkspace | None) -> Iterator[ShadowWorkspace | None]:
    """Bind ``workspace`` for file_editor / python_repl hooks in this context."""
    token = _active_shadow.set(workspace)
    try:
        yield workspace
    finally:
        _active_shadow.reset(token)


class ShadowWorkspace:
    """Isolated staging dir + dest→stage path map for transactional writes."""

    def __init__(
        self,
        session_id: str,
        *,
        base_dir: Path | str | None = None,
    ) -> None:
        sid = str(session_id or "").strip() or "default"
        # Keep path segment safe / portable.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:120]
        self.session_id = safe
        self.base_dir = (
            Path(base_dir).resolve()
            if base_dir is not None
            else default_scratch_base()
        )
        self.scratch_dir = self.base_dir / self.session_id
        # dest.resolve() as posix key → staged Path
        self._map: dict[str, Path] = {}
        self._committed = False
        self._rolled_back = False

    def ensure(self) -> Path:
        """Create scratch session dir; return it."""
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        return self.scratch_dir

    def map_path(self, dest: str | Path) -> Path:
        """Return the staging path for ``dest`` (creates parent dirs)."""
        target = Path(dest).expanduser()
        if not target.is_absolute():
            target = (Path(PROJECT_ROOT).resolve() / target).resolve()
        else:
            target = target.resolve()
        key = target.as_posix()
        if key in self._map:
            return self._map[key]
        self.ensure()
        # Mirror relative structure under scratch when under PROJECT_ROOT.
        root = Path(PROJECT_ROOT).resolve()
        try:
            rel = target.relative_to(root)
            staged = (self.scratch_dir / rel).resolve()
        except ValueError:
            # Outside repo: flatten under scratch by name.
            staged = (self.scratch_dir / "_abs" / target.name).resolve()
        staged.parent.mkdir(parents=True, exist_ok=True)
        self._map[key] = staged
        return staged

    def stage_write(
        self,
        dest: str | Path,
        content: str | bytes,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        """Write ``content`` to the staging path for ``dest``; return staged path."""
        staged = self.map_path(dest)
        if isinstance(content, bytes):
            staged.write_bytes(content)
        else:
            staged.write_text(str(content), encoding=encoding)
        return staged

    def staged_paths(self) -> dict[str, Path]:
        """Copy of dest(posix) → staged Path mapping."""
        return dict(self._map)

    def commit(self) -> list[Path]:
        """Copy all staged files to destinations; then clear scratch.

        Only call when the wrapped execution succeeded (``exit_code == 0``).
        """
        if self._rolled_back:
            raise RuntimeError("ShadowWorkspace already rolled back")
        written: list[Path] = []
        for dest_key, staged in list(self._map.items()):
            dest = Path(dest_key)
            if not staged.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, dest)
            written.append(dest)
        self._committed = True
        self._clear_scratch()
        self._map.clear()
        return written

    def rollback(self) -> None:
        """Discard scratch; leave destination paths untouched."""
        if self._committed:
            raise RuntimeError("ShadowWorkspace already committed")
        self._rolled_back = True
        self._clear_scratch()
        self._map.clear()

    def _clear_scratch(self) -> None:
        if self.scratch_dir.exists():
            try:
                shutil.rmtree(self.scratch_dir)
            except OSError as exc:
                logger.warning("shadow rollback/cleanup failed: %s", exc)


RunnerResult = tuple[int, str]
ShadowRunner = Callable[["ShadowWorkspace"], RunnerResult]


def run_shadow_transaction(
    session_id: str,
    runner: ShadowRunner,
    *,
    base_dir: Path | str | None = None,
) -> tuple[ShadowWorkspace, int, str]:
    """Run ``runner(workspace)``; commit on ``exit_code == 0`` else rollback.

    ``runner`` returns ``(exit_code, observation)``. Exceptions trigger rollback
    and are re-raised after cleanup.
    """
    ws = ShadowWorkspace(session_id, base_dir=base_dir)
    ws.ensure()
    try:
        exit_code, observation = runner(ws)
    except BaseException:
        ws.rollback()
        raise
    code = int(exit_code or 0)
    if code == 0:
        ws.commit()
    else:
        ws.rollback()
    return ws, code, str(observation or "")


def apply_repl_shadow_outcome(
    workspace: ShadowWorkspace | None,
    *,
    exit_code: int | None,
    error: BaseException | str | None = None,
) -> None:
    """Thin hook for python_repl / tools: commit on success, else rollback."""
    if workspace is None:
        return
    if error is not None or (exit_code is not None and int(exit_code) != 0):
        if not workspace._rolled_back and not workspace._committed:
            workspace.rollback()
        return
    if not workspace._committed and not workspace._rolled_back:
        workspace.commit()
