"""Keeps a manual script's CAD runs out of the user's real workspace.

pytest and ``scripts/run_offline_evals.py`` already redirect
``freecad_output/``, ``exports/`` and ``agent_workspace/``. The scripts here
drive the same code by hand and used to write into the real directories:
an engine call with no session set lands in
``freecad_output/sessions/default/Session_Active.FCStd``, the user's own
default session, and a ``/ws/chat`` run saves a chat into their history.

Usage, from a script's ``__main__`` block::

    parser = argparse.ArgumentParser()
    scratch_workspace.add_arguments(parser)
    args = parser.parse_args()
    with scratch_workspace.scratch_workspace("e2e", session_id=args.session_id, keep=args.keep_artifacts):
        ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def add_arguments(parser: argparse.ArgumentParser, *, session_id: bool = True) -> None:
    """``--keep-artifacts``, and ``--session-id`` for scripts that call the
    engine directly (a ``/ws/chat`` run gets its session from the server)."""
    if session_id:
        parser.add_argument("--session-id", help="session id for this run (default: a fresh scratch id)")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="keep the scratch workspace (documents, exports, saved chats) and print where it is",
    )


@contextmanager
def scratch_workspace(name: str, *, session_id: str | None = None, keep: bool = False) -> Iterator[str]:
    """Point every workspace directory at a fresh temp directory and make a
    scratch session current; yields the session id. The temp directory is
    removed afterwards unless ``keep``."""
    from workspace_isolation import import_path_holders, workspace_redirects

    from dana.session_context import get_session_id, set_session_id

    import_path_holders()
    root = Path(tempfile.mkdtemp(prefix=f"dana-{name}-"))
    saved = []
    for module, attr, value in list(workspace_redirects(root)):
        saved.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)
    session = session_id or f"{name}-scratch-{uuid.uuid4().hex[:8]}"
    previous_session = get_session_id()
    set_session_id(session)
    print(f"[{name}] scratch workspace: {root} (session {session})", flush=True)
    try:
        yield session
    finally:
        for module, attr, value in reversed(saved):
            setattr(module, attr, value)
        set_session_id(previous_session)
        if keep:
            print(f"[{name}] artifacts kept in {root}", flush=True)
        else:
            shutil.rmtree(root, ignore_errors=True)
