"""_run_freecad_script's Session_Active.FCStd backup/rollback: a timed-out,
failed, or "successful but empty" run must leave the session document
exactly as it was before the call."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine


@pytest.fixture
def session_doc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "get_freecadcmd_path", lambda **_k: "FreeCADCmd")
    return engine._session_document_path()


def _session_script(path: Path) -> str:
    return (
        "import FreeCAD as App\n"
        + engine._SESSION_OPEN_SNIPPET.format(session_path=str(path), session_doc_name="Session_Active")
        + "doc.recompute()\n"
        + engine._SESSION_SAVE_SNIPPET
        + f'print("{engine._OK_MARKER} path=" + _session_path)\n'
    )


def _fake_run(monkeypatch: pytest.MonkeyPatch, effect: Any) -> None:
    def _run(argv: list[str], **_kwargs: Any) -> Any:
        return effect()

    monkeypatch.setattr(engine.subprocess, "run", _run)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_timeout_mid_save_restores_previous_document(monkeypatch: pytest.MonkeyPatch, session_doc: Path) -> None:
    session_doc.parent.mkdir(parents=True, exist_ok=True)
    session_doc.write_bytes(b"GOOD")

    def _effect() -> Any:
        session_doc.write_bytes(b"TRUNC")
        raise subprocess.TimeoutExpired(cmd="FreeCADCmd", timeout=60)

    _fake_run(monkeypatch, _effect)
    result = engine._run_freecad_script(_session_script(session_doc))
    assert result["ok"] is False
    assert result["session_restored"] is True
    assert "restored" in result["error"]
    assert session_doc.read_bytes() == b"GOOD"


def test_success_that_leaves_an_empty_document_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, session_doc: Path
) -> None:
    session_doc.parent.mkdir(parents=True, exist_ok=True)
    session_doc.write_bytes(b"GOOD")

    def _effect() -> Any:
        session_doc.write_bytes(b"")
        return _completed(stdout=f"{engine._OK_MARKER} path=x")

    _fake_run(monkeypatch, _effect)
    result = engine._run_freecad_script(_session_script(session_doc))
    assert result["ok"] is False
    assert "missing or empty" in result["error"]
    assert session_doc.read_bytes() == b"GOOD"


def test_successful_save_keeps_new_document_and_a_backup_of_the_old_one(
    monkeypatch: pytest.MonkeyPatch, session_doc: Path
) -> None:
    session_doc.parent.mkdir(parents=True, exist_ok=True)
    session_doc.write_bytes(b"OLD")

    def _effect() -> Any:
        session_doc.write_bytes(b"NEW")
        return _completed(stdout=f"{engine._OK_MARKER} path=x")

    _fake_run(monkeypatch, _effect)
    result = engine._run_freecad_script(_session_script(session_doc))
    assert result["ok"] is True
    assert result["session_restored"] is False
    assert session_doc.read_bytes() == b"NEW"
    assert session_doc.with_name(session_doc.name + ".bak").read_bytes() == b"OLD"


def test_failed_first_save_leaves_no_half_written_document(
    monkeypatch: pytest.MonkeyPatch, session_doc: Path
) -> None:
    def _effect() -> Any:
        session_doc.parent.mkdir(parents=True, exist_ok=True)
        session_doc.write_bytes(b"PARTIAL")
        return _completed(returncode=1)

    _fake_run(monkeypatch, _effect)
    result = engine._run_freecad_script(_session_script(session_doc))
    assert result["ok"] is False
    assert not session_doc.exists()


def test_read_only_script_takes_no_backup(monkeypatch: pytest.MonkeyPatch, session_doc: Path) -> None:
    session_doc.parent.mkdir(parents=True, exist_ok=True)
    session_doc.write_bytes(b"GOOD")
    read_only = (
        "import FreeCAD as App\n"
        + engine._SESSION_OPEN_SNIPPET.format(session_path=str(session_doc), session_doc_name="Session_Active")
        + f'print("{engine._OK_MARKER}")\n'
    )
    _fake_run(monkeypatch, lambda: _completed(stdout=engine._OK_MARKER))
    assert engine._run_freecad_script(read_only)["ok"] is True
    assert not session_doc.with_name(session_doc.name + ".bak").exists()
