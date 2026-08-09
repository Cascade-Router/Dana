"""Stage 9.3 — auto-updater unit tests (subprocess mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_check_for_updates_detects_mismatch(tmp_path: Path) -> None:
    from dana.utils import updater as up

    calls: list[tuple[str, ...]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        args = tuple(cmd)
        calls.append(args)
        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = ""
        proc.returncode = 0
        if args[:2] == ("git", "fetch"):
            return proc
        if args == ("git", "rev-parse", "HEAD"):
            proc.stdout = "aaa111\n"
            return proc
        if args == ("git", "rev-parse", "@{u}"):
            proc.stdout = "bbb222\n"
            return proc
        raise AssertionError(f"unexpected: {args}")

    with patch.object(up.subprocess, "run", side_effect=fake_run):
        assert up.check_for_updates(cwd=tmp_path) is True
    assert any(c[:2] == ("git", "fetch") for c in calls)


def test_check_for_updates_current(tmp_path: Path) -> None:
    from dana.utils import updater as up

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        args = tuple(cmd)
        proc = MagicMock()
        proc.stdout = "samehash\n"
        proc.stderr = ""
        proc.returncode = 0
        if args[:2] == ("git", "fetch"):
            proc.stdout = ""
            return proc
        if "rev-parse" in args:
            return proc
        raise AssertionError(args)

    with patch.object(up.subprocess, "run", side_effect=fake_run):
        assert up.check_for_updates(cwd=tmp_path) is False


def test_check_for_updates_fetch_failure_returns_false(tmp_path: Path) -> None:
    from dana.utils import updater as up

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        raise subprocess.CalledProcessError(1, cmd, stderr="network down")

    with patch.object(up.subprocess, "run", side_effect=fake_run):
        assert up.check_for_updates(cwd=tmp_path) is False


def test_apply_update_pull_conflict_does_not_restart(tmp_path: Path) -> None:
    from dana.utils import updater as up

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        args = tuple(cmd)
        if args[:2] == ("git", "pull"):
            raise subprocess.CalledProcessError(
                1, cmd, stderr="CONFLICT (content): Merge conflict"
            )
        raise AssertionError(args)

    with (
        patch.object(up.subprocess, "run", side_effect=fake_run),
        patch.object(up, "_spawn_new_instance") as spawn,
        patch.object(up.sys, "exit") as exit_fn,
    ):
        result = up.apply_update_and_restart(cwd=tmp_path, restart=True)
        assert result.ok is False
        assert "Update Failed" in result.message
        assert "CONFLICT" in result.stderr
        spawn.assert_not_called()
        exit_fn.assert_not_called()


def test_apply_update_pip_failure_does_not_restart(tmp_path: Path) -> None:
    from dana.utils import updater as up

    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        args = tuple(cmd)
        proc = MagicMock()
        proc.stdout = "ok\n"
        proc.stderr = ""
        proc.returncode = 0
        if args[:2] == ("git", "pull"):
            return proc
        if "pip" in args:
            raise subprocess.CalledProcessError(1, cmd, stderr="Could not find version")
        raise AssertionError(args)

    with (
        patch.object(up.subprocess, "run", side_effect=fake_run),
        patch.object(up, "_spawn_new_instance") as spawn,
        patch.object(up.sys, "exit") as exit_fn,
    ):
        result = up.apply_update_and_restart(cwd=tmp_path, restart=True)
        assert result.ok is False
        assert "Update Failed" in result.message
        spawn.assert_not_called()
        exit_fn.assert_not_called()


def test_settings_tab_has_update_widgets() -> None:
    from dana.core_agent import DanaGUI

    app = DanaGUI()
    try:
        assert app._update_check_btn is not None
        assert "Check for Updates" in str(app._update_check_btn.cget("text"))
        assert app._update_apply_btn is not None
        assert app._update_status_lbl is not None
        app._select_tab("Settings")
        app.update_idletasks()
        # Apply button starts unpacked.
        with pytest.raises(Exception):
            app._update_apply_btn.pack_info()
        app._set_update_available(True)
        app.update_idletasks()
        info = app._update_apply_btn.pack_info()
        assert info.get("side") == "right"
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
