"""Smoke tests for Windows CREATE_NO_WINDOW helpers (no GUI required)."""

from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows-only creationflags helper")
def test_windows_no_window_creationflags_sets_create_no_window() -> None:
    from dana.vault_service import windows_no_window_creationflags

    flags = windows_no_window_creationflags()
    no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    assert flags & no_window


@pytest.mark.skipif(os.name != "nt", reason="Windows-only creationflags helper")
def test_windows_no_window_creationflags_merges_extra() -> None:
    from dana.vault_service import windows_no_window_creationflags

    extra = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
    flags = windows_no_window_creationflags(extra)
    no_window = int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    assert flags & no_window
    assert flags & extra


def test_stop_dana_scripts_exist() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "stop_dana.bat").is_file()
    assert (root / "stop_dana.vbs").is_file()
    assert not (root / "stop_donna.bat").exists()
