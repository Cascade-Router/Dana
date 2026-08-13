"""Driver resolution — dana.platform.factory.get_control_plane()/get_cad_engine().

Each ``factory.IS_*`` flag is monkeypatched directly on the module rather than
via env vars / ``sys.platform``, so every branch is exercised regardless of
which OS actually runs this suite (e.g. ``MacOSControlPlane`` resolution is
tested the same way on Windows CI as on a real Mac).
"""

from __future__ import annotations

import importlib

import pytest

from dana.platform import factory
from dana.platform.darwin import MacOSControlPlane
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.platform.win32 import RealFreeCADEngine, Win32ControlPlane


def test_get_control_plane_resolves_mock_when_hf_space(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "IS_HF_SPACE", True)
    monkeypatch.setattr(factory, "IS_WINDOWS", True)
    monkeypatch.setattr(factory, "IS_MAC", False)

    assert isinstance(factory.get_control_plane(), MockControlPlane)
    assert isinstance(factory.get_cad_engine(), MockFreeCADEngine)


def test_get_control_plane_resolves_mock_on_non_windows_non_mac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "IS_HF_SPACE", False)
    monkeypatch.setattr(factory, "IS_WINDOWS", False)
    monkeypatch.setattr(factory, "IS_MAC", False)

    assert isinstance(factory.get_control_plane(), MockControlPlane)
    assert isinstance(factory.get_cad_engine(), MockFreeCADEngine)


def test_get_control_plane_resolves_win32_when_windows_and_not_hf_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "IS_HF_SPACE", False)
    monkeypatch.setattr(factory, "IS_WINDOWS", True)
    monkeypatch.setattr(factory, "IS_MAC", False)

    assert isinstance(factory.get_control_plane(), Win32ControlPlane)
    assert isinstance(factory.get_cad_engine(), RealFreeCADEngine)


def test_get_control_plane_resolves_macos_stub_on_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "IS_HF_SPACE", False)
    monkeypatch.setattr(factory, "IS_WINDOWS", False)
    monkeypatch.setattr(factory, "IS_MAC", True)

    assert isinstance(factory.get_control_plane(), MacOSControlPlane)
    # No macOS CAD driver exists yet — falls back to the mock engine rather
    # than assuming dana.plugins.freecad.engine's Windows-oriented binary
    # discovery works unmodified off Windows.
    assert isinstance(factory.get_cad_engine(), MockFreeCADEngine)


def test_hf_space_takes_priority_over_windows_and_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPACE_ID must win even if the host also looks like Windows or macOS
    (e.g. a container image reporting sys.platform == "win32")."""
    monkeypatch.setattr(factory, "IS_HF_SPACE", True)
    monkeypatch.setattr(factory, "IS_WINDOWS", True)
    monkeypatch.setattr(factory, "IS_MAC", True)

    assert isinstance(factory.get_control_plane(), MockControlPlane)
    assert isinstance(factory.get_cad_engine(), MockFreeCADEngine)


def test_macos_control_plane_methods_are_unimplemented_stubs() -> None:
    control_plane = MacOSControlPlane()
    with pytest.raises(NotImplementedError):
        control_plane.resync_workspace()
    with pytest.raises(NotImplementedError):
        control_plane.prevent_focus_steal()
    with pytest.raises(NotImplementedError):
        control_plane.get_active_display()


def test_is_hf_space_reflects_space_id_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end check of the actual detection logic (not just the
    downstream branching above) — SPACE_ID presence flips IS_HF_SPACE on
    module (re)load."""
    monkeypatch.setenv("SPACE_ID", "some-space")
    try:
        reloaded = importlib.reload(factory)
        assert reloaded.IS_HF_SPACE is True
    finally:
        monkeypatch.delenv("SPACE_ID", raising=False)
        importlib.reload(factory)  # restore ambient (no SPACE_ID) state for later tests
