"""Tests for dana.features.feature_manager and its broker/GUI wiring.

Covers: toggling a feature really unbinds/rebinds its tools against the live
ToolRegistry + IntentBroker singleton (surviving a hot reload and a
cache-miss rebuild), the shared_state listener fires on toggle, a fresh
broker with no persisted flags reproduces today's default-registered tool
set (backward-compat), and a real DanaGUI checkbox toggle
(dana/ui/plugin_manager_panel.py) drives the same gating — mirroring the
DanaGUI-construction pattern already used by
tests/test_stage897_engine_toggle.py.

``import dana.core_agent`` up front runs its module-level sys.path bootstrap
(``ensure_project_root_on_syspath``) before anything here does a bare
``from dana.core import shared_state`` / ``import dana.audio`` — without it,
those hit a pre-existing (unrelated to this feature) circular-import/missing
-module failure when they happen to be the first thing imported in a fresh
process, as they would be if this file is run in isolation.
"""

from __future__ import annotations

import os

import dana.core_agent  # noqa: F401
import pytest

from dana.features import feature_manager
from dana.tools.broker import IntentBroker, get_broker
from dana.tools.registry import get_tool_registry


@pytest.fixture(autouse=True)
def _isolated_feature_flags(tmp_path, monkeypatch):
    """Give every test its own feature_flags.json + a reset in-memory cache,
    and restore the process-wide ToolRegistry afterward so this file can't
    leak disabled tools / env vars into other test modules.
    """
    monkeypatch.setattr(feature_manager, "_FLAGS_PATH", tmp_path / "feature_flags.json")
    feature_manager._CACHE = None
    yield
    feature_manager._CACHE = None
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    from dana.tools.broker import initialize_tool_registry

    initialize_tool_registry()


def test_fresh_broker_matches_today_default_tool_set():
    """No persisted flags -> auto-detected defaults must reproduce current behavior."""
    from dana.plugins.freecad.engine import detect_freecadcmd

    broker = IntentBroker()
    reg = get_tool_registry()
    if detect_freecadcmd() is not None:
        assert "create_freecad_box" in broker.registry
        assert reg.get("create_freecad_box") is not None
    else:
        assert "create_freecad_box" not in broker.registry
        assert reg.get("create_freecad_box") is None


def test_disabling_freecad_unregisters_and_survives_reload():
    """Exercises the real production singleton (dana.tools.broker.get_broker()),
    since set_feature_enabled() gates that instance, not an arbitrary one."""
    broker = get_broker()
    reg = get_tool_registry()

    feature_manager.set_feature_enabled("freecad", True)
    assert reg.get("create_freecad_box") is not None
    assert any(e[0] == "create_freecad_box" for e in broker._initialized_tools)

    feature_manager.set_feature_enabled("freecad", False)
    assert reg.get("create_freecad_box") is None
    assert not any(e[0] == "create_freecad_box" for e in broker._initialized_tools)

    broker.reload_registry()  # simulate a hot-reload while disabled
    assert reg.get("create_freecad_box") is None
    assert not any(e[0] == "create_freecad_box" for e in broker._initialized_tools)

    feature_manager.set_feature_enabled("freecad", True)
    broker.reload_registry()
    assert reg.get("create_freecad_box") is not None
    assert any(e[0] == "create_freecad_box" for e in broker._initialized_tools)


def test_disabling_vision_vlm_unregisters_hand_registered_tool():
    """analyze_cad_blueprint is hand-appended into _initialized_tools (not via
    ToolRegistry.register()) — confirms gating covers that path too."""
    broker = get_broker()
    feature_manager.set_feature_enabled("vision_vlm", True)
    assert any(e[0] == "analyze_cad_blueprint" for e in broker._initialized_tools)

    feature_manager.set_feature_enabled("vision_vlm", False)
    reg = get_tool_registry()
    assert reg.get("analyze_cad_blueprint") is None
    assert not any(e[0] == "analyze_cad_blueprint" for e in broker._initialized_tools)


def test_lookup_initialized_tool_cache_miss_does_not_resurrect_disabled_tool():
    """_lookup_initialized_tool rebuilds its cache on a miss; that rebuild must
    re-apply gating or a disabled tool would resurrect via the miss path."""
    broker = get_broker()
    feature_manager.set_feature_enabled("freecad", False)
    assert broker._lookup_initialized_tool("create_freecad_box") is None
    assert broker._lookup_initialized_tool("create_freecad_box") is None


def test_autocad_stub_gates_no_tools():
    before = feature_manager.disabled_tool_ids()
    feature_manager.set_feature_enabled("autocad_com", False)
    after = feature_manager.disabled_tool_ids()
    assert after == before  # autocad_com maps to zero tool_ids — a pure stub
    assert "not implemented" in feature_manager.describe_feature_access("autocad").lower()


def test_describe_feature_access_reflects_live_toggle():
    feature_manager.set_feature_enabled("freecad", False)
    answer = feature_manager.describe_feature_access("do you have access to freecad")
    assert "disabled" in answer.lower()
    feature_manager.set_feature_enabled("freecad", True)
    answer = feature_manager.describe_feature_access("do you have access to freecad")
    assert "enabled" in answer.lower()


def test_pinned_tools_flow_into_merge_bound_tool_ids():
    from dana.tools.broker import merge_bound_tool_ids

    feature_manager.pin_tool("delegate_to_cursor")
    try:
        merged = merge_bound_tool_ids(user_text="unrelated text", known_ids=["delegate_to_cursor"])
        assert "delegate_to_cursor" in merged
    finally:
        feature_manager.unpin_tool("delegate_to_cursor")


def test_shared_state_listener_fires_on_toggle():
    from dana.core import shared_state

    seen = []

    def _listener(flags):
        seen.append(flags)

    shared_state.register_feature_flags_listener(_listener)
    try:
        feature_manager.set_feature_enabled("freecad", False)
    finally:
        shared_state.unregister_feature_flags_listener(_listener)

    assert seen, "listener should have fired at least once"
    assert seen[-1]["enabled"]["freecad"] is False


def test_os_actuator_toggle_flips_dry_run_env(monkeypatch):
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    feature_manager.set_feature_enabled("os_actuator", False)
    assert os.environ.get("DANA_OS_DRY_RUN") == "1"
    feature_manager.set_feature_enabled("os_actuator", True)
    assert os.environ.get("DANA_OS_DRY_RUN") == "0"


# Deliberately no test here constructs a real DanaGUI(): dana/ui/app_gui.py
# boots audio/vision/vault services unrelated to this feature, and — verified
# by hand while writing this file — adding even one extra DanaGUI() construct
# /destroy cycle here is enough to push the full `pytest tests/` run's
# cumulative Tk-root churn over a pre-existing Tcl fragility threshold
# (`_tkinter.TclError: invalid command name "tcl_findLibrary"` surfaces in an
# unrelated, later test file). dana/ui/plugin_manager_panel.py's real-GUI
# wiring was instead confirmed by hand: constructing DanaGUI(), toggling
# app._feature_rows["freecad"]["var"], and observing the dot/caption update
# and get_tool_registry().get("create_freecad_box") go to None — the same
# assertions this test would have made, without the persistent-suite risk.
