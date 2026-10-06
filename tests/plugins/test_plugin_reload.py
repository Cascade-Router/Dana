"""A plugin refresh must re-execute the entry point in place, never swap in a
second module object (which left two copies of dana.plugins.freecad.engine's
state, including its FreeCADCmd lock, after every startup)."""

from __future__ import annotations

import sys

import dana.core.react_dispatch  # noqa: F401 — its import-time refresh_plugin_tools() is the trigger
from dana.plugins import plugin_manager
from dana.plugins.freecad import engine


def test_freecad_engine_is_one_module_after_startup() -> None:
    assert sys.modules["dana.plugins.freecad.engine"] is engine


def test_forced_refresh_reexecutes_into_the_same_module() -> None:
    engine._refresh_probe = "stale"  # type: ignore[attr-defined]
    plugin_manager.load_all_plugins_grouped(force_refresh=True)
    assert sys.modules["dana.plugins.freecad.engine"] is engine
    assert engine._refresh_probe == "stale"  # same module dict, file re-executed into it
    assert callable(engine.create_box)
    del engine._refresh_probe  # type: ignore[attr-defined]
