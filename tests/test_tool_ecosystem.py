"""Focused tests for the custom_tools wipe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def custom_dir(tmp_path, monkeypatch):
    from dana import paths as paths_mod
    from dana.tools import registry as reg_mod

    custom = tmp_path / "custom_tools"
    custom.mkdir()
    (custom / "__init__.py").write_text('"""test custom"""\n', encoding="utf-8")
    general = tmp_path / "general"
    general.mkdir()
    (general / "__init__.py").write_text('"""test general"""\n', encoding="utf-8")

    monkeypatch.setattr(paths_mod, "CUSTOM_TOOLS_DIR", custom)
    monkeypatch.setattr(paths_mod, "GENERATED_TOOLS_DIR", custom)
    monkeypatch.setattr(paths_mod, "GENERAL_TOOLS_DIR", general)
    monkeypatch.setattr(reg_mod, "CUSTOM_TOOLS_DIR", custom)
    monkeypatch.setattr(reg_mod, "GENERAL_TOOLS_DIR", general)

    # Fresh registry singleton per test.
    reg_mod._registry_singleton = None
    yield custom, general
    reg_mod._registry_singleton = None


def test_wipe_custom_tools_deletes_py_keeps_init(custom_dir, monkeypatch):
    custom, _general = custom_dir
    tool = custom / "echo_demo.py"
    tool.write_text(
        "def echo_demo(text: str = '') -> str:\n    return text\n",
        encoding="utf-8",
    )
    assert tool.is_file()

    # Put a fake module in sys.modules
    sys.modules["custom_tools.echo_demo"] = type(sys)("custom_tools.echo_demo")
    sys.modules["custom_tools"] = type(sys)("custom_tools")

    from dana.tools.registry import get_tool_registry, wipe_custom_tools
    from dana.tools.schema import ToolSpec

    reg = get_tool_registry()
    reg.register(
        ToolSpec(id="echo_demo", description_en="demo", description_fa=""),
        source="forge",
        ephemeral=True,
        metadata={"path": str(tool)},
    )

    wiped = wipe_custom_tools(reason="unit_test")
    assert "echo_demo" in wiped
    assert not tool.exists()
    assert (custom / "__init__.py").is_file()
    assert reg.get("echo_demo") is None
    assert "custom_tools.echo_demo" not in sys.modules
