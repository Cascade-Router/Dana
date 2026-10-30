"""The hosted demo (app.py, Gradio) exposes the Parameters tab, the
zero-token rebuild and the macro export, not just chat.

app.py is imported in a subprocess: importing it strips tools out of the
shared registry and tees stdout/stderr, which must not leak into the rest of
the suite. The subprocess drives one scripted chat turn that builds a box on
the mock engine, then calls the new endpoints exactly as the Vercel UI does.
Runs on Gradio 5.x (the Space's version, installed in CI) and 6.x (local dev
venvs); app.py shims the differences. Needs ``spaces``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

gradio = pytest.importorskip("gradio")
pytest.importorskip("spaces")
pytestmark = pytest.mark.skipif(
    gradio.__version__.split(".", 1)[0] not in ("5", "6"),
    reason=f"app.py supports Gradio 5.x and 6.x, not {gradio.__version__}",
)

_DRIVER = r'''
import asyncio, json, sys
from pathlib import Path

root, tmp = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(root), str(root / "tests")]

import app  # noqa: E402 — the Space's own entry point
import dana.api.server as server
import dana.core.react_dispatch as rd
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.tools.schema import ToolCall
from workspace_isolation import workspace_redirects

for module, attr, value in list(workspace_redirects(tmp / "workspace")):
    setattr(module, attr, value)
import dana.plugins.os.file_system as fs
Path(fs._SANDBOX_ROOT).mkdir(parents=True, exist_ok=True)

server.get_cad_engine = lambda: MockFreeCADEngine()
server.get_control_plane = lambda: MockControlPlane()
rd._get_has_plan = lambda *_a, **_k: True
turns = [[ToolCall("create_freecad_box", {"name": "Plate", "length": 60, "width": 40, "height": 5})], "Built the plate."]

class Provider:
    def complete_with_tool_calls(self, messages, *, tools, provider=None, **kwargs):
        turn = turns.pop(0) if turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}

rd.ModelProvider = lambda **_k: Provider()

async def main():
    out = {}
    out["empty_export"] = app._export_macro(None)[0]
    out["empty_rebuild"] = (await app._rebuild({"0.length": 1}, None))[0]
    session = None
    async for step in app._respond("build a 60x40x5 plate", [], None):
        session = step[4]
    out["reply"] = session["chat_history"][-1]["content"] if session.get("chat_history") else None
    table = app._parameters(session)
    out["params"] = table["params"]
    length = next(p for p in table["params"] if p["id"].endswith(".length"))
    result, mesh, session = await app._rebuild({length["id"]: 75}, session)
    out["rebuild"] = result
    out["rebuild_mesh_exists"] = bool(mesh) and Path(mesh).is_file()
    out["bad_rebuild"] = (await app._rebuild({}, session))[0]
    info, macro = app._export_macro(session)
    out["export"] = info
    out["macro_text"] = Path(macro).read_text(encoding="utf-8") if macro else None
    out["macro_in_tmp"] = bool(macro) and Path(macro).resolve().is_relative_to(tmp.resolve())
    api = app.demo.get_api_info()["named_endpoints"]
    out["endpoints"] = {name: [p["parameter_name"] for p in spec["parameters"]] for name, spec in api.items()}
    (tmp / "result.json").write_text(json.dumps(out, default=str), encoding="utf-8")

asyncio.run(main())
'''


@pytest.fixture(scope="module")
def space_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("space")
    driver = tmp / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    env = {
        **__import__("os").environ,
        "DANA_HEADLESS": "true",
        "DANA_OS_DRY_RUN": "1",
        "DANA_CONTEXT_DISTILL": "0",
        "DANA_AUTO_APPROVE": "0",
    }
    env.pop("SPACE_ID", None)
    proc = subprocess.run(
        [sys.executable, str(driver), str(ROOT), str(tmp)],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    result = tmp / "result.json"
    assert proc.returncode == 0 and result.is_file(), proc.stderr[-4000:]
    return json.loads(result.read_text(encoding="utf-8"))


def test_space_exposes_parameters_rebuild_and_macro_endpoints(space_run: dict) -> None:
    endpoints = space_run["endpoints"]
    assert {"/chat", "/artifacts", "/parameters", "/rebuild", "/export_macro"} <= set(endpoints)
    # The JS client passes the overrides under this keyword.
    assert endpoints["/rebuild"] == ["overrides"]
    assert endpoints["/parameters"] == [] and endpoints["/export_macro"] == []


def test_parameters_list_the_chat_built_box(space_run: dict) -> None:
    fields = {p["id"].split(".", 1)[1]: p["value"] for p in space_run["params"]}
    assert (fields["length"], fields["width"], fields["height"]) == (60, 40, 5)


def test_rebuild_runs_on_the_mock_engine_with_zero_tokens(space_run: dict) -> None:
    result = space_run["rebuild"]
    assert result["ok"] is True, result
    assert result["zero_token_rebuilds"] == 1
    assert space_run["rebuild_mesh_exists"]
    lengths = [p["value"] for p in result["parameters"]["params"] if p["id"].endswith(".length")]
    assert lengths == [75]


def test_macro_export_downloads_a_script_with_the_rebuilt_value(space_run: dict) -> None:
    assert space_run["export"]["ok"] is True
    assert space_run["export"]["filename"].endswith(".py")
    assert "75" in space_run["macro_text"]
    assert space_run["macro_in_tmp"]


def test_endpoints_refuse_cleanly_before_anything_was_built(space_run: dict) -> None:
    assert space_run["empty_export"]["ok"] is False
    assert space_run["empty_rebuild"]["ok"] is False
    assert space_run["bad_rebuild"]["ok"] is False
