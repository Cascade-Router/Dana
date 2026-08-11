"""Phase B Milestone 4 — mock-based tests for dana.operators.autocad_engine.

No real AutoCAD instance is required or touched: every test either forces
``DANA_OS_DRY_RUN=1`` (no COM call attempted) or monkeypatches
``get_application``/``win32com.client`` with plain fakes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dana.operators import autocad_engine


@pytest.fixture(autouse=True)
def _clear_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)


def _fake_app(model_space: MagicMock) -> SimpleNamespace:
    doc = SimpleNamespace(ModelSpace=model_space)
    return SimpleNamespace(ActiveDocument=doc, Visible=False)


# ---------------------------------------------------------------------------
# COM connection handling
# ---------------------------------------------------------------------------


def test_get_application_attaches_to_running_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(Visible=False)
    monkeypatch.setattr(autocad_engine.win32com.client, "GetActiveObject", lambda _progid: fake)
    monkeypatch.setattr(autocad_engine.pythoncom, "CoInitialize", lambda: None)

    app = autocad_engine.get_application()

    assert app is fake
    assert fake.Visible is True


def test_get_application_raises_when_not_running_and_start_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_progid: str) -> None:
        raise Exception("no running instance")

    monkeypatch.setattr(autocad_engine.win32com.client, "GetActiveObject", _raise)
    monkeypatch.setattr(autocad_engine.pythoncom, "CoInitialize", lambda: None)

    with pytest.raises(autocad_engine.AutoCADConnectionError, match="not running"):
        autocad_engine.get_application(start_if_missing=False)


def test_get_application_launches_new_instance_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(Visible=False)

    def _raise(_progid: str) -> None:
        raise Exception("no running instance")

    monkeypatch.setattr(autocad_engine.win32com.client, "GetActiveObject", _raise)
    monkeypatch.setattr(autocad_engine.win32com.client, "Dispatch", lambda _progid: fake)
    monkeypatch.setattr(autocad_engine.pythoncom, "CoInitialize", lambda: None)

    app = autocad_engine.get_application(start_if_missing=True)

    assert app is fake
    assert fake.Visible is True


def test_get_active_document_wraps_com_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        @property
        def ActiveDocument(self) -> None:
            raise Exception("RPC server unavailable")

    monkeypatch.setattr(autocad_engine, "get_application", lambda **_kw: _Boom())

    with pytest.raises(autocad_engine.AutoCADConnectionError, match="no active AutoCAD document"):
        autocad_engine.get_active_document()


# ---------------------------------------------------------------------------
# Drawing primitives — tool response handling
# ---------------------------------------------------------------------------


def test_add_line_returns_ok_with_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_line = SimpleNamespace(Handle="1A2B")
    model_space = MagicMock()
    model_space.AddLine.return_value = fake_line
    monkeypatch.setattr(autocad_engine, "get_application", lambda **_kw: _fake_app(model_space))

    result = json.loads(autocad_engine.add_line([0, 0], [10, 10]))

    assert result == {"ok": True, "op": "add_line", "handle": "1A2B", "start": [0, 0], "end": [10, 10]}
    model_space.AddLine.assert_called_once()


def test_add_circle_returns_error_when_autocad_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**_kw: object) -> None:
        raise autocad_engine.AutoCADConnectionError("AutoCAD is not running")

    monkeypatch.setattr(autocad_engine, "get_application", _raise)

    result = json.loads(autocad_engine.add_circle([0, 0], 5))

    assert result["ok"] is False
    assert "AutoCAD is not running" in result["error"]


def test_add_polyline_rejects_fewer_than_two_points() -> None:
    result = json.loads(autocad_engine.add_polyline([[0, 0]]))

    assert result == {"ok": False, "error": "add_polyline requires at least 2 points"}


def test_add_extruded_solid_rejects_fewer_than_three_points() -> None:
    result = json.loads(autocad_engine.add_extruded_solid([[0, 0], [1, 1]], 5))

    assert result["ok"] is False
    assert "at least 3 profile points" in result["error"]


def test_add_extruded_solid_deletes_scratch_polyline_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poly = MagicMock()
    region = MagicMock()
    solid = SimpleNamespace(Handle="55FF")
    model_space = MagicMock()
    model_space.AddLightWeightPolyline.return_value = poly
    model_space.AddRegion.return_value = [region]
    model_space.AddExtrudedSolid.return_value = solid
    monkeypatch.setattr(autocad_engine, "get_application", lambda **_kw: _fake_app(model_space))

    result = json.loads(
        autocad_engine.add_extruded_solid([[0, 0], [10, 0], [10, 10], [0, 10]], 5)
    )

    assert result == {
        "ok": True,
        "op": "add_extruded_solid",
        "handle": "55FF",
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "height": 5,
    }
    assert poly.Closed is True
    poly.Delete.assert_called_once()
    region.Delete.assert_called_once()


# ---------------------------------------------------------------------------
# AutoLISP dispatch — string formatting
# ---------------------------------------------------------------------------


def test_run_autolisp_command_appends_trailing_space_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = MagicMock()
    monkeypatch.setattr(autocad_engine, "get_active_document", lambda **_kw: doc)

    autocad_engine.run_autolisp_command('(command "LINE")')

    doc.SendCommand.assert_called_once_with('(command "LINE") ')


def test_run_autolisp_command_strips_and_appends_single_trailing_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = MagicMock()
    monkeypatch.setattr(autocad_engine, "get_active_document", lambda **_kw: doc)

    autocad_engine.run_autolisp_command("  _ZOOM _E \n")

    doc.SendCommand.assert_called_once_with("_ZOOM _E ")


def test_run_autolisp_command_rejects_empty_string() -> None:
    result = json.loads(autocad_engine.run_autolisp_command("   "))

    assert result == {"ok": False, "error": "run_autolisp_command requires a non-empty command string"}


def test_run_autolisp_command_error_on_com_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kw: object) -> None:
        raise autocad_engine.AutoCADConnectionError("no active AutoCAD document")

    monkeypatch.setattr(autocad_engine, "get_active_document", _raise)

    result = json.loads(autocad_engine.run_autolisp_command("_ZOOM _E"))

    assert result["ok"] is False
    assert "no active AutoCAD document" in result["error"]


# ---------------------------------------------------------------------------
# Dry-run gate — no COM call attempted at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (autocad_engine.add_line, ([0, 0], [1, 1])),
        (autocad_engine.add_circle, ([0, 0], 5)),
        (autocad_engine.add_polyline, ([[0, 0], [1, 1]],)),
        (autocad_engine.add_extruded_solid, ([[0, 0], [1, 0], [1, 1]], 5)),
        (autocad_engine.run_autolisp_command, ("_ZOOM _E",)),
    ],
)
def test_dry_run_never_touches_com(monkeypatch, fn, args) -> None:  # noqa: ANN001
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")

    def _fail_if_called(**_kw: object) -> None:
        raise AssertionError("dry-run must not call into COM")

    monkeypatch.setattr(autocad_engine, "get_application", _fail_if_called)
    monkeypatch.setattr(autocad_engine, "get_active_document", _fail_if_called)

    result = json.loads(fn(*args))

    assert result["ok"] is True
    assert result["dry_run"] is True
