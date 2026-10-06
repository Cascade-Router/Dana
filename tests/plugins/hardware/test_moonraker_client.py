"""Moonraker client + dispatch_to_printer, against a fake Moonraker.

Every HTTP request goes through ``requests.Session.request``, patched with
``unittest.mock`` to a ``FakeMoonraker`` that answers with payloads shaped like
Moonraker's documented responses and records each call, so the tests can
assert the exact request sequence (and that nothing is sent to a printer
that isn't ready). No network, no printer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import urlsplit

import pytest
import requests
from fastapi.testclient import TestClient

import dana.core.react_dispatch as rd
from dana.api import server as server_module
from dana.platform.mock import MockControlPlane, MockFreeCADEngine
from dana.plugins.hardware import printer_tools
from dana.plugins.hardware.moonraker_client import MoonrakerClient, MoonrakerError
from dana.plugins.hardware.printer_tools import dispatch_to_printer, printer_base_url
from dana.plugins.os import file_system
from dana.tools.schema import ToolCall

PRINTER = "192.168.1.50"
BASE = "http://192.168.1.50:7125"


def _response(status: int, body: Any) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 500: "Internal Server Error"}.get(status, "")
    resp._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    resp.headers["Content-Type"] = "application/json"
    return resp


class FakeMoonraker:
    """Just enough of Moonraker's REST API, with switchable state."""

    def __init__(self) -> None:
        self.klippy_state = "ready"
        self.print_state = "standby"
        self.current_file = ""
        self.calls: list[tuple[str, str]] = []
        self.uploads: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.fail: dict[str, tuple[int, Any]] = {}  # path -> (status, body)
        self.legacy_upload_reply = False

    def __call__(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        parts = urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}" == BASE, url
        path = parts.path
        self.calls.append((method, path))
        self.headers.append(dict(kwargs.get("headers") or {}))
        assert kwargs.get("timeout"), "every request must carry a timeout"
        if path in self.fail:
            return _response(*self.fail[path])

        if (method, path) == ("GET", "/printer/info"):
            return _response(200, {"result": {
                "state": self.klippy_state,
                "state_message": "Printer is ready" if self.klippy_state == "ready" else "Klipper reports: SHUTDOWN",
                "hostname": "voron", "software_version": "v0.12.0", "cpu_info": "Raspberry Pi 4",
            }})
        if (method, path) == ("GET", "/printer/objects/query"):
            objects = parts.query.split("&")
            status: dict[str, Any] = {}
            if "print_stats" in objects:
                status["print_stats"] = {"filename": self.current_file, "state": self.print_state,
                                         "print_duration": 0.0, "message": ""}
            if "heaters" in objects:
                status["heaters"] = {"available_heaters": ["heater_bed", "extruder"],
                                     "available_sensors": ["temperature_sensor raspberry_pi", "heater_bed", "extruder"]}
            if "extruder" in objects:
                status["extruder"] = {"temperature": 24.6, "target": 0.0, "power": 0.0}
            if "heater_bed" in objects:
                status["heater_bed"] = {"temperature": 23.9, "target": 0.0, "power": 0.0}
            return _response(200, {"result": {"eventtime": 1234.5, "status": status}})
        if (method, path) == ("POST", "/server/files/upload"):
            name, fh, _ctype = kwargs["files"]["file"]
            self.uploads.append({"name": name, "content": fh.read(), "data": kwargs.get("data")})
            if self.legacy_upload_reply:
                return _response(201, {"result": name})
            return _response(201, {"result": {
                "item": {"path": name, "root": "gcodes", "modified": 1700000000.0, "size": 42, "permissions": "rw"},
                "print_started": False, "print_queued": False, "action": "create_file",
            }})
        if (method, path) == ("POST", "/printer/print/start"):
            filename = kwargs["json"]["filename"]
            self.started.append(filename)
            self.print_state, self.current_file = "printing", filename
            self.history.insert(0, {"job_id": "000123", "filename": filename, "status": "in_progress"})
            return _response(200, {"result": "ok"})
        if (method, path) == ("GET", "/server/history/list"):
            return _response(200, {"result": {"count": len(self.history), "jobs": self.history[:1]}})
        return _response(404, {"error": {"code": 404, "message": f"Not Found: {path}"}})


@pytest.fixture
def printer() -> Any:
    fake = FakeMoonraker()

    def route(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs["headers"] = {**self.headers, **(kwargs.get("headers") or {})}
        return fake(method, url, **kwargs)

    with mock.patch.object(requests.Session, "request", autospec=True, side_effect=route):
        yield fake


@pytest.fixture
def gcode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", tmp_path.resolve())
    path = tmp_path / "bracket.gcode"
    path.write_text("G28\nG1 X10 Y10 F3000\nM104 S0\n", encoding="utf-8")
    return path


# -- client --------------------------------------------------------------------------


def test_status_of_an_idle_printer_is_ready_with_temperatures(printer: FakeMoonraker) -> None:
    status = MoonrakerClient(BASE).get_status()
    assert status["state"] == "ready"
    assert status["klippy_state"] == "ready" and status["print_state"] == "standby"
    assert status["heaters"] == {"heater_bed": {"temperature": 23.9, "target": 0.0},
                                 "extruder": {"temperature": 24.6, "target": 0.0}}
    assert printer.calls[:2] == [("GET", "/printer/info"), ("GET", "/printer/objects/query")]


@pytest.mark.parametrize(
    ("klippy", "print_state", "expected"),
    [
        ("ready", "printing", "printing"),
        ("ready", "paused", "paused"),
        ("ready", "error", "error"),
        ("ready", "complete", "ready"),
        ("ready", "cancelled", "ready"),
        ("shutdown", "standby", "shutdown"),
        ("startup", "standby", "startup"),
        ("error", "standby", "error"),
    ],
)
def test_status_summarizes_klippy_and_job_state(
    printer: FakeMoonraker, klippy: str, print_state: str, expected: str
) -> None:
    printer.klippy_state, printer.print_state = klippy, print_state
    assert MoonrakerClient(BASE).get_status()["state"] == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_upload_posts_multipart_into_gcodes_root(printer: FakeMoonraker, gcode: Path, legacy: bool) -> None:
    printer.legacy_upload_reply = legacy
    assert MoonrakerClient(BASE).upload_file(gcode) == "bracket.gcode"
    upload = printer.uploads[0]
    assert upload["name"] == "bracket.gcode"
    assert upload["content"] == gcode.read_bytes()
    assert upload["data"] == {"root": "gcodes"}


def test_start_print_sends_the_filename(printer: FakeMoonraker) -> None:
    MoonrakerClient(BASE).start_print("bracket.gcode")
    assert printer.started == ["bracket.gcode"]
    assert printer.calls == [("POST", "/printer/print/start")]


def test_api_key_is_sent_as_x_api_key(printer: FakeMoonraker) -> None:
    MoonrakerClient(BASE, api_key="s3cret").get_info()
    assert printer.headers[0].get("X-Api-Key") == "s3cret"


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (401, {"error": {"code": 401, "message": "Unauthorized"}}, "requires a Moonraker API key"),
        (500, {"error": {"code": 500, "message": "Klippy Disconnected"}}, "HTTP 500 — Klippy Disconnected"),
        (200, b"<html>nginx</html>", "non-Moonraker"),
    ],
)
def test_http_failures_become_moonraker_errors(
    printer: FakeMoonraker, status: int, body: Any, message: str
) -> None:
    printer.fail["/printer/info"] = (status, body)
    with pytest.raises(MoonrakerError, match=message):
        MoonrakerClient(BASE).get_info()


@pytest.mark.parametrize(
    ("exc", "message"),
    [(requests.ConnectionError("refused"), "could not connect"), (requests.Timeout("slow"), "did not answer in time")],
)
def test_network_failures_become_moonraker_errors(exc: Exception, message: str) -> None:
    with mock.patch.object(requests.Session, "request", side_effect=exc):
        with pytest.raises(MoonrakerError, match=message):
            MoonrakerClient(BASE).get_info()


# -- dispatch_to_printer: the strict sequence ------------------------------------------


def test_ready_printer_gets_upload_then_start_and_returns_the_job_id(printer: FakeMoonraker, gcode: Path) -> None:
    result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is True, result
    assert (result["job_id"], result["filename"], result["printer"]) == ("000123", "bracket.gcode", BASE)
    assert result["heaters"]["extruder"]["temperature"] == 24.6
    assert printer.calls == [
        ("GET", "/printer/info"),
        ("GET", "/printer/objects/query"),  # print_stats & heaters
        ("GET", "/printer/objects/query"),  # the heaters' temperatures
        ("POST", "/server/files/upload"),
        ("POST", "/printer/print/start"),
        ("GET", "/server/history/list"),
    ]
    assert printer.started == ["bracket.gcode"]


@pytest.mark.parametrize(
    ("klippy", "print_state", "state"),
    [("ready", "printing", "printing"), ("ready", "paused", "paused"), ("ready", "error", "error"),
     ("shutdown", "standby", "shutdown"), ("startup", "standby", "startup")],
)
def test_busy_or_faulted_printer_aborts_before_sending_anything(
    printer: FakeMoonraker, gcode: Path, klippy: str, print_state: str, state: str
) -> None:
    printer.klippy_state, printer.print_state = klippy, print_state
    printer.current_file = "other_job.gcode"
    result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is False
    assert (result["stage"], result["printer_state"]) == ("status", state)
    assert f"printer is {state}, not ready — nothing was sent" in result["error"]
    assert not any(method == "POST" for method, _ in printer.calls)
    assert printer.uploads == [] and printer.started == []


def test_upload_failure_never_starts_a_print(printer: FakeMoonraker, gcode: Path) -> None:
    printer.fail["/server/files/upload"] = (400, {"error": {"code": 400, "message": "No space left on device"}})
    result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is False and result["stage"] == "upload"
    assert "print not started" in result["error"] and "No space left" in result["error"]
    assert printer.started == []


def test_start_failure_reports_the_file_already_on_the_printer(printer: FakeMoonraker, gcode: Path) -> None:
    printer.fail["/printer/print/start"] = (400, {"error": {"code": 400, "message": "Printer not ready"}})
    result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is False and result["stage"] == "start"
    assert result["uploaded_filename"] == "bracket.gcode"
    assert "was uploaded but the print did not start" in result["error"]


def test_missing_job_id_still_reports_the_started_print(printer: FakeMoonraker, gcode: Path) -> None:
    printer.fail["/server/history/list"] = (404, {"error": {"code": 404, "message": "history component not loaded"}})
    result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is True and result["job_id"] is None
    assert "print started" in result["note"]
    assert printer.started == ["bracket.gcode"]


@pytest.mark.parametrize("unreachable", [requests.ConnectionError("refused"), requests.Timeout("slow")])
def test_unreachable_printer_fails_at_the_status_step(gcode: Path, unreachable: Exception) -> None:
    with mock.patch.object(requests.Session, "request", side_effect=unreachable):
        result = dispatch_to_printer(PRINTER, "bracket.gcode")
    assert result["ok"] is False and result["stage"] == "status"


def test_api_key_comes_from_session_keys_then_env(
    printer: FakeMoonraker, gcode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatch_to_printer(PRINTER, "bracket.gcode", api_keys={"moonraker": "from-session"})
    assert printer.headers[0].get("X-Api-Key") == "from-session"
    printer.calls.clear(); printer.headers.clear(); printer.print_state = "standby"
    monkeypatch.setenv("DANA_MOONRAKER_API_KEY", "from-env")
    dispatch_to_printer(PRINTER, "bracket.gcode")
    assert printer.headers[0].get("X-Api-Key") == "from-env"


# -- validation: nothing leaves the machine ---------------------------------------------


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("192.168.1.50", "http://192.168.1.50:7125"),
        ("10.0.0.7:80", "http://10.0.0.7:80"),
        ("https://voron.local", "https://voron.local:7125"),
        ("fe80::1", "http://[fe80::1]:7125"),
        ("127.0.0.1", "http://127.0.0.1:7125"),
    ],
)
def test_lan_addresses_are_accepted(address: str, expected: str) -> None:
    assert printer_base_url(address) == expected


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "example.com", "http://192.168.1.5/admin", "ftp://192.168.1.5", "http://u:p@192.168.1.5", ""]
)
def test_non_lan_or_malformed_addresses_are_refused(address: str, gcode: Path) -> None:
    with mock.patch.object(requests.Session, "request") as request:
        result = dispatch_to_printer(address, "bracket.gcode")
    assert result["ok"] is False and result["stage"] == "validate"
    request.assert_not_called()


def test_printer_hosts_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_PRINTER_HOSTS", "printer.lab.example, 100.101.1.2")
    assert printer_base_url("printer.lab.example") == "http://printer.lab.example:7125"
    assert printer_base_url("100.101.1.2:7125") == "http://100.101.1.2:7125"  # e.g. Tailscale


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda root: root / "model.stl", "not a G-code file"),
        (lambda root: root / "missing.gcode", "no such file"),
        (lambda root: root.parent / "outside.gcode", "outside the sandbox"),
    ],
)
def test_file_must_be_existing_gcode_inside_the_workspace(gcode: Path, make: Any, message: str) -> None:
    target = make(gcode.parent)
    if target.suffix == ".stl" or target.name == "outside.gcode":
        target.write_text("solid\n", encoding="utf-8")
    with mock.patch.object(requests.Session, "request") as request:
        result = dispatch_to_printer(PRINTER, str(target))
    assert result["ok"] is False and result["stage"] == "validate" and message in result["error"]
    request.assert_not_called()


# -- agent wiring ---------------------------------------------------------------------


def test_tool_is_gated_always_prompts_and_lives_in_the_hardware_domain() -> None:
    assert rd.is_mutating_tool("dispatch_to_printer") is True
    assert "dispatch_to_printer" in rd.ALWAYS_PROMPT_TOOL_IDS
    assert rd._CAPABILITY_TOOL_IDS["hardware"] == frozenset({"dispatch_to_printer"})
    assert "PHYSICAL PRINT" in rd.describe_tool_call(
        ToolCall(tool_id="dispatch_to_printer", arguments={"printer_ip": PRINTER, "gcode_filepath": "a.gcode"})
    )


def test_dispatch_tool_call_threads_mounts_and_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake(printer_ip: str, gcode_filepath: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(printer_ip=printer_ip, gcode_filepath=gcode_filepath, **kwargs)
        return {"ok": True, "printer": BASE, "filename": "a.gcode", "job_id": "1"}

    monkeypatch.setattr(rd, "_hw_dispatch_to_printer", fake)
    monkeypatch.setattr(rd, "_fsm_out_of_order_check", lambda *a, **k: None, raising=False)
    result = rd.dispatch_tool_call(
        ToolCall(tool_id="dispatch_to_printer", arguments={"printer_ip": PRINTER, "gcode_filepath": "a.gcode"}),
        engine=None, control_plane=None, api_keys={"moonraker": "k"}, allowed_mounts=["D:/prints"],
    )
    assert result.ok is True, result.message
    assert seen == {"printer_ip": PRINTER, "gcode_filepath": "a.gcode",
                    "api_keys": {"moonraker": "k"}, "allowed_mounts": ["D:/prints"]}


def test_refused_on_the_public_hf_space(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform import factory as platform_factory

    monkeypatch.setattr(platform_factory, "IS_HF_SPACE", True)
    result = rd._tool_dispatch_to_printer({"printer_ip": PRINTER, "gcode_filepath": "a.gcode"}, None, None)
    assert result["ok"] is False and "disabled in the cloud demo" in result["error"]


# -- server: every call asks, even with auto-approve on ----------------------------------


class _FakeProvider:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


def _drain_until(ws: Any, msg_type: str, limit: int = 30) -> dict[str, Any]:
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


@pytest.mark.timeout(30)  # a regression shows up as a missing prompt, i.e. a blocked receive
def test_every_print_asks_even_with_auto_approve_and_a_prior_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    dispatched: list[str] = []
    monkeypatch.setattr(
        rd, "_hw_dispatch_to_printer",
        lambda ip, path, **_kw: dispatched.append(path) or {"ok": True, "printer": BASE, "filename": path, "job_id": "1"},
    )
    call = lambda name: [ToolCall(tool_id="dispatch_to_printer",  # noqa: E731
                                  arguments={"printer_ip": PRINTER, "gcode_filepath": name})]
    fake = _FakeProvider([call("a.gcode"), "Printing.", call("b.gcode"), "Printing."])
    monkeypatch.setattr(rd, "ModelProvider", lambda **_kw: fake)

    with TestClient(server_module.app).websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["hardware"]})
        ws.send_json({"type": "set_auto_approve", "payload": {"enabled": True}})
        for name in ("a.gcode", "b.gcode"):
            ws.send_json({"text": f"print {name} on my 3d printer at {PRINTER}"})
            approval = _drain_until(ws, "hitl_approval_required")
            assert approval["payload"]["action_name"] == "dispatch_to_printer"
            assert "PHYSICAL PRINT" in approval["payload"]["description"]
            ws.send_json({"type": "hitl_response",
                          "payload": {"request_id": approval["payload"]["request_id"], "approved": True}})
            _drain_until(ws, "tool_dispatch_end")
            _drain_until(ws, "assistant_message")
    assert dispatched == ["a.gcode", "b.gcode"]
