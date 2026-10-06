"""Printer watchdog + the fail-safe pause_print/emergency_stop tools, against a
fake Moonraker.

The watchdog drives the real MoonrakerClient; ``requests.Session.request`` is
patched to a ``FakeMoonraker`` with switchable Klippy/job/heater/webcam state
that records every request, so the tests assert what was actually sent to the
printer (an e-stop, a pause, or nothing). No network, no printer, no model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
from dana.plugins.hardware import watchdog as watchdog_module
from dana.plugins.hardware.moonraker_client import MoonrakerClient, MoonrakerError
from dana.plugins.hardware.printer_tools import emergency_stop, pause_print
from dana.plugins.hardware.watchdog import PrinterWatchdog
from dana.tools.schema import ToolCall

PRINTER = "192.168.1.50"
BASE = "http://192.168.1.50:7125"
SNAPSHOT = "http://192.168.1.50/webcam/?action=snapshot"
FIXED_NOW = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

E_STOP = ("POST", "/printer/emergency_stop")
PAUSE = ("POST", "/printer/print/pause")


def _response(status: int, body: Any, content_type: str = "application/json") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    resp.headers["Content-Type"] = content_type
    return resp


class FakeMoonraker:
    def __init__(self) -> None:
        self.klippy_state = "ready"
        self.print_state = "standby"
        self.heaters: dict[str, dict[str, float]] = {
            "extruder": {"temperature": 25.0, "target": 0.0},
            "heater_bed": {"temperature": 24.0, "target": 0.0},
        }
        self.webcams: list[dict[str, Any]] = [
            {"name": "cam", "enabled": True, "snapshot_url": "/webcam/?action=snapshot", "stream_url": "/webcam/"}
        ]
        self.snapshot = (b"\xff\xd8jpeg", "image/jpeg")
        self.objects_query_fails = False
        self.unreachable = False
        self.control_reply: Any = "ok"
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        assert kwargs.get("timeout"), "every request must carry a timeout"
        parts = urlsplit(url)
        if self.unreachable:
            raise requests.ConnectionError("no route to host")
        if url.split("?")[0] == SNAPSHOT.split("?")[0]:
            self.calls.append((method, "snapshot"))
            return _response(200, self.snapshot[0], self.snapshot[1])
        assert f"{parts.scheme}://{parts.netloc}" == BASE, url
        self.calls.append((method, parts.path))
        if (method, parts.path) == ("GET", "/printer/info"):
            return _response(200, {"result": {"state": self.klippy_state, "state_message": f"klippy {self.klippy_state}"}})
        if (method, parts.path) == ("GET", "/printer/objects/query"):
            if self.objects_query_fails:
                return _response(503, {"error": {"code": 503, "message": "Klippy Host not connected"}})
            objects = parts.query.split("&")
            status: dict[str, Any] = {}
            if "print_stats" in objects:
                status["print_stats"] = {"state": self.print_state, "filename": "part.gcode"}
            if "heaters" in objects:
                status["heaters"] = {"available_heaters": list(self.heaters)}
            for name, reading in self.heaters.items():
                if name in objects:
                    status[name] = dict(reading)
            return _response(200, {"result": {"status": status}})
        if (method, parts.path) in (E_STOP, PAUSE):
            return _response(200, {"result": self.control_reply})
        if (method, parts.path) == ("GET", "/server/webcams/list"):
            return _response(200, {"result": {"webcams": self.webcams}})
        return _response(404, {"error": {"code": 404, "message": f"Not Found: {parts.path}"}})

    def count(self, call: tuple[str, str]) -> int:
        return self.calls.count(call)


@pytest.fixture
def printer() -> Any:
    fake = FakeMoonraker()

    def route(self: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
        return fake(method, url, **kwargs)

    with mock.patch.object(requests.Session, "request", autospec=True, side_effect=route):
        yield fake


class Frames:
    """A scripted frame classifier: returns the next verdict each call."""

    def __init__(self, *verdicts: str | None) -> None:
        self.verdicts = list(verdicts)
        self.seen: list[tuple[bytes, str]] = []

    def __call__(self, image: bytes, mime: str) -> str | None:
        self.seen.append((image, mime))
        verdict = self.verdicts.pop(0)
        if verdict == "raise":
            raise RuntimeError("model crashed")
        return verdict


def _watchdog(tmp_path: Path, classify: Any = None, **kwargs: Any) -> PrinterWatchdog:
    return PrinterWatchdog(
        MoonrakerClient(BASE), log_path=tmp_path / "watchdog.jsonl", classify_frame=classify, now=lambda: FIXED_NOW,
        **kwargs,
    )


def _log(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "watchdog.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- client + tools ---------------------------------------------------------------------


def test_client_pause_and_emergency_stop_post_the_documented_endpoints(printer: FakeMoonraker) -> None:
    client = MoonrakerClient(BASE)
    client.pause_print()
    client.emergency_stop()
    assert printer.calls == [PAUSE, E_STOP]
    printer.control_reply = "nope"
    with pytest.raises(MoonrakerError, match="unexpected reply"):
        client.emergency_stop()


def test_client_reads_webcams_and_fetches_a_raw_snapshot(printer: FakeMoonraker) -> None:
    client = MoonrakerClient(BASE)
    assert client.list_webcams()[0]["snapshot_url"] == "/webcam/?action=snapshot"
    assert client.fetch_image(SNAPSHOT) == (b"\xff\xd8jpeg", "image/jpeg")
    printer.snapshot = (b"<html>", "text/html")
    with pytest.raises(MoonrakerError, match="not an image"):
        client.fetch_image(SNAPSHOT)


@pytest.mark.parametrize(("tool", "call"), [(pause_print, PAUSE), (emergency_stop, E_STOP)])
def test_stop_tools_send_one_request_and_report_it(printer: FakeMoonraker, tool: Any, call: tuple[str, str]) -> None:
    result = tool(PRINTER)
    assert result == {"ok": True, "printer": BASE, "action": tool.__name__}
    assert printer.calls == [call]


def test_stop_tools_refuse_non_lan_printers_and_report_printer_errors(printer: FakeMoonraker) -> None:
    refused = emergency_stop("8.8.8.8")
    assert refused["ok"] is False and "not a local-network address" in refused["error"]
    assert printer.calls == []
    printer.unreachable = True
    failed = pause_print(PRINTER)
    assert failed["ok"] is False and "could not connect" in failed["error"]


# -- watchdog: Klipper faults -------------------------------------------------------------


def test_klipper_shutdown_triggers_one_emergency_stop_and_a_json_log(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path)
    watchdog.poll_once()
    assert printer.count(E_STOP) == 0 and _log(tmp_path) == []

    printer.klippy_state = "shutdown"
    for _ in range(3):
        watchdog.poll_once()
    assert printer.count(E_STOP) == 1  # once per fault, not once per poll
    (record,) = _log(tmp_path)
    assert record == {
        "timestamp": "2026-10-06T12:00:00+00:00",
        "event": "klipper_fault",
        "printer": BASE,
        "threshold": "Klipper host state is 'shutdown'",
        "detail": "klippy shutdown",
        "action": "emergency_stop",
        "action_result": "ok",
        "recommended_intervention": record["recommended_intervention"],
    }
    assert "FIRMWARE_RESTART" in record["recommended_intervention"]

    printer.klippy_state = "ready"
    watchdog.poll_once()
    printer.klippy_state = "error"
    watchdog.poll_once()
    assert printer.count(E_STOP) == 2  # a new fault after recovery is acted on again
    assert [r["event"] for r in _log(tmp_path)] == ["klipper_fault", "klipper_fault_cleared", "klipper_fault"]


def test_print_stats_error_is_a_fault_too(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.print_state = "error"
    _watchdog(tmp_path).poll_once()
    assert printer.count(E_STOP) == 1
    assert _log(tmp_path)[0]["threshold"] == "Klipper print_stats state is 'error'"


def test_shutdown_is_caught_even_when_the_objects_query_fails(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.klippy_state = "shutdown"
    printer.objects_query_fails = True
    _watchdog(tmp_path).poll_once()
    assert printer.count(E_STOP) == 1


def test_failed_emergency_stop_is_recorded(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.klippy_state = "shutdown"
    printer.control_reply = {"unexpected": True}
    _watchdog(tmp_path).poll_once()
    assert _log(tmp_path)[0]["action_result"].startswith("failed: ")


def test_unreachable_printer_is_logged_once_and_never_stopped(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path)
    printer.unreachable = True
    assert watchdog.poll_once() is None
    assert watchdog.poll_once() is None
    printer.unreachable = False
    watchdog.poll_once()
    assert [r["event"] for r in _log(tmp_path)] == ["printer_unreachable", "printer_reachable"]
    assert printer.count(E_STOP) == 0


def test_log_timestamps_are_utc_from_the_clock_not_a_model(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.klippy_state = "shutdown"
    PrinterWatchdog(MoonrakerClient(BASE), log_path=tmp_path / "watchdog.jsonl", classify_frame=None).poll_once()
    stamp = datetime.fromisoformat(_log(tmp_path)[0]["timestamp"])
    assert stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0
    assert abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 60


# -- watchdog: temperature drift (log only) -----------------------------------------------


def _set(printer: FakeMoonraker, temperature: float, target: float = 210.0) -> None:
    printer.heaters["extruder"] = {"temperature": temperature, "target": target}


def test_heat_up_and_small_fluctuations_are_not_drift(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path)
    for temperature in (25.0, 120.0, 190.0, 206.0, 214.5, 205.5, 210.0):
        _set(printer, temperature)
        watchdog.poll_once()
    assert _log(tmp_path) == []


def test_drift_after_reaching_target_is_logged_once_with_no_action(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path)
    for temperature in (25.0, 209.0, 199.0, 198.0, 197.0, 208.0):
        _set(printer, temperature)
        watchdog.poll_once()
    drift, recovered = _log(tmp_path)
    assert drift["event"] == "temperature_drift" and drift["action"] is None
    assert drift["threshold"] == "extruder: |temperature - target| > 5C after reaching target"
    assert drift["detail"] == {"heater": "extruder", "temperature": 199.0, "target": 210.0, "drift": -11.0}
    assert recovered["event"] == "temperature_recovered"
    assert printer.count(E_STOP) == 0 and printer.count(PAUSE) == 0


def test_a_new_target_restarts_the_heat_up_grace(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path)
    for temperature, target in ((210.0, 210.0), (210.0, 240.0), (225.0, 240.0), (240.0, 240.0), (20.0, 0.0)):
        _set(printer, temperature, target)
        watchdog.poll_once()
    assert _log(tmp_path) == []


# -- watchdog: spaghetti (3 consecutive frames) -------------------------------------------


def test_three_consecutive_spaghetti_frames_pause_the_print(printer: FakeMoonraker, tmp_path: Path) -> None:
    frames = Frames("spaghetti", "spaghetti", "spaghetti")
    watchdog = _watchdog(tmp_path, frames)
    assert [watchdog.check_frame_once() for _ in range(2)] == ["spaghetti", "spaghetti"]
    assert printer.count(PAUSE) == 0 and _log(tmp_path) == []
    watchdog.check_frame_once()
    assert printer.count(PAUSE) == 1 and printer.count(E_STOP) == 0
    (record,) = _log(tmp_path)
    assert record["event"] == "spaghetti_detected"
    assert record["threshold"] == "3 consecutive webcam frames classified as spaghetti"
    assert record["action"] == "pause_print" and record["action_result"] == "ok"
    assert frames.seen[0] == (b"\xff\xd8jpeg", "image/jpeg")
    assert watchdog.spaghetti_streak == 0


def test_a_clean_frame_resets_the_streak(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path, Frames("spaghetti", "spaghetti", "clean", "spaghetti", "spaghetti"))
    for _ in range(5):
        watchdog.check_frame_once()
    assert printer.count(PAUSE) == 0
    assert watchdog.spaghetti_streak == 2


def test_unclassifiable_frames_neither_count_nor_reset(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path, Frames("spaghetti", None, "spaghetti", "raise", "spaghetti"))
    verdicts = [watchdog.check_frame_once() for _ in range(5)]
    assert verdicts == ["spaghetti", None, "spaghetti", None, "spaghetti"]
    assert printer.count(PAUSE) == 1


def test_frames_are_only_checked_while_printing_and_at_the_snapshot_interval(
    printer: FakeMoonraker, tmp_path: Path
) -> None:
    clock = [1000.0]
    frames = Frames(*["spaghetti"] * 10)
    watchdog = _watchdog(tmp_path, frames, snapshot_interval=30.0, monotonic=lambda: clock[0])

    watchdog.tick()  # standby: no frame
    assert frames.seen == []

    printer.print_state = "printing"
    for step in (0.0, 5.0, 10.0, 30.0, 35.0):  # frames at t=1000 and t=1030 only
        clock[0] = 1000.0 + step
        watchdog.tick()
    assert len(frames.seen) == 2 and watchdog.spaghetti_streak == 2

    printer.print_state = "paused"  # a job that stops printing resets the streak
    watchdog.tick()
    assert watchdog.spaghetti_streak == 0
    assert printer.count(PAUSE) == 0


def test_snapshot_urls_resolve_to_the_printer_host_and_stay_on_the_lan(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.webcams = [
        {"name": "off", "enabled": False, "snapshot_url": "http://192.168.1.60/off"},
        {"name": "cloud", "enabled": True, "snapshot_url": "http://8.8.8.8/snap.jpg"},
        {"name": "cam", "enabled": True, "snapshot_url": "/webcam/?action=snapshot"},
    ]
    watchdog = _watchdog(tmp_path, Frames("clean"))
    assert watchdog.check_frame_once() == "clean"
    assert ("GET", "snapshot") in printer.calls

    printer.webcams = [{"name": "cloud", "enabled": True, "snapshot_url": "http://8.8.8.8/snap.jpg"}]
    assert _watchdog(tmp_path, Frames("clean")).check_frame_once() is None


def test_vision_disabled_never_fetches_frames(printer: FakeMoonraker, tmp_path: Path) -> None:
    printer.print_state = "printing"
    _watchdog(tmp_path, classify=None).tick()
    assert ("GET", "/server/webcams/list") not in printer.calls


def test_vlm_classifier_accepts_only_the_two_states(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.plugins.vision import image_analysis

    replies: list[Any] = [{"state": "spaghetti"}, {"state": "clean"}, {"state": "maybe"}, None]
    monkeypatch.setattr(image_analysis, "_extract_json_pass", lambda *_a: (replies.pop(0), []))
    monkeypatch.setattr("dana.core.model_provider.ModelProvider", lambda **_kw: object())
    assert [watchdog_module.classify_frame_with_vlm(b"x", "image/jpeg") for _ in range(4)] == [
        "spaghetti", "clean", None, None,
    ]


# -- startup -------------------------------------------------------------------------------


def test_start_from_env_needs_a_valid_lan_printer(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[PrinterWatchdog] = []
    monkeypatch.setattr(PrinterWatchdog, "start", lambda self: started.append(self))

    monkeypatch.delenv("DANA_PRINTER_IP", raising=False)
    assert watchdog_module.start_from_env() is None
    monkeypatch.setenv("DANA_PRINTER_IP", "example.com")
    assert watchdog_module.start_from_env() is None
    assert started == []

    monkeypatch.setenv("DANA_PRINTER_IP", PRINTER)
    monkeypatch.setenv("DANA_MOONRAKER_API_KEY", "secret")
    watchdog = watchdog_module.start_from_env()
    assert watchdog is not None and started == [watchdog]
    assert watchdog.client.base_url == BASE
    assert watchdog.client._session.headers["X-Api-Key"] == "secret"
    assert watchdog.classify_frame is watchdog_module.classify_frame_with_vlm

    monkeypatch.setenv("DANA_WATCHDOG_VISION", "0")
    assert watchdog_module.start_from_env().classify_frame is None


def test_watchdog_thread_starts_and_stops(printer: FakeMoonraker, tmp_path: Path) -> None:
    watchdog = _watchdog(tmp_path, poll_interval=0.01)
    watchdog.start()
    watchdog.stop(timeout=5)
    assert watchdog._thread is None
    assert ("GET", "/printer/info") in printer.calls


def test_api_server_runs_the_watchdog_for_its_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Stub:
        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(server_module.printer_watchdog, "start_from_env", lambda: events.append("start") or Stub())
    with TestClient(server_module.app):
        assert events == ["start"]
        assert isinstance(server_module._printer_watchdog, Stub)
    assert events == ["start", "stop"]
    assert server_module._printer_watchdog is None


# -- agent wiring: stops never wait for approval ------------------------------------------


def test_stop_tools_are_fail_safe_mutating_tools_in_the_hardware_domain() -> None:
    for tool_id in ("pause_print", "emergency_stop"):
        assert tool_id in rd.FAIL_SAFE_TOOL_IDS
        assert tool_id not in rd.ALWAYS_PROMPT_TOOL_IDS
        assert rd.is_mutating_tool(tool_id) is True  # an exemption, not a read-only label
        assert tool_id in rd._CAPABILITY_TOOL_IDS["hardware"]
    assert "emergency stop" in rd._HARDWARE_INTENT_KEYWORDS


def test_dispatch_threads_the_session_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(rd, "_hw_emergency_stop", lambda ip, **kw: seen.update(ip=ip, **kw) or {"ok": True, "printer": BASE})
    monkeypatch.setattr(rd, "_fsm_out_of_order_check", lambda *a, **k: None, raising=False)
    result = rd.dispatch_tool_call(
        ToolCall(tool_id="emergency_stop", arguments={"printer_ip": PRINTER}),
        engine=None, control_plane=None, api_keys={"moonraker": "k"},
    )
    assert result.ok is True, result.message
    assert seen == {"ip": PRINTER, "api_keys": {"moonraker": "k"}}


def test_stop_tools_refused_on_the_public_hf_space(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana.platform import factory as platform_factory

    monkeypatch.setattr(platform_factory, "IS_HF_SPACE", True)
    for handler in (rd._tool_pause_print, rd._tool_emergency_stop):
        result = handler({"printer_ip": PRINTER}, None, None)
        assert result["ok"] is False and "disabled in the cloud demo" in result["error"]


class _FakeProvider:
    def __init__(self, turns: list[Any]) -> None:
        self._turns = list(turns)

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        turn = self._turns.pop(0) if self._turns else "Done."
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "provider": "test"}
        return {"content": "", "tool_calls": turn, "provider": "test"}


@pytest.mark.timeout(30)
@pytest.mark.parametrize("tool_id", ["emergency_stop", "pause_print"])
def test_stop_runs_without_an_approval_prompt_even_with_auto_approve_off(
    monkeypatch: pytest.MonkeyPatch, tool_id: str
) -> None:
    monkeypatch.setattr(server_module, "get_cad_engine", lambda: MockFreeCADEngine())
    monkeypatch.setattr(server_module, "get_control_plane", lambda: MockControlPlane())
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")
    sent: list[str] = []
    monkeypatch.setattr(rd, f"_hw_{tool_id}", lambda ip, **_kw: sent.append(ip) or {"ok": True, "printer": BASE})
    fake = _FakeProvider([[ToolCall(tool_id=tool_id, arguments={"printer_ip": PRINTER})], "Stopped."])
    monkeypatch.setattr(rd, "ModelProvider", lambda **_kw: fake)

    with TestClient(server_module.app).websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "update_context", "active_plugins": ["hardware"]})
        ws.send_json({"type": "set_auto_approve", "payload": {"enabled": False}})
        ws.send_json({"text": f"emergency stop the 3d printer at {PRINTER}"})
        seen: list[str] = []
        for _ in range(40):
            msg_type = ws.receive_json().get("type")
            seen.append(msg_type)
            if msg_type == "tool_dispatch_end":
                break
        assert "hitl_approval_required" not in seen
        assert seen[-1] == "tool_dispatch_end"
    assert sent == [PRINTER]
