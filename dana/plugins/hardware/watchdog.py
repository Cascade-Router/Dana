"""Printer watchdog — a deterministic background monitor for one Klipper
printer. No LLM decides anything here; the only model call is the optional
per-frame spaghetti classification, and acting on it still takes a fixed
rule (3 consecutive "spaghetti" frames).

What it does, every ``POLL_INTERVAL_S`` (5 s):

* reads Moonraker's status (``/printer/info`` + ``/printer/objects/query?
  print_stats&heaters``);
* **Klipper fault** — Klippy ``shutdown``/``error``, or ``print_stats``
  ``error``: logs it and sends ``emergency_stop`` once per fault, so the
  machine is held in a known-dead state until someone restarts it;
* **temperature drift** — a heater that has reached its target and then
  strays more than ``TEMP_DRIFT_C`` from it is logged, never acted on.
  Thermal-runaway protection stays with Klipper's own ``verify_heater``
  check, which cuts the heaters in firmware far faster than a 5 s poll.
  Heat-up is not drift: a heater only counts once it has come within
  ``TEMP_DRIFT_C`` of a new target;
* **spaghetti** — while a job is printing, every ``SNAPSHOT_INTERVAL_S`` it
  classifies a webcam snapshot as ``clean`` or ``spaghetti``, and pauses the
  print after ``SPAGHETTI_FRAMES_TO_PAUSE`` consecutive ``spaghetti`` frames
  (a ``clean`` frame resets the count; a frame that couldn't be fetched or
  classified leaves it unchanged).

Every event is one JSON object per line in ``LOG_PATH``, timestamped by this
code with ``datetime.now(timezone.utc)``, never by a model.

``dana.api.server`` starts it with the API server when ``DANA_PRINTER_IP``
holds a valid LAN printer address (``start_from_env``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from dana.paths import LOGS_DIR
from dana.plugins.hardware.moonraker_client import MoonrakerClient, MoonrakerError
from dana.plugins.hardware.printer_tools import (
    PrinterAddressError,
    _moonraker_api_key,
    printer_base_url,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5.0
SNAPSHOT_INTERVAL_S = 30.0
TEMP_DRIFT_C = 5.0
SPAGHETTI_FRAMES_TO_PAUSE = 3
FAULT_KLIPPY_STATES = frozenset({"shutdown", "error"})
LOG_PATH = LOGS_DIR / "printer_watchdog.jsonl"

FrameClassifier = Callable[[bytes, str], "str | None"]

_SPAGHETTI_PROMPT = (
    "This is a webcam frame of a 3D printer during a print. Decide whether the print has failed "
    "into 'spaghetti': loose, tangled strands of extruded filament in the air or piled on the bed, "
    "or a part knocked loose from the bed. A normal print in progress, an empty bed or an unclear "
    'frame is "clean". Answer with ONLY one JSON object, exactly {"state": "clean"} or '
    '{"state": "spaghetti"}, and nothing else.'
)


def classify_frame_with_vlm(image: bytes, mime_type: str) -> str | None:
    """``"clean"``/``"spaghetti"`` from the configured vision model (local
    Ollama first, then the cloud fallback — the same provider order
    analyze_reference_design uses), or None if no model answered with that
    exact JSON."""
    import base64

    from dana.core.model_provider import ModelProvider
    from dana.plugins.vision.image_analysis import _extract_json_pass

    parsed, attempts = _extract_json_pass(
        ModelProvider(), _SPAGHETTI_PROMPT, base64.b64encode(image).decode("ascii"), mime_type
    )
    state = parsed.get("state") if isinstance(parsed, dict) else None
    if state in ("clean", "spaghetti"):
        return state
    logger.debug("watchdog: frame not classified (%s; attempts: %s)", parsed, attempts)
    return None


class PrinterWatchdog:
    def __init__(
        self,
        client: MoonrakerClient,
        *,
        log_path: Path = LOG_PATH,
        classify_frame: FrameClassifier | None = classify_frame_with_vlm,
        poll_interval: float = POLL_INTERVAL_S,
        snapshot_interval: float = SNAPSHOT_INTERVAL_S,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.log_path = Path(log_path)
        self.classify_frame = classify_frame  # None disables spaghetti detection
        self.poll_interval = poll_interval
        self.snapshot_interval = snapshot_interval
        self._now = now
        self._monotonic = monotonic

        self._fault: str | None = None  # the threshold text of the active fault
        self._unreachable = False
        # heater name -> {"target", "settled", "drifting"}
        self._heaters: dict[str, dict[str, Any]] = {}
        self.spaghetti_streak = 0
        self._last_frame_at = -math.inf
        self._snapshot_url: str | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- event log ---------------------------------------------------------------------

    def _log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"timestamp": self._now().isoformat(), "event": event, "printer": self.client.base_url, **fields}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("watchdog: could not write %s: %s", self.log_path, exc)
        logger.warning("printer watchdog: %s", json.dumps(record))
        return record

    def _act(self, action: str) -> str:
        try:
            getattr(self.client, action)()
        except MoonrakerError as exc:
            return f"failed: {exc}"
        return "ok"

    # -- telemetry -----------------------------------------------------------------------

    def _read_status(self) -> dict[str, Any] | None:
        try:
            status = self.client.get_status()
        except MoonrakerError as status_exc:
            # A shut-down Klippy can fail the objects query while /printer/info
            # still answers, and that state is exactly the one to catch.
            try:
                info = self.client.get_info()
            except MoonrakerError:
                if not self._unreachable:
                    self._unreachable = True
                    self._log("printer_unreachable", detail=str(status_exc), action=None)
                return None
            status = {
                "klippy_state": str(info.get("state") or "unknown"),
                "state_message": info.get("state_message"),
                "print_state": "unknown",
                "heaters": {},
            }
        if self._unreachable:
            self._unreachable = False
            self._log("printer_reachable", action=None)
        return status

    def poll_once(self) -> dict[str, Any] | None:
        """One telemetry pass. Returns the status it read, or None if the
        printer couldn't be reached."""
        status = self._read_status()
        if status is None:
            return None
        self._check_fault(status)
        self._check_temperatures(status.get("heaters") or {})
        return status

    def _check_fault(self, status: dict[str, Any]) -> None:
        klippy_state = status.get("klippy_state")
        print_state = status.get("print_state")
        if klippy_state in FAULT_KLIPPY_STATES:
            threshold = f"Klipper host state is '{klippy_state}'"
        elif print_state == "error":
            threshold = "Klipper print_stats state is 'error'"
        else:
            if self._fault is not None:
                self._log("klipper_fault_cleared", previous=self._fault, action=None)
                self._fault = None
            return
        if self._fault is not None:
            return  # already handled this fault; one e-stop per fault
        self._fault = threshold
        self._log(
            "klipper_fault",
            threshold=threshold,
            detail=status.get("state_message"),
            action="emergency_stop",
            action_result=self._act("emergency_stop"),
            recommended_intervention=(
                "Do not restart blind: read klippy.log for the cause, check heater/thermistor wiring and "
                "the toolhead if it was a heater or motion fault, clear the bed, then FIRMWARE_RESTART."
            ),
        )

    def _check_temperatures(self, heaters: dict[str, dict[str, Any]]) -> None:
        for name, reading in heaters.items():
            temperature, target = reading.get("temperature"), reading.get("target")
            if temperature is None or target is None:
                continue
            state = self._heaters.get(name)
            if not target:
                self._heaters.pop(name, None)  # heater off: nothing to hold
                continue
            if state is None or state["target"] != target:
                state = self._heaters[name] = {"target": target, "settled": False, "drifting": False}
            drift = temperature - target
            if not state["settled"]:
                if abs(drift) <= TEMP_DRIFT_C:
                    state["settled"] = True  # heat-up done; from here on, a deviation is drift
                continue
            if abs(drift) > TEMP_DRIFT_C and not state["drifting"]:
                state["drifting"] = True
                self._log(
                    "temperature_drift",
                    threshold=f"{name}: |temperature - target| > {TEMP_DRIFT_C:g}C after reaching target",
                    detail={"heater": name, "temperature": temperature, "target": target, "drift": round(drift, 2)},
                    action=None,
                    recommended_intervention=(
                        "Watch it; Klipper's verify_heater shuts the printer down on real thermal runaway. "
                        "Check the part-cooling fan, thermistor seating and drafts."
                    ),
                )
            elif abs(drift) <= TEMP_DRIFT_C and state["drifting"]:
                state["drifting"] = False
                self._log(
                    "temperature_recovered",
                    detail={"heater": name, "temperature": temperature, "target": target},
                    action=None,
                )

    # -- vision ----------------------------------------------------------------------------

    def _resolve_snapshot_url(self) -> str | None:
        if self._snapshot_url:
            return self._snapshot_url
        for cam in self.client.list_webcams():
            if cam.get("enabled") is False or not cam.get("snapshot_url"):
                continue
            # Relative URLs (the default "/webcam/?action=snapshot") are served
            # by the printer host's web server on port 80, not Moonraker's
            # port. base_url is always "scheme://host:port" (printer_base_url).
            base = urlsplit(self.client.base_url)
            host = base.netloc.rsplit(":", 1)[0]
            url = urljoin(f"{base.scheme}://{host}/", str(cam["snapshot_url"]))
            parts = urlsplit(url)
            try:
                printer_base_url(f"{parts.scheme}://{parts.netloc}")  # LAN-only, like every printer request
            except PrinterAddressError as exc:
                logger.warning("watchdog: ignoring webcam %r: %s", cam.get("name"), exc)
                continue
            self._snapshot_url = url
            return url
        return None

    def check_frame_once(self) -> str | None:
        """Fetch and classify one snapshot; pause after the Nth consecutive
        spaghetti frame. Returns the classification (None if none was made)."""
        if self.classify_frame is None:
            return None
        try:
            url = self._resolve_snapshot_url()
            if url is None:
                return None
            image, mime = self.client.fetch_image(url)
        except MoonrakerError as exc:
            self._snapshot_url = None  # re-read the webcam list next time
            logger.debug("watchdog: no snapshot: %s", exc)
            return None
        try:
            verdict = self.classify_frame(image, mime)
        except Exception as exc:  # noqa: BLE001 — a classifier failure must never kill the watchdog
            logger.debug("watchdog: classifier raised: %s", exc)
            return None
        if verdict == "clean":
            self.spaghetti_streak = 0
        elif verdict == "spaghetti":
            self.spaghetti_streak += 1
            if self.spaghetti_streak >= SPAGHETTI_FRAMES_TO_PAUSE:
                self.spaghetti_streak = 0
                self._log(
                    "spaghetti_detected",
                    threshold=f"{SPAGHETTI_FRAMES_TO_PAUSE} consecutive webcam frames classified as spaghetti",
                    detail={"snapshot_url": url},
                    action="pause_print",
                    action_result=self._act("pause_print"),
                    recommended_intervention=(
                        "Look at the printer: if the print failed, clear the bed and cancel the job; if it was a "
                        "false alarm, resume it from the printer's web UI."
                    ),
                )
        return verdict

    # -- loop --------------------------------------------------------------------------------

    def tick(self) -> None:
        status = self.poll_once()
        printing = bool(status) and status.get("print_state") == "printing"
        if not printing:
            self.spaghetti_streak = 0
            return
        if self.classify_frame is not None and self._monotonic() - self._last_frame_at >= self.snapshot_interval:
            self._last_frame_at = self._monotonic()
            self.check_frame_once()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — keep watching; a bad tick is logged, not fatal
                logger.exception("watchdog: tick failed")
            self._stop.wait(self.poll_interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="printer-watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            self._thread = None


def start_from_env() -> PrinterWatchdog | None:
    """Start a watchdog for ``DANA_PRINTER_IP`` (Moonraker key from
    ``DANA_MOONRAKER_API_KEY``; ``DANA_WATCHDOG_VISION=0`` turns spaghetti
    detection off). Returns None, starting nothing, when no printer is
    configured or the address isn't a LAN printer."""
    printer = (os.environ.get("DANA_PRINTER_IP") or "").strip()
    if not printer:
        return None
    try:
        base_url = printer_base_url(printer)
    except PrinterAddressError as exc:
        logger.warning("printer watchdog not started: DANA_PRINTER_IP=%r: %s", printer, exc)
        return None
    vision = (os.environ.get("DANA_WATCHDOG_VISION") or "1").strip().lower() not in ("0", "false", "no", "off")
    watchdog = PrinterWatchdog(
        MoonrakerClient(base_url, _moonraker_api_key(None)),
        classify_frame=classify_frame_with_vlm if vision else None,
    )
    watchdog.start()
    logger.info("printer watchdog watching %s (vision %s), log: %s", base_url, "on" if vision else "off", LOG_PATH)
    return watchdog


__all__ = ("PrinterWatchdog", "classify_frame_with_vlm", "start_from_env")
