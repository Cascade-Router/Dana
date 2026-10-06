"""Minimal Moonraker (Klipper) HTTP API client — status, G-code upload, print
start, and the job-history lookup that gives a started print its job id.

Endpoints (Moonraker's documented REST API):

* ``GET  /printer/info`` — Klippy host state: ``ready`` | ``startup`` |
  ``shutdown`` | ``error``, plus ``state_message``.
* ``GET  /printer/objects/query?print_stats&heaters`` — print job state
  (``standby`` | ``printing`` | ``paused`` | ``complete`` | ``cancelled`` |
  ``error``) and the names of every heater; a second query reads those
  heaters' ``temperature``/``target``.
* ``POST /server/files/upload`` — multipart ``file`` into the ``gcodes`` root.
* ``POST /printer/print/start`` — JSON ``{"filename": ...}``; replies ``"ok"``.
* ``GET  /server/history/list?limit=1&order=desc`` — newest job, for its id.
* ``POST /printer/print/pause`` — pause the running job (heaters stay on).
* ``POST /printer/emergency_stop`` — M112: Klipper shuts down at once, the
  job is lost and the printer needs a FIRMWARE_RESTART.
* ``GET  /server/webcams/list`` — configured webcams and their snapshot URLs.

Every response wraps its payload as ``{"result": ...}``; failures carry
``{"error": {"code", "message"}}``. Auth, when the printer requires it, is
the ``X-Api-Key`` header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

DEFAULT_PORT = 7125
_TIMEOUT_S = 10.0
_UPLOAD_TIMEOUT_S = 300.0  # large G-code over Wi-Fi

# print_stats states in which the machine is busy with (or stuck on) a job.
BUSY_PRINT_STATES = frozenset({"printing", "paused"})


class MoonrakerError(RuntimeError):
    """A request to the printer failed: unreachable, rejected, or malformed."""


class MoonrakerClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = _TIMEOUT_S,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        if api_key:
            self._session.headers["X-Api-Key"] = api_key

    # -- transport -----------------------------------------------------------------

    def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=timeout or self.timeout, **kwargs)
        except requests.Timeout as exc:
            raise MoonrakerError(f"{method} {path}: printer at {self.base_url} did not answer in time") from exc
        except requests.ConnectionError as exc:
            raise MoonrakerError(f"{method} {path}: could not connect to printer at {self.base_url}") from exc
        except requests.RequestException as exc:
            raise MoonrakerError(f"{method} {path}: request failed ({exc})") from exc

        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code in (401, 403):
            raise MoonrakerError(
                f"{method} {path}: printer rejected the request (HTTP {response.status_code}) — "
                "it requires a Moonraker API key"
            )
        if not response.ok:
            message = (body or {}).get("error", {}).get("message") if isinstance(body, dict) else None
            raise MoonrakerError(f"{method} {path}: HTTP {response.status_code} — {message or response.reason}")
        if not isinstance(body, dict) or "result" not in body:
            raise MoonrakerError(f"{method} {path}: unexpected (non-Moonraker) response from {self.base_url}")
        return body["result"]

    # -- API -------------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        return self._request("GET", "/printer/info")

    def query_objects(self, *objects: str) -> dict[str, Any]:
        # Moonraker takes bare object names as query keys: ?print_stats&heaters
        query = "&".join(objects)
        return self._request("GET", f"/printer/objects/query?{query}").get("status", {})

    def get_status(self) -> dict[str, Any]:
        """The machine's state in one dict. ``state`` is the summary the
        dispatch tool gates on: ``ready`` only when Klippy is ready AND no job
        is printing/paused/errored; otherwise the Klippy state
        (``startup``/``shutdown``/``error``) or the busy print state."""
        info = self.get_info()
        klippy_state = str(info.get("state") or "unknown")
        status = self.query_objects("print_stats", "heaters")
        print_stats = status.get("print_stats") or {}
        print_state = str(print_stats.get("state") or "unknown")

        heater_names = list((status.get("heaters") or {}).get("available_heaters") or [])
        heaters: dict[str, dict[str, float | None]] = {}
        if heater_names:
            temps = self.query_objects(*heater_names)
            for name in heater_names:
                obj = temps.get(name) or {}
                heaters[name] = {"temperature": obj.get("temperature"), "target": obj.get("target")}

        if klippy_state != "ready":
            state = klippy_state
        elif print_state in BUSY_PRINT_STATES:
            state = print_state
        elif print_state == "error":
            state = "error"
        else:
            state = "ready"
        return {
            "state": state,
            "klippy_state": klippy_state,
            "state_message": info.get("state_message"),
            "print_state": print_state,
            "current_file": print_stats.get("filename") or None,
            "heaters": heaters,
            "hostname": info.get("hostname"),
        }

    def upload_file(self, filepath: str | Path, *, remote_name: str | None = None) -> str:
        """Upload a G-code file into the printer's ``gcodes`` root; returns the
        stored filename (Moonraker's path, relative to that root)."""
        path = Path(filepath)
        name = remote_name or path.name
        with path.open("rb") as fh:
            result = self._request(
                "POST",
                "/server/files/upload",
                files={"file": (name, fh, "application/octet-stream")},
                data={"root": "gcodes"},
                timeout=_UPLOAD_TIMEOUT_S,
            )
        # Current Moonraker: {"item": {"path": ..., "root": ...}, "action": ...};
        # older releases replied with the bare path string.
        if isinstance(result, dict):
            stored = (result.get("item") or {}).get("path")
        else:
            stored = result
        if not stored:
            raise MoonrakerError("POST /server/files/upload: printer did not report the stored file")
        return str(stored)

    def start_print(self, filename: str) -> None:
        result = self._request("POST", "/printer/print/start", json={"filename": filename})
        if result != "ok":
            raise MoonrakerError(f"POST /printer/print/start: unexpected reply {result!r}")

    def latest_job(self) -> dict[str, Any] | None:
        result = self._request("GET", "/server/history/list?limit=1&order=desc")
        jobs = (result.get("jobs") or []) if isinstance(result, dict) else []
        return jobs[0] if jobs else None

    def pause_print(self) -> None:
        result = self._request("POST", "/printer/print/pause")
        if result != "ok":
            raise MoonrakerError(f"POST /printer/print/pause: unexpected reply {result!r}")

    def emergency_stop(self) -> None:
        result = self._request("POST", "/printer/emergency_stop")
        if result != "ok":
            raise MoonrakerError(f"POST /printer/emergency_stop: unexpected reply {result!r}")

    def list_webcams(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/server/webcams/list")
        webcams = result.get("webcams") if isinstance(result, dict) else None
        return [cam for cam in webcams or [] if isinstance(cam, dict)]

    def fetch_image(self, url: str) -> tuple[bytes, str]:
        """GET a webcam snapshot (raw image, not a Moonraker JSON reply);
        returns (bytes, mime type). ``url`` must already be absolute."""
        try:
            response = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise MoonrakerError(f"GET {url}: snapshot request failed ({exc})") from exc
        if not response.ok:
            raise MoonrakerError(f"GET {url}: HTTP {response.status_code}")
        mime = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            raise MoonrakerError(f"GET {url}: not an image ({mime or 'no content type'})")
        return response.content, mime
