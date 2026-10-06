"""``dispatch_to_printer`` — sends a G-code file to a Klipper printer through
Moonraker and starts the print, in a strict order:

1. read the printer's state and require ``ready`` (a printing, paused,
   starting-up or errored machine is refused, nothing is sent);
2. upload the G-code;
3. start the print, then look up the new job's id.

This is the one tool here that acts on the physical world, so it is
deliberately narrow:

* the printer must be a LAN address (private/loopback/link-local IP, an
  mDNS ``.local`` name, or a host listed in ``DANA_PRINTER_HOSTS``) — the agent
  can't use it to post files to an arbitrary internet host;
* the file must be G-code inside the sandboxed workspace or a mounted
  directory (``resolve_sandboxed_path``);
* the Moonraker API key comes from the session's saved keys or
  ``DANA_MOONRAKER_API_KEY``, never from a tool argument;
* dana.api.server prompts for approval on EVERY call
  (``dana.core.react_dispatch.ALWAYS_PROMPT_TOOL_IDS``), whatever the
  auto-approve setting or earlier approvals this session.

``pause_print`` and ``emergency_stop`` are the opposite case: they only ever
stop a machine, so they run WITHOUT an approval prompt
(``dana.core.react_dispatch.FAIL_SAFE_TOOL_IDS``) — waiting for a human to
approve a stop defeats it. Same LAN-only address check and key handling.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlsplit

from dana.plugins.hardware.moonraker_client import DEFAULT_PORT, MoonrakerClient, MoonrakerError
from dana.plugins.os.file_system import PathEscapeError, resolve_sandboxed_path

GCODE_SUFFIXES = frozenset({".gcode", ".gco", ".g", ".bgcode"})


class PrinterAddressError(ValueError):
    """The printer address is malformed or not on the local network."""


def _allowed_hostnames() -> set[str]:
    raw = os.environ.get("DANA_PRINTER_HOSTS") or ""
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def printer_base_url(printer: str) -> str:
    """``192.168.1.50`` / ``192.168.1.50:7125`` / ``http://voron.local`` ->
    ``http://<host>:<port>`` (port defaults to Moonraker's 7125). Raises
    PrinterAddressError for anything that isn't a LAN printer."""
    raw = (printer or "").strip()
    if not raw:
        raise PrinterAddressError("printer_ip is required")
    try:
        if ipaddress.ip_address(raw).version == 6:
            raw = f"[{raw}]"  # bare IPv6: its colons would otherwise read as a port
    except ValueError:
        pass
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if parts.scheme not in ("http", "https"):
        raise PrinterAddressError(f"unsupported scheme {parts.scheme!r} — use http or https")
    if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username:
        raise PrinterAddressError("printer_ip must be a host[:port], not a URL with a path or credentials")
    host = (parts.hostname or "").lower()
    try:
        port = parts.port or DEFAULT_PORT
    except ValueError as exc:
        raise PrinterAddressError(f"invalid port in {raw!r}") from exc
    if not host:
        raise PrinterAddressError(f"no host in {raw!r}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if host in _allowed_hostnames():  # explicit opt-in, e.g. a Tailscale (100.64/10) printer
        host_part = f"[{host}]" if ip is not None and ip.version == 6 else host
    elif ip is not None:
        if not (ip.is_private or ip.is_loopback or ip.is_link_local) or ip.is_multicast or ip.is_unspecified:
            raise PrinterAddressError(f"{host} is not a local-network address; refusing to send files to it")
        host_part = f"[{host}]" if ip.version == 6 else host
    elif host.endswith(".local"):
        host_part = host
    else:
        raise PrinterAddressError(
            f"{host!r} is not a LAN printer address — use its IP, its mDNS name (e.g. voron.local), "
            "or add it to DANA_PRINTER_HOSTS"
        )
    return f"{parts.scheme}://{host_part}:{port}"


def _moonraker_api_key(api_keys: dict[str, str] | None) -> str | None:
    return (api_keys or {}).get("moonraker") or os.environ.get("DANA_MOONRAKER_API_KEY") or None


def dispatch_to_printer(
    printer_ip: str,
    gcode_filepath: str,
    *,
    allowed_mounts: list[str] | None = None,
    api_keys: dict[str, str] | None = None,
    client: MoonrakerClient | None = None,
) -> dict[str, Any]:
    """Check -> upload -> start. Returns ``{"ok": True, "job_id", "filename",
    "printer", ...}`` or ``{"ok": False, "error", "stage", ...}``, where
    ``stage`` (validate/status/upload/start) says how far the sequence got
    and, from ``start`` on, ``uploaded_filename`` says what is now on the
    printer."""
    try:
        base_url = printer_base_url(printer_ip)
    except PrinterAddressError as exc:
        return {"ok": False, "stage": "validate", "error": f"dispatch_to_printer: {exc}"}
    try:
        path = resolve_sandboxed_path(gcode_filepath, allowed_mounts)
    except PathEscapeError as exc:
        return {"ok": False, "stage": "validate", "error": f"dispatch_to_printer: {exc}"}
    if path.suffix.lower() not in GCODE_SUFFIXES:
        return {
            "ok": False,
            "stage": "validate",
            "error": f"dispatch_to_printer: {path.name} is not a G-code file ({', '.join(sorted(GCODE_SUFFIXES))})",
        }
    if not path.is_file():
        return {"ok": False, "stage": "validate", "error": f"dispatch_to_printer: no such file: {path}"}

    client = client or MoonrakerClient(base_url, _moonraker_api_key(api_keys))

    try:
        status = client.get_status()
    except MoonrakerError as exc:
        return {"ok": False, "stage": "status", "printer": base_url, "error": f"dispatch_to_printer: {exc}"}
    if status["state"] != "ready":
        detail = status.get("state_message") or status.get("current_file") or ""
        return {
            "ok": False,
            "stage": "status",
            "printer": base_url,
            "printer_state": status["state"],
            "error": (
                f"dispatch_to_printer: printer is {status['state']}, not ready — nothing was sent"
                + (f" ({detail})" if detail else "")
            ),
            "status": status,
        }

    try:
        stored = client.upload_file(path)
    except (MoonrakerError, OSError) as exc:
        return {
            "ok": False,
            "stage": "upload",
            "printer": base_url,
            "error": f"dispatch_to_printer: upload failed, print not started — {exc}",
        }

    try:
        client.start_print(stored)
    except MoonrakerError as exc:
        return {
            "ok": False,
            "stage": "start",
            "printer": base_url,
            "uploaded_filename": stored,
            "error": f"dispatch_to_printer: {stored} was uploaded but the print did not start — {exc}",
        }

    # print/start only answers "ok"; the job id lives in the history. Best
    # effort: the print has already started either way.
    job_id = None
    note = None
    try:
        job = client.latest_job()
        if job and job.get("filename") == stored:
            job_id = job.get("job_id")
        else:
            note = "print started, but its job id wasn't in the printer's history yet"
    except MoonrakerError as exc:
        note = f"print started, but the job id lookup failed ({exc})"

    result: dict[str, Any] = {
        "ok": True,
        "printer": base_url,
        "filename": stored,
        "job_id": job_id,
        "previous_print_state": status["print_state"],
        "heaters": status["heaters"],
    }
    if note:
        result["note"] = note
    return result


def _stop_action(
    action: str, printer_ip: str, api_keys: dict[str, str] | None, client: MoonrakerClient | None
) -> dict[str, Any]:
    try:
        base_url = printer_base_url(printer_ip)
    except PrinterAddressError as exc:
        return {"ok": False, "error": f"{action}: {exc}"}
    client = client or MoonrakerClient(base_url, _moonraker_api_key(api_keys))
    try:
        getattr(client, action)()
    except MoonrakerError as exc:
        return {"ok": False, "printer": base_url, "error": f"{action}: {exc}"}
    return {"ok": True, "printer": base_url, "action": action}


def pause_print(
    printer_ip: str, *, api_keys: dict[str, str] | None = None, client: MoonrakerClient | None = None
) -> dict[str, Any]:
    """Pause the running job. Recoverable: the job resumes from the printer's
    UI. Heaters stay on, so this is for print failures (spaghetti, a lifted
    part), not for thermal faults — those need ``emergency_stop``."""
    return _stop_action("pause_print", printer_ip, api_keys, client)


def emergency_stop(
    printer_ip: str, *, api_keys: dict[str, str] | None = None, client: MoonrakerClient | None = None
) -> dict[str, Any]:
    """M112: Klipper halts motion and cuts heaters immediately. The job is
    lost and the printer stays down until a FIRMWARE_RESTART."""
    return _stop_action("emergency_stop", printer_ip, api_keys, client)


__all__ = (
    "GCODE_SUFFIXES",
    "PrinterAddressError",
    "dispatch_to_printer",
    "emergency_stop",
    "pause_print",
    "printer_base_url",
)
