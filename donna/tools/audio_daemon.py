"""Silent background daemon: switch default audio when Bluetooth connects/disconnects."""

from __future__ import annotations

import subprocess
import time

from donna.tools.audio_switcher import toggle_audio_endpoint

_PS_BLUETOOTH_CHECK = (
    "Get-PnpDevice -Class Bluetooth | "
    "Where-Object { $_.Status -eq 'OK' -and ("
    "$_.FriendlyName -match 'Headset|Headphone|Audio|Speaker|Earphone|AirPods|Buds'"
    ") } | "
    "Select-Object -First 1 -ExpandProperty FriendlyName"
)


def is_bluetooth_connected() -> bool:
    """Return True when a Bluetooth audio device is present and Status is OK."""
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _PS_BLUETOOTH_CHECK,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return False
    name = (completed.stdout or "").strip()
    return bool(name) and completed.returncode == 0


def run_daemon() -> None:
    """Poll Bluetooth audio presence every 5s and switch the default endpoint."""
    last_state = False
    while True:
        connected = is_bluetooth_connected()
        if connected != last_state:
            if connected:
                toggle_audio_endpoint("bluetooth")
            else:
                toggle_audio_endpoint("wired")
            last_state = connected
        time.sleep(5)


if __name__ == "__main__":
    try:
        from donna.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    run_daemon()
