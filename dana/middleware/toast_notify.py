"""Silent Windows toast helper for Stage 4.3 actuator callbacks."""

from __future__ import annotations

import os
import subprocess
import sys
import threading


def show_silent_toast(
    title: str,
    message: str,
    *,
    app_id: str = "Donna",
    icon: str | None = None,
) -> bool:
    """Show a native Windows toast with audio silenced.

    Prefers ``win11toast``; falls back to a PowerShell WinRT toast. Returns
    True when a notification was dispatched (best-effort; never raises).

    Prefer ``show_silent_toast_async`` from actuator workers — WinRT/win11toast
    can block for several seconds on dismissal timeout.

    ``icon`` may be a filesystem path to a transparent PNG (RGBA toast logo).
    """
    if (os.environ.get("DONNA_DISABLE_TOAST") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    title_s = (title or "Donna").strip() or "Donna"
    body_s = (message or "").strip() or "Task update"
    if sys.platform != "win32":
        return False

    icon_path = (icon or "").strip() or None
    if icon_path is None:
        try:
            from dana.ui.logo import toast_logo_path

            path = toast_logo_path((64, 64))
            icon_path = str(path) if path is not None else None
        except Exception:  # noqa: BLE001
            icon_path = None

    try:
        from win11toast import toast as _toast

        # win11toast: audio={'silent': 'true'} suppresses the chime.
        kwargs: dict = {
            "app_id": app_id,
            "audio": {"silent": "true"},
        }
        if icon_path:
            kwargs["icon"] = icon_path
        _toast(title_s, body_s, **kwargs)
        return True
    except Exception:  # noqa: BLE001
        pass

    return _powershell_silent_toast(title_s, body_s)


def show_silent_toast_async(
    title: str,
    message: str,
    *,
    app_id: str = "Donna",
) -> None:
    """Fire-and-forget toast so actuator workers never block on UI (Stage 4.4)."""

    def _run() -> None:
        try:
            show_silent_toast(title, message, app_id=app_id)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_run, name="donna-toast", daemon=True).start()


def _powershell_silent_toast(title: str, message: str) -> bool:
    """Fallback silent toast via Windows.UI.Notifications (no extra pip deps)."""
    # Escape for single-quoted PowerShell strings.
    def _esc(s: str) -> str:
        return (s or "").replace("'", "''")

    ps = f"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$template = @"
<toast>
  <visual>
    <binding template="ToastGeneric">
      <text>{_esc(title)}</text>
      <text>{_esc(message)}</text>
    </binding>
  </visual>
  <audio silent="true" />
</toast>
"@
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Donna').Show($toast)
"""
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return int(completed.returncode) == 0
    except Exception:  # noqa: BLE001
        return False


def format_actuator_toast(tool_name: str, status: str) -> tuple[str, str]:
    """Return ``(title, body)`` for a completed/failed actuator task."""
    tool = (tool_name or "task").strip() or "task"
    st = (status or "").strip().lower() or "completed"
    outcome = "completed" if st == "completed" else "failed"
    return "Donna Task", f"Donna Task: {tool} {outcome}."
