"""Register Donna in the current user's Windows Startup (HKCU Run key).

Run once:
  python register_startup.py

Also creates Desktop + Startup-folder shortcuts with ``dana/assets/donna.ico``.
Uses ``pythonw.exe`` / ``start_donna.bat`` so no console window appears at logon.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from dana.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    if sys.platform != "win32":
        print("[Startup] ERROR: Windows-only (winreg).", file=sys.stderr)
        return 1

    # Canonical installer path — HKCU Run + Desktop/Startup .lnk with IconLocation.
    try:
        from dana.tools.setup_startup import (
            app_icon_path,
            enable_startup,
            write_desktop_shortcut,
            write_startup_folder_shortcut,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Startup] ERROR: cannot import setup_startup ({exc})", file=sys.stderr)
        return 1

    ico = app_icon_path()
    if not ico.is_file():
        print(
            f"[Startup] WARNING: donna.ico missing at {os.path.abspath(str(ico))}",
            file=sys.stderr,
        )

    code = int(enable_startup())
    # Belt-and-suspenders: ensure shortcuts exist even if enable_startup path changes.
    desk = write_desktop_shortcut()
    start = write_startup_folder_shortcut()
    if desk is not None:
        print(f"[Startup] Desktop shortcut: {desk}")
    if start is not None:
        print(f"[Startup] Startup-folder shortcut: {start}")
    if ico.is_file():
        print(f"[Startup] IconLocation: {os.path.abspath(str(ico))},0")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
