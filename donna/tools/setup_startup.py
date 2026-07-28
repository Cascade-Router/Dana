"""Cross-platform Donna login/startup registration.

Windows: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
macOS:   ~/Library/LaunchAgents/com.donna.agent.plist
Linux:   ~/.config/autostart/donna.desktop

Usage:
  python -m donna.tools.setup_startup install
  python -m donna.tools.setup_startup uninstall
  python -m donna.tools.setup_startup status
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import sys
from pathlib import Path

VALUE_NAME = "DonnaAssistant"
MACOS_LABEL = "com.donna.agent"
MACOS_PLIST_NAME = f"{MACOS_LABEL}.plist"
LINUX_DESKTOP_NAME = "donna.desktop"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
UNIX_STARTUP_LOG = "/tmp/donna_startup.log"


def project_root() -> Path:
    from donna.paths import PROJECT_ROOT

    return Path(os.path.abspath(str(PROJECT_ROOT)))


def absolute_workdir() -> str:
    """Absolute Donna repo path for WorkingDirectory / batch ``cd``."""
    try:
        return os.path.abspath(str(project_root()))
    except Exception:  # noqa: BLE001
        return os.path.abspath(os.getcwd())


def _system() -> str:
    return platform.system()


def python_launcher(*, headless: bool = True) -> Path:
    """Prefer a venv interpreter; headless startup uses console ``python`` (not pythonw)."""
    root = project_root()
    system = _system()
    if system == "Windows":
        venv = root / ".venv" / "Scripts"
        # Headless needs a console interpreter so stdout/stderr redirect to the log.
        order = ("python.exe", "pythonw.exe") if headless else ("pythonw.exe", "python.exe")
        for name in order:
            candidate = venv / name
            if candidate.is_file():
                return Path(os.path.abspath(str(candidate)))
    else:
        venv_bin = root / ".venv" / "bin"
        for name in ("python3", "python"):
            candidate = venv_bin / name
            if candidate.is_file():
                return Path(os.path.abspath(str(candidate)))
    return Path(os.path.abspath(sys.executable))


def entry_script() -> Path:
    return Path(os.path.abspath(str(project_root() / "run.py")))


def bat_path() -> Path:
    return project_root() / "start_donna.bat"


def app_icon_path() -> Path:
    """Absolute path to ``donna/assets/donna.ico`` (Windows shortcut IconLocation)."""
    return Path(os.path.abspath(str(project_root() / "donna" / "assets" / "donna.ico")))


def desktop_shortcut_path() -> Path:
    """``%USERPROFILE%\\Desktop\\Donna.lnk`` (Windows)."""
    return Path.home() / "Desktop" / "Donna.lnk"


def startup_folder_shortcut_path() -> Path:
    """``%APPDATA%\\...\\Startup\\Donna.lnk`` (Windows login Startup folder)."""
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return (
        Path(appdata)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / "Donna.lnk"
    )


def _write_windows_shortcut(
    *,
    lnk: Path,
    target: Path,
    working_directory: str,
    description: str,
) -> Path | None:
    """Create/update a ``.lnk`` with ``IconLocation`` → absolute ``donna.ico``."""
    if _system() != "Windows":
        return None
    ico = app_icon_path()
    ico_abs = os.path.abspath(str(ico))
    try:
        import subprocess

        lnk.parent.mkdir(parents=True, exist_ok=True)
        ps = (
            "$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut({_ps_quote(str(lnk))}); "
            f"$s.TargetPath = {_ps_quote(str(target))}; "
            f"$s.WorkingDirectory = {_ps_quote(working_directory)}; "
            "$s.WindowStyle = 7; "
            f"$s.Description = {_ps_quote(description)}; "
        )
        if os.path.isfile(ico_abs):
            # Explicit absolute IconLocation — required for Desktop / Startup .lnk icons.
            ps += f"$s.IconLocation = {_ps_quote(ico_abs + ',0')}; "
        ps += "$s.Save()"
        creationflags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = int(subprocess.CREATE_NO_WINDOW)
        subprocess.run(  # noqa: S603
            [
                "powershell",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps,
            ],
            check=False,
            capture_output=True,
            timeout=30,
            creationflags=creationflags,
        )
        if lnk.is_file():
            return lnk
    except Exception:  # noqa: BLE001
        return None
    return None


def write_desktop_shortcut() -> Path | None:
    """Create/update Desktop ``Donna.lnk`` → ``start_donna.bat`` with ``donna.ico``."""
    if _system() != "Windows":
        return None
    bat = write_start_bat()
    return _write_windows_shortcut(
        lnk=desktop_shortcut_path(),
        target=Path(os.path.abspath(str(bat))),
        working_directory=absolute_workdir(),
        description="Donna — local-first voice agent",
    )


def write_startup_folder_shortcut() -> Path | None:
    """Create/update Startup-folder ``Donna.lnk`` with ``donna.ico`` (login launch)."""
    if _system() != "Windows":
        return None
    bat = write_start_bat()
    return _write_windows_shortcut(
        lnk=startup_folder_shortcut_path(),
        target=Path(os.path.abspath(str(bat))),
        working_directory=absolute_workdir(),
        description="Donna — launch at Windows login",
    )


def _ps_quote(value: str) -> str:
    """Single-quote a string for PowerShell (escape embedded quotes)."""
    return "'" + value.replace("'", "''") + "'"


def write_start_bat() -> Path:
    """Create ``start_donna.bat``: abs ``cd``, GUI enabled (no ``--no-gui``)."""
    root = absolute_workdir()
    # Prefer pythonw for a windowed GUI launch (no console flash).
    py = str(python_launcher(headless=False))
    entry = str(entry_script())
    # ``start`` detaches so the Run-key / Startup launcher returns immediately.
    lines = [
        "@echo off",
        f'cd /d "{root}"',
        "REM GUI enabled (CustomTkinter). Use stop_donna.bat before relaunch.",
        f'start "Donna" "{py}" "{entry}"',
        "",
    ]
    path = bat_path()
    path.write_text("\n".join(lines), encoding="utf-8", newline="\r\n")
    return path


def macos_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / MACOS_PLIST_NAME


def linux_desktop_path() -> Path:
    return Path.home() / ".config" / "autostart" / LINUX_DESKTOP_NAME


def _write_macos_plist() -> Path:
    root = absolute_workdir()
    py = str(python_launcher(headless=True))
    entry = str(entry_script())
    log_path = UNIX_STARTUP_LOG
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>Label</key>
\t<string>{MACOS_LABEL}</string>
\t<key>ProgramArguments</key>
\t<array>
\t\t<string>{py}</string>
\t\t<string>{entry}</string>
\t\t<string>--no-gui</string>
\t</array>
\t<key>WorkingDirectory</key>
\t<string>{root}</string>
\t<key>StandardOutPath</key>
\t<string>{log_path}</string>
\t<key>StandardErrorPath</key>
\t<string>{log_path}</string>
\t<key>RunAtLoad</key>
\t<true/>
\t<key>KeepAlive</key>
\t<false/>
</dict>
</plist>
"""
    path = macos_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist, encoding="utf-8")
    return path


def _write_linux_desktop() -> Path:
    root = absolute_workdir()
    py = str(python_launcher(headless=True))
    entry = str(entry_script())
    log_path = UNIX_STARTUP_LOG
    icon = app_icon_path()
    # Desktop Entry Exec → bash -c with stdout/stderr piped to the startup log.
    inner = f"{shlex.quote(py)} {shlex.quote(entry)} --no-gui > {log_path} 2>&1"
    exec_cmd = f"/bin/bash -c {shlex.quote(inner)}"
    body = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.0\n"
        "Name=Donna\n"
        "Comment=Donna local-first voice agent (headless)\n"
        f"Exec={exec_cmd}\n"
        f"Path={root}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    if icon.is_file():
        body += f"Icon={icon.as_posix()}\n"
    path = linux_desktop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _enable_windows() -> int:
    import winreg

    bat = write_start_bat()
    command = f'"{bat}"'
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
    print(f"[OK] Startup enabled: HKCU\\...\\Run\\{VALUE_NAME}")
    print(f"     Command: {command}")
    print(f"     Launcher: {bat}")
    ico = app_icon_path()
    if ico.is_file():
        print(f"     Icon: {os.path.abspath(str(ico))}")
    else:
        print(f"     WARNING: icon missing at {ico}")
    lnk = write_desktop_shortcut()
    if lnk is not None:
        print(f"     Desktop shortcut: {lnk}")
    startup_lnk = write_startup_folder_shortcut()
    if startup_lnk is not None:
        print(f"     Startup-folder shortcut: {startup_lnk}")
    return 0


def _disable_windows() -> int:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, VALUE_NAME)
        print(f"[OK] Startup removed: {VALUE_NAME}")
        return 0
    except FileNotFoundError:
        print(f"[OK] Startup entry already absent: {VALUE_NAME}")
        return 0


def _status_windows() -> int:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ
        ) as key:
            value, regtype = winreg.QueryValueEx(key, VALUE_NAME)
        print(f"[ON] {VALUE_NAME} = {value} (type={regtype})")
        return 0
    except FileNotFoundError:
        print(f"[OFF] {VALUE_NAME} is not in HKCU Run")
        return 1


def is_startup_enabled() -> bool:
    """Quiet Boolean: True when login/startup registration is present."""
    system = _system()
    if system == "Windows":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ
            ) as key:
                winreg.QueryValueEx(key, VALUE_NAME)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False
    if system == "Darwin":
        return macos_plist_path().is_file()
    if system == "Linux":
        return linux_desktop_path().is_file()
    return False


# Alias used by tray ``checked=`` bindings.
check_startup_registry_status = is_startup_enabled


def _enable_macos() -> int:
    path = _write_macos_plist()
    print(f"[OK] Startup enabled: {path}")
    print(f"     Label: {MACOS_LABEL}")
    return 0


def _disable_macos() -> int:
    path = macos_plist_path()
    if path.is_file():
        path.unlink()
        print(f"[OK] Startup removed: {path}")
    else:
        print(f"[OK] Startup entry already absent: {path}")
    return 0


def _status_macos() -> int:
    path = macos_plist_path()
    if path.is_file():
        print(f"[ON] {path}")
        return 0
    print(f"[OFF] {path} is not present")
    return 1


def _enable_linux() -> int:
    path = _write_linux_desktop()
    print(f"[OK] Startup enabled: {path}")
    return 0


def _disable_linux() -> int:
    path = linux_desktop_path()
    if path.is_file():
        path.unlink()
        print(f"[OK] Startup removed: {path}")
    else:
        print(f"[OK] Startup entry already absent: {path}")
    return 0


def _status_linux() -> int:
    path = linux_desktop_path()
    if path.is_file():
        print(f"[ON] {path}")
        return 0
    print(f"[OFF] {path} is not present")
    return 1


def enable_startup() -> int:
    """Register Donna to launch at user login on the current OS."""
    system = _system()
    if system == "Windows":
        return _enable_windows()
    if system == "Darwin":
        return _enable_macos()
    if system == "Linux":
        return _enable_linux()
    print(f"[ERROR] Unsupported platform for startup registration: {system}", file=sys.stderr)
    return 2


def disable_startup() -> int:
    """Remove Donna from login/startup on the current OS."""
    system = _system()
    if system == "Windows":
        return _disable_windows()
    if system == "Darwin":
        return _disable_macos()
    if system == "Linux":
        return _disable_linux()
    print(f"[ERROR] Unsupported platform for startup registration: {system}", file=sys.stderr)
    return 2


def startup_status() -> int:
    """Print whether Donna is registered for login/startup (0=on, 1=off)."""
    system = _system()
    if system == "Windows":
        return _status_windows()
    if system == "Darwin":
        return _status_macos()
    if system == "Linux":
        return _status_linux()
    print(f"[ERROR] Unsupported platform for startup registration: {system}", file=sys.stderr)
    return 2


# CLI aliases (kept for existing docs / scripts).
def install() -> int:
    return enable_startup()


def uninstall() -> int:
    return disable_startup()


def status() -> int:
    return startup_status()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add or remove Donna from user login/startup "
            "(Windows Run key, macOS LaunchAgent, or Linux autostart)."
        )
    )
    parser.add_argument(
        "action",
        choices=("install", "uninstall", "status", "enable", "disable"),
        help="install|enable / uninstall|disable / status",
    )
    args = parser.parse_args(argv)
    if args.action in ("install", "enable"):
        return enable_startup()
    if args.action in ("uninstall", "disable"):
        return disable_startup()
    return startup_status()


if __name__ == "__main__":
    raise SystemExit(main())
