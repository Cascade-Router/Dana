@echo off
setlocal EnableExtensions

REM --- Auto-cleanup: kill any orphaned Dana process left over from a
REM prior crash (a pythonw.exe that never got torn down still holds port
REM 8000 and the log file open, breaking THIS launch) before trying to
REM start anything new. `call` (not `start`) so this blocks until the
REM kill actually finishes -- stop_dana.bat's own PowerShell runs hidden
REM but still synchronously from the batch file's point of view, so by
REM the time this line returns, port 8000 is guaranteed free.
call "%~dp0stop_dana.bat"

REM Project root is two levels above scripts\launchers\
cd /d "%~dp0..\.." >nul 2>&1

REM Boots the current stack: dana.api.server (FastAPI/WebSocket backend) +
REM the Tauri/React desktop GUI. Replaces the old run.py CustomTkinter
REM launch — use stop_dana.bat only if you still run the legacy GUI.

if not exist ".venv\Scripts\activate.bat" (
    echo [Dana] ERROR: .venv not found. Run "python -m venv .venv" and install requirements.txt first.
    exit /b 1
)
if not exist "frontend\node_modules" (
    echo [Dana] ERROR: frontend\node_modules not found. Run "npm install" inside frontend\ first.
    exit /b 1
)

echo [Dana] Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo [Dana] Starting FastAPI backend and Tauri desktop app (fully detached)...
REM True detachment: NOT `start /B` (that keeps the child attached to
REM THIS console, which is exactly what kept this window from closing
REM immediately) -- Start-Process, in a separate PowerShell helper script
REM so the nested quoting stays sane, gives each child its own process
REM with no console handle at all. See start_dana_detached.ps1 for both
REM Start-Process calls (backend + frontend) and the 2s gap between them.
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0start_dana_detached.ps1"

REM Teardown-on-close lives in frontend\src-tauri\src\lib.rs's
REM on_window_event handler (CloseRequested on the main window -> shells
REM out to stop_dana.vbs) -- nothing left for this script to wait for.
exit
