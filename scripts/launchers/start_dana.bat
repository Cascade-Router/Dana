@echo off
setlocal EnableExtensions
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

echo [Dana] Starting FastAPI backend (scripts\launchers\launch_api_server.py) on port 8000...
start "DanaAPI" python "scripts\launchers\launch_api_server.py"

REM Give uvicorn a moment to bind before the Tauri webview's first request.
timeout /t 2 /nobreak >nul

echo [Dana] Launching Tauri desktop app — closing its window returns here...
pushd "frontend"
call npm run tauri -- dev
popd

echo [Dana] Tauri window closed — stopping the FastAPI backend...
set "DANA_API_STOPPED="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >nul 2>&1
    set "DANA_API_STOPPED=1"
)
if defined DANA_API_STOPPED (
    echo [Dana] Backend stopped.
) else (
    echo [Dana] No process was listening on port 8000 — nothing to stop.
)

endlocal
