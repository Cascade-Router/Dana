# Launches both Dana processes as fully independent, windowless processes
# -- called by start_dana.bat instead of that batch file using `start /B`
# directly, because `/B` keeps a child attached to the LAUNCHING cmd.exe's
# own console. That attachment is exactly what made the previous version
# of start_dana.bat unable to close its window immediately: a console
# can't fully go away while something is still attached to it. Start-Process
# gives each child its own process object with no console handle at all,
# so the batch script's window can vanish the instant this script returns
# without dragging either child down or hanging.
#
# Output redirection (-RedirectStandardOutput/-RedirectStandardError) is
# required, not cosmetic, for the pythonw.exe case: a windowless Python
# process has no console, so an unredirected print()/logging call can
# crash the moment anything tries to write to a null stdout.

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logsDir = Join-Path $root "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$apiScript = Join-Path $root "scripts\launchers\launch_api_server.py"

Start-Process -FilePath $pythonw `
    -ArgumentList "`"$apiScript`"" `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logsDir "dana_backend.log") `
    -RedirectStandardError (Join-Path $logsDir "dana_backend.err.log")

# Give uvicorn a moment to bind before the Tauri webview's first request.
Start-Sleep -Seconds 2

# cmd.exe /c is the target here, not `npm` directly -- npm on Windows is
# npm.cmd (a batch shim), and Start-Process's non-shell process-creation
# path (required for -RedirectStandardOutput/-Error to work at all) can't
# launch a .cmd file on its own; it needs a command interpreter to run it.
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run tauri -- dev" `
    -WorkingDirectory (Join-Path $root "frontend") `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logsDir "dana_frontend.log") `
    -RedirectStandardError (Join-Path $logsDir "dana_frontend.err.log")
