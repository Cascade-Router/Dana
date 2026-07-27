@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM -WindowStyle Hidden prevents PowerShell console flashes during STOP DONNA.
powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command ^
  "$root = [regex]::Escape((Resolve-Path -LiteralPath '%~dp0').Path); " ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name = 'pythonw.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $root) }; " ^
  "if (-not $procs) { exit 0 }; " ^
  "foreach ($p in $procs) { " ^
  "  try { " ^
  "    Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; " ^
  "  } catch { } " ^
  "}"

REM Also clear console python.exe run.py instances for this workspace (dev launches).
powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command ^
  "$root = [regex]::Escape((Resolve-Path -LiteralPath '%~dp0').Path); " ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $root) -and ($_.CommandLine -match 'run\.py') }; " ^
  "foreach ($p in @($procs)) { " ^
  "  try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; " ^
  "  } catch {} " ^
  "}"

endlocal
