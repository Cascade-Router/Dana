@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [stop_donna] Stopping CAMGRASPER Donna pythonw.exe processes...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [regex]::Escape((Resolve-Path -LiteralPath '%~dp0').Path); " ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name = 'pythonw.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $root) }; " ^
  "if (-not $procs) { Write-Host '[stop_donna] No matching pythonw.exe processes.'; exit 0 }; " ^
  "foreach ($p in $procs) { " ^
  "  try { " ^
  "    Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; " ^
  "    Write-Host ('[stop_donna] Killed PID ' + $p.ProcessId + ' :: ' + $p.CommandLine); " ^
  "  } catch { " ^
  "    Write-Host ('[stop_donna] WARNING: could not kill PID ' + $p.ProcessId + ': ' + $_.Exception.Message); " ^
  "  } " ^
  "}"

REM Also clear console python.exe run.py instances for this workspace (dev launches).
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [regex]::Escape((Resolve-Path -LiteralPath '%~dp0').Path); " ^
  "$procs = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $root) -and ($_.CommandLine -match 'run\.py') }; " ^
  "foreach ($p in @($procs)) { " ^
  "  try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; " ^
  "    Write-Host ('[stop_donna] Killed python PID ' + $p.ProcessId); " ^
  "  } catch {} " ^
  "}"

echo [stop_donna] Done.
endlocal
