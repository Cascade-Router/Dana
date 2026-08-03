@echo off
setlocal EnableExtensions
REM Project root is two levels above scripts\launchers\
cd /d "%~dp0..\.." >nul 2>&1

REM Silent STOP DANA: Hidden PowerShell + redirected console noise.
REM Targets only workspace-bound Dana/Donna app PIDs (not broad kills).
powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command ^
  "$root = (Resolve-Path -LiteralPath '%CD%').Path; " ^
  "$esc = [regex]::Escape($root); " ^
  "$pyw = Get-CimInstance Win32_Process -Filter \"Name = 'pythonw.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $esc) }; " ^
  "$py = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | " ^
  "  Where-Object { $_.CommandLine -and ($_.CommandLine -match $esc) -and ( " ^
  "    ($_.CommandLine -match 'run\.py') -or ($_.CommandLine -match '(?i)-m\s+dana\b') " ^
  "  ) }; " ^
  "$apps = @(); foreach ($n in @('Donna.exe','Dana.exe','dana.exe')) { " ^
  "  $apps += @(Get-CimInstance Win32_Process -Filter (\"Name = '{0}'\" -f $n) | " ^
  "    Where-Object { -not $_.CommandLine -or ($_.CommandLine -match $esc) -or ($_.CommandLine -match '(?i)[\\/]dana([\\/]|$)') }) " ^
  "}; " ^
  "$procs = @($pyw) + @($py) + @($apps); " ^
  "if (-not $procs) { exit 0 }; " ^
  "foreach ($p in $procs) { " ^
  "  try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch { } " ^
  "}" >nul 2>&1

endlocal
