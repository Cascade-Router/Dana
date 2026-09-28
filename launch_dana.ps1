# Dana desktop bootstrap (Windows).
#
# Auto-discovers a local FreeCAD install and bridges it into the environment
# BEFORE the backend/Tauri stack boots, so a missing/non-standard FreeCAD
# install is surfaced immediately with a clear, friendly message instead of
# silently degrading deep inside a chat turn's first CAD tool call.
#
# Deliberately does NOT reimplement backend/frontend process orchestration —
# scripts\launchers\start_dana.bat -> start_dana_detached.ps1 is already the
# real, tested pipeline (it's what scripts\launchers\register_startup.py
# registers for Windows Startup), so this script's only job is the FreeCAD
# preflight, then handing off to that existing pipeline. Two independent
# copies of "how do we launch the Tauri app" would only drift apart.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File launch_dana.ps1

$root = $PSScriptRoot

# --- FreeCAD auto-discovery -------------------------------------------------
# Mirrors dana/plugins/freecad/engine.py's detect_freecadcmd(): env override
# > PATH > common Windows install globs (newest wins). Kept in sync manually
# with that function's own _COMMON_INSTALL_GLOBS — if one changes, update
# the other.
$freecadEnvOverride = $env:DANA_FREECADCMD_PATH
$freecadPath = $null

if ($freecadEnvOverride -and (Test-Path -LiteralPath $freecadEnvOverride -PathType Leaf)) {
    $freecadPath = $freecadEnvOverride
}

if (-not $freecadPath) {
    $onPath = Get-Command "FreeCADCmd.exe" -ErrorAction SilentlyContinue
    if ($onPath) {
        $freecadPath = $onPath.Source
    }
}

if (-not $freecadPath) {
    $installGlobs = @(
        "C:\Program Files\FreeCAD*\bin\FreeCADCmd.exe",
        "C:\Program Files (x86)\FreeCAD*\bin\FreeCADCmd.exe"
    )
    $candidates = @()
    foreach ($pattern in $installGlobs) {
        $candidates += Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    }
    if ($candidates.Count -gt 0) {
        # Newest install wins — same "highest version folder" tiebreak
        # detect_freecadcmd's own candidates.sort(..., reverse=True) uses.
        $freecadPath = ($candidates | Sort-Object FullName -Descending | Select-Object -First 1).FullName
    }
}

if ($freecadPath) {
    Write-Host "[Dana] Found FreeCAD: $freecadPath" -ForegroundColor Green
    if (-not $freecadEnvOverride) {
        $env:DANA_FREECADCMD_PATH = $freecadPath
    }
} else {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Yellow
    Write-Host " [Dana] FreeCAD was not found on this machine." -ForegroundColor Yellow
    Write-Host " CAD tool calls (create_box, apply_boolean, etc.) will fail with a" -ForegroundColor Yellow
    Write-Host " clear per-call error until it's installed — everything else (chat," -ForegroundColor Yellow
    Write-Host " other tools) still works normally." -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Download FreeCAD (free, official): https://www.freecad.org/downloads.php" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Installed it somewhere non-standard? Set DANA_FREECADCMD_PATH to the" -ForegroundColor Yellow
    Write-Host " full path of FreeCADCmd.exe before re-running this script." -ForegroundColor Yellow
    Write-Host "=================================================================" -ForegroundColor Yellow
    Write-Host ""
}

# --- Hand off to the existing, already-tested launch pipeline --------------
$startBat = Join-Path $root "scripts\launchers\start_dana.bat"
if (-not (Test-Path -LiteralPath $startBat)) {
    Write-Host "[Dana] ERROR: $startBat not found." -ForegroundColor Red
    exit 1
}

Write-Host "[Dana] Launching backend + native Tauri app..." -ForegroundColor Cyan
# cmd.exe /c, not calling the .bat directly — consistent with how this
# repo's own scripts already invoke .bat/.cmd shims (see start_dana.py's
# start_frontend(), which does the same for npm.cmd on Windows).
& cmd.exe /c "`"$startBat`""
