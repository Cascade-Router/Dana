#!/usr/bin/env bash
# Dana desktop bootstrap (Linux/macOS).
#
# Auto-discovers a local FreeCAD install and bridges it into the environment
# BEFORE the backend/Tauri stack boots, so a missing/non-standard FreeCAD
# install is surfaced immediately with a clear, friendly message instead of
# silently degrading deep inside a chat turn's first CAD tool call.
#
# Unlike Windows (scripts/launchers/start_dana.bat -> start_dana_detached.ps1
# already exists and is registered for Windows Startup), there is no
# existing Linux/macOS pipeline that launches the native Tauri window (only
# start_dana.py's `npm run dev`, the plain web dev server) — so this script
# owns backend + frontend process lifecycle itself: foreground, Ctrl+C stops
# both.
#
# Usage:
#   chmod +x launch_dana.sh   # once, if the executable bit didn't survive checkout
#   ./launch_dana.sh
# or simply:
#   bash launch_dana.sh

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- FreeCAD auto-discovery -------------------------------------------------
# Mirrors dana/plugins/freecad/engine.py's detect_freecadcmd(): env override
# > PATH > common install globs for this OS (newest wins). Kept in sync
# manually with that function's own _COMMON_INSTALL_GLOBS — if one changes,
# update the other.
FREECAD_PATH=""

if [ -n "${DANA_FREECADCMD_PATH:-}" ] && [ -f "${DANA_FREECADCMD_PATH}" ]; then
    FREECAD_PATH="$DANA_FREECADCMD_PATH"
fi

if [ -z "$FREECAD_PATH" ]; then
    FOUND_ON_PATH="$(command -v freecadcmd 2>/dev/null || true)"
    if [ -z "$FOUND_ON_PATH" ]; then
        FOUND_ON_PATH="$(command -v FreeCADCmd 2>/dev/null || true)"
    fi
    FREECAD_PATH="$FOUND_ON_PATH"
fi

if [ -z "$FREECAD_PATH" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        # The official freecad.org macOS build is a plain .app bundle
        # dragged into /Applications — never on PATH by default, unlike a
        # Homebrew install (already caught by `command -v` above).
        CANDIDATE_PATTERNS=(
            "/Applications/FreeCAD*.app/Contents/Resources/bin/FreeCADCmd"
            "/Applications/FreeCAD*.app/Contents/MacOS/FreeCADCmd"
            "$HOME/Applications/FreeCAD*.app/Contents/Resources/bin/FreeCADCmd"
        )
    else
        # Linux desktop installs outside a package manager (the official
        # freecad.org AppImage extracted to a fixed prefix, or a manual
        # /opt install). An apt/dnf-managed install already lands
        # freecadcmd on PATH and is caught by `command -v` above.
        CANDIDATE_PATTERNS=(
            "/opt/freecad*/bin/freecadcmd"
            "/opt/FreeCAD*/bin/freecadcmd"
            "/usr/lib/freecad*/bin/freecadcmd"
            "$HOME/.local/opt/freecad*/bin/freecadcmd"
        )
    fi
    for pattern in "${CANDIDATE_PATTERNS[@]}"; do
        for match in $pattern; do
            if [ -f "$match" ]; then
                FREECAD_PATH="$match"
                break 2
            fi
        done
    done
fi

if [ -n "$FREECAD_PATH" ]; then
    echo "[Dana] Found FreeCAD: $FREECAD_PATH"
    export DANA_FREECADCMD_PATH="$FREECAD_PATH"
else
    echo ""
    echo "================================================================="
    echo " [Dana] FreeCAD was not found on this machine."
    echo " CAD tool calls (create_box, apply_boolean, etc.) will fail with a"
    echo " clear per-call error until it's installed — everything else"
    echo " (chat, other tools) still works normally."
    echo ""
    echo " Download FreeCAD (free, official): https://www.freecad.org/downloads.php"
    echo ""
    echo " Installed it somewhere non-standard? Set DANA_FREECADCMD_PATH to the"
    echo " full path of the freecadcmd binary before re-running this script."
    echo "================================================================="
    echo ""
fi

# --- Preconditions -----------------------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "[Dana] ERROR: .venv not found. Run \"python3 -m venv .venv\" and install requirements.txt first."
    exit 1
fi
if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "[Dana] ERROR: frontend/node_modules not found. Run \"npm install\" inside frontend/ first."
    exit 1
fi

# --- Launch backend + native Tauri window -----------------------------------
mkdir -p "$ROOT/logs"
echo "[Dana] Starting FastAPI backend..."
"$ROOT/.venv/bin/python" "$ROOT/scripts/launchers/launch_api_server.py" \
    > "$ROOT/logs/dana_backend.log" 2> "$ROOT/logs/dana_backend.err.log" &
BACKEND_PID=$!

cleanup() {
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[Dana] Stopping backend (pid $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# Give uvicorn a moment to bind before the Tauri webview's first request —
# mirrors start_dana_detached.ps1's own 2s gap on Windows.
sleep 2

echo "[Dana] Starting native Tauri app (npm run tauri -- dev)..."
cd "$ROOT/frontend"
npm run tauri -- dev
