# Dana frontend (Tauri + React + Vite)

Talks to `dana/api/server.py` over WebSocket (`/ws/chat`) and HTTP (`/api/*`).
This directory was hand-authored (not generated via `npm create tauri-app`)
because Node.js/npm and Rust/Cargo aren't available in the environment that
built it — **you must install both locally before any of this runs.**

## Prerequisites

- Node.js 18+ and npm
- Rust + Cargo (https://www.rust-lang.org/tools/install)
- Tauri v2 system dependencies for your OS: https://v2.tauri.app/start/prerequisites/
  (on Windows: the WebView2 runtime — usually already present — plus the
  Visual Studio C++ Build Tools)

## First-time setup

```sh
cd frontend
npm install
```

## Phase 3 verification (do this before any legacy UI is deleted)

1. **Start the backend** (from the repo root, in a separate terminal):

   ```sh
   python scripts/launchers/launch_api_server.py
   ```

   Confirm it's up: `curl http://localhost:8000/api/health` should return
   `{"ok": true, ...}` with `control_plane`/`cad_engine` naming your real
   drivers (`Win32ControlPlane` / `RealFreeCADEngine` on Windows).

2. **Start the Tauri dev app** (from `frontend/`):

   ```sh
   npm run tauri dev
   ```

   This boots Vite on `http://localhost:1420` and opens it in a native
   Tauri window. (`npm run dev` alone also works if you just want the page
   in a browser tab instead of the desktop shell.)

3. **Verify tool dispatch end-to-end**: in the chat panel, send
   `Create a 10x10x10 box` (or click the quick-prompt "Build a box
   60x40x20"). You should see, in order:
   - a `tool_call` line appear in the Terminal Drawer at the bottom-left
   - a `tool_result` line with `ok=True`
   - the 3D viewer on the right load and display the generated box mesh
   - an assistant reply naming the driver that created it

   If the tool_result's `ok` is `False`, or the drawer never receives a
   `tool_call`, check the backend terminal for the traceback — the FastAPI
   process logs any dispatch exception.

4. **Verify OS actuation**: send `Resync the workspace` or `System status`
   and confirm the reply reports your real driver names (not `Mock*`) when
   running on the target desktop OS, and that no exception is raised.

Only once all of the above passes should Phase 4 (deleting the legacy
Gradio/Tkinter UI) proceed.

## Production build

```sh
npm run build        # emits frontend/dist — dana/api/server.py auto-mounts
                      # this at "/" if present, so the FastAPI process alone
                      # can serve the whole app (e.g. in a Docker deploy).
npm run tauri build  # native desktop bundle (msi/nsis/dmg/AppImage per OS)
```

Set `VITE_API_BASE` at build time if the backend isn't reachable at
`http://localhost:8000` in your deployment (e.g. `VITE_API_BASE=https://api.example.com npm run build`).

## Icons

`src-tauri/icons/*` were generated from `assets/dana_logo.png`. Regenerate
with `npm run tauri icon ../../assets/dana_logo.png` from `frontend/` if the
source logo changes.
