// Thin desktop shell — the React frontend talks to dana/api/server.py
// directly over WebSocket (ws://localhost:8000/ws/chat); this crate just
// hosts the webview, it does not proxy any Dana traffic itself, and (see
// teardown_backend below) never SPAWNS the backend either — start_dana.bat
// launches Python separately, so closing this window has no way to reach
// it unless we go looking.
//
// tauri-plugin-store backs the Secrets menu (frontend/src/secrets) — API
// keys are persisted to a JSON file under the app's local data dir, not
// synced to the Dana backend. Plugin windows are spawned from JS directly
// (WebviewWindow) and don't need a dedicated Rust plugin; see
// frontend/src-tauri/capabilities/default.json for the permissions that
// allow that plus the emit/listen IPC events used to sync state to them.
//
// tauri-plugin-dialog backs Dynamic Workspace Mounting's native folder
// picker (WorkspacePlugin.tsx's "Mount Local Folder" button, via
// @tauri-apps/plugin-dialog's open({ directory: true })) — the absolute
// path it resolves is sent to dana/api/server.py's REST
// POST /api/workspace/mount, never handled here in Rust.
use std::path::{Path, PathBuf};
use std::process::Command;

/// Walk upward from `start` looking for the repo root — identified by
/// having BOTH `stop_dana.vbs` and `start_dana.bat` directly inside it
/// (the two root launcher wrappers; see start_dana.bat/stop_dana.vbs at
/// the repo root). Path-independent on purpose: `tauri dev`'s cwd is
/// `frontend/src-tauri` while a packaged build's `current_exe()` lives
/// somewhere else entirely, so this has to work from either starting
/// point without hardcoding a fixed number of `..` hops.
fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    for _ in 0..8 {
        if dir.join("stop_dana.vbs").is_file() && dir.join("start_dana.bat").is_file() {
            return Some(dir);
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    None
}

/// Hitting the main window's 'X' must kill the Python backend — this
/// crate never spawned it (see the module comment above), so the only way
/// to reach it from here is the SAME teardown stop_dana.vbs already
/// implements (a hidden PowerShell that targets pythonw.exe/python.exe
/// processes whose command line names launch_api_server.py/`-m dana`,
/// plus any stray Dana.exe). Shelling out to stop_dana.vbs instead of
/// re-filtering processes here in Rust keeps exactly ONE place that knows
/// how to recognize "a Dana process" — scripts/launchers/stop_dana.bat.
/// wscript.exe runs it with a hidden window (`WshShell.Run ..., 0, False`
/// inside the vbs itself) and this call doesn't wait for it, so window
/// teardown never delays the window from actually closing.
fn teardown_backend() {
    let candidates = [
        std::env::current_dir().ok(),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf())),
    ];
    for candidate in candidates.into_iter().flatten() {
        if let Some(root) = find_repo_root(&candidate) {
            let vbs = root.join("stop_dana.vbs");
            match Command::new("wscript.exe").arg("//B").arg(&vbs).spawn() {
                Ok(_) => eprintln!(
                    "[Dana] main window closed -- teardown dispatched via {}",
                    vbs.display()
                ),
                Err(err) => eprintln!(
                    "[Dana] main window closed -- failed to launch {}: {err}",
                    vbs.display()
                ),
            }
            return;
        }
    }
    eprintln!(
        "[Dana] main window closed -- could not locate stop_dana.vbs from cwd or exe dir; \
         backend process(es) may still be running"
    );
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .on_window_event(|window, event| {
            // Only the main chat window's close ends the whole session —
            // the "orb" overlay (see tauri.conf.json) can be closed and
            // reopened independently without tearing down the backend.
            if window.label() == "main" {
                if let tauri::WindowEvent::CloseRequested { .. } = event {
                    teardown_backend();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Dana");
}
