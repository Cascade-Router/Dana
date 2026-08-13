// Thin desktop shell — the React frontend talks to dana/api/server.py
// directly over WebSocket (ws://localhost:8000/ws/chat); this crate just
// hosts the webview, it does not proxy any Dana traffic itself.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running Dana");
}
