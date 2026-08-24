// Resolves where dana/api/server.py is reachable.
//
// - `VITE_API_BASE` env var always wins (explicit override for any setup).
// - In `npm run dev` / `npm run tauri dev`, Vite serves the frontend itself
//   (port 1420) so the API is a separate process — default to localhost:8000.
// - In a production build served BY the FastAPI process itself
//   (frontend/dist mounted at "/"), the API is same-origin.
const DEV_PORTS = new Set(["1420", "5173"]);

function resolveHttpBase(): string {
  const override = import.meta.env.VITE_API_BASE;
  if (override) return override.replace(/\/$/, "");

  const { protocol, hostname, port } = window.location;
  if (DEV_PORTS.has(port) || protocol === "tauri:") {
    return "http://localhost:8000";
  }
  return `${protocol}//${hostname}${port ? `:${port}` : ""}`;
}

// True for the same build useChat.ts picks useGradioChat for (see
// lib/useChat.ts) — the pure-Gradio HF backend (app.py) has no REST API at
// all, so anything hitting /api/* directly (not through useChat's
// abstraction) needs this same check to avoid a guaranteed failed fetch.
export const IS_GRADIO_MODE = Boolean(import.meta.env.VITE_HF_SPACE_URL);

export const API_HTTP_BASE = resolveHttpBase();

export const API_WS_BASE = API_HTTP_BASE.replace(/^http/, "ws");

export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_HTTP_BASE}${path}`;
}

export function resolveMeshUrl(meshUrl: string): string {
  return resolveApiUrl(meshUrl);
}
