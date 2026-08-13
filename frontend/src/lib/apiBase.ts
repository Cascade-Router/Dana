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

export const API_HTTP_BASE = resolveHttpBase();

export const API_WS_BASE = API_HTTP_BASE.replace(/^http/, "ws");

export function resolveMeshUrl(meshUrl: string): string {
  if (/^https?:\/\//.test(meshUrl)) return meshUrl;
  return `${API_HTTP_BASE}${meshUrl}`;
}
