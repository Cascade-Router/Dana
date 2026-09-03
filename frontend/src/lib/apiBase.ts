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

const API_HTTP_BASE = resolveHttpBase();

export const API_WS_BASE = API_HTTP_BASE.replace(/^http/, "ws");

export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${API_HTTP_BASE}${path}`;
}

// Every /api/* call in this app already goes through resolveApiUrl (grep
// confirms it — no raw fetch("/api/...") anywhere), so this is the one
// place that needs to know the pure-Gradio HF backend has no REST API at
// all. On Vercel a same-origin /api/* request doesn't 404 — the SPA
// rewrite (vercel.json) serves index.html for it instead, so
// `.then(res => res.json())` blows up on HTML with "Unexpected token '<'"
// rather than failing cleanly. Rejecting here — not resolving with a
// faked Response — deliberately reuses every call site's EXISTING
// .catch()/try-catch (a real network failure already has to go
// somewhere); a faked Response risks a shape mismatch a genuine caller
// never hits (e.g. .map() on a guessed-wrong empty payload).
export function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  if (IS_GRADIO_MODE && path.startsWith("/api/")) {
    return Promise.reject(new Error(`REST API unavailable in Gradio mode: ${path}`));
  }
  return fetch(resolveApiUrl(path), init);
}

export function resolveMeshUrl(meshUrl: string): string {
  return resolveApiUrl(meshUrl);
}
