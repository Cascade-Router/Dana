import { defineConfig, devices } from "@playwright/test";

// Spec files live at repo-root `tests/e2e/` (not `frontend/tests/e2e`) so
// they sit next to the Python integration suite (`tests/test_mesh_pipeline.py`)
// rather than inside the frontend's own `src/` tree — this config (and the
// installed `@playwright/test`/browser binaries) still lives in `frontend/`
// since that's where `npm`/`node_modules` already are.
export default defineConfig({
  testDir: "../tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    // Fixed by frontend/vite.config.ts (Tauri requires a predictable dev
    // port — see that file's own comment); NOT Vite's usual default 5173.
    baseURL: "http://localhost:1420",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // Both processes this test needs — the real dana.api.server backend (for
  // a real /ws/chat session_id; no FreeCAD calls are ever made, so no real
  // FreeCAD install is required for this suite) and the Vite dev server
  // (frontend/src, live/HMR — the whole point of this suite is asserting
  // against the actual current source, never a possibly-stale frontend/dist
  // build). `reuseExistingServer` locally means an already-running `npm run
  // dev`/backend (e.g. from `npm run tauri dev`) is left alone rather than
  // fighting over the same port.
  webServer: [
    {
      // The project's own .venv, not whatever bare `python` resolves to on
      // PATH — the backend needs its real installed deps (dotenv, fastapi,
      // uvicorn, ...), which only live in .venv here.
      command: ".venv\\Scripts\\python.exe scripts/launchers/launch_api_server.py",
      cwd: "..",
      // No dedicated /health route exists — dana.api.cad's artifact list is
      // a real, auth-free, always-200 GET (session_id is optional), so it
      // doubles as a readiness probe.
      url: "http://127.0.0.1:8000/api/cad/artifacts",
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      url: "http://localhost:1420",
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
