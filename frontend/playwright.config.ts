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
    // A dedicated Vite instance (see webServer below), not the usual 1420
    // dev server, so a running `npm run dev` / `tauri dev` pointed at the
    // real backend is never reused by mistake.
    baseURL: "http://localhost:1421",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // The real dana.api.server, but started by tests/e2e/mock_backend.py: the
  // trimesh mock CAD engine, a scripted LLM (streams slowly, reports token
  // usage) and a throwaway temp workspace — so a UI test can run whole CAD
  // turns, rebuilds and exports without FreeCAD, an API key, or ever writing
  // into the real freecad_output/, exports/ or agent_workspace/. Never
  // reused: both ports belong to this suite only.
  webServer: [
    {
      command: ".venv\\Scripts\\python.exe tests/e2e/mock_backend.py --port 8765",
      cwd: "..",
      // dana.api.cad's artifact list: auth-free and always 200, so it
      // doubles as a readiness probe.
      url: "http://127.0.0.1:8765/api/cad/artifacts",
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: "npm run dev -- --port 1421",
      url: "http://localhost:1421",
      env: { VITE_API_BASE: "http://127.0.0.1:8765" },
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
