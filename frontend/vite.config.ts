import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri expects a fixed, predictable port from `vite dev` — see
// https://v2.tauri.app/start/frontend/vite/
const HOST = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],

  // Vite options tailored for Tauri development.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: HOST || false,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  // Bundle a static web build the FastAPI server can serve directly
  // (dana/api/server.py mounts frontend/dist/ at "/" or "/ui" depending on
  // platform), while `src-tauri` still points Tauri's own bundler at the
  // same output. Relative base so the same build's asset URLs resolve
  // correctly under either mount prefix, and under Tauri's own asset
  // protocol (which serves dist/ at its own root, unrelated to FastAPI).
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
