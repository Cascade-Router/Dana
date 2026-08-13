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
  // (dana/api/server.py mounts frontend/dist/ at "/" when it exists),
  // while `src-tauri` still points Tauri's own bundler at the same output.
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
