// Imported first, before anything else in this bundle, so window.console is
// patched (see consoleCapture.ts) before any other module's own top-level
// console.log calls can run — this is the actual SPA entry point shared by
// every window (main, orb, spawned plugin windows; see resolveRoot below),
// so this is the one place a plain `import` here is guaranteed to cover all
// of them, unlike importing it from App.tsx alone would (the orb/plugin-
// window roots never import App.tsx at all).
import "./lib/consoleCapture";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { OrbOverlay } from "./components/OrbOverlay";
import { PluginWindowApp } from "./windows/PluginWindowApp";
import type { PluginId } from "./plugins/types";

// Every Tauri window in this app (main, the always-on-top orb, and any
// dynamically spawned plugin window) shares this one SPA bundle — a hash
// route needs no server-side rewrite rule to serve it, unlike a path route
// would on the static frontend/dist mount. See src-tauri/tauri.conf.json
// for the "orb" window and windows/windowSync.ts's openPluginWindow() for
// how "/#/plugin/:id" windows get created.
const hash = window.location.hash;
const pluginMatch = hash.match(/^#\/plugin\/(.+)$/);

function resolveRoot(): React.ReactElement {
  if (hash === "#/orb") return <OrbOverlay />;
  if (pluginMatch) return <PluginWindowApp pluginId={pluginMatch[1] as PluginId} />;
  return <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{resolveRoot()}</React.StrictMode>
);
