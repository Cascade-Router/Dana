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
