import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { OrbOverlay } from "./components/OrbOverlay";

// The floating voice-orb overlay is a separate always-on-top Tauri window
// (see src-tauri/tauri.conf.json's "orb" window, url "/#/orb") sharing this
// same SPA bundle — a hash route needs no server-side rewrite rule to serve
// it, unlike a path route would on the static frontend/dist mount.
const isOrbWindow = window.location.hash === "#/orb";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>{isOrbWindow ? <OrbOverlay /> : <App />}</React.StrictMode>
);
