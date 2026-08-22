import { useEffect } from "react";
import { AssistiveOrb } from "./AssistiveOrb";
import { useOrbWindowSync } from "../windows/windowSync";
import "./OrbOverlay.css";

// Window-level shell for the dedicated always-on-top "orb" Tauri window
// (see src-tauri/tauri.conf.json). AssistiveOrb itself is a plain,
// window-agnostic component — this file just supplies the transparent
// background and drag-to-move behavior that only make sense for a
// borderless HUD window.
//
// This window does NOT open its own WebSocket — it used to, which meant a
// voice-finalized transcript got dispatched twice (once per connected
// session) whenever this window happened to be open. It now mirrors the
// main window's voiceState purely over Tauri IPC (useOrbWindowSync) and
// relays clicks back the same way, consistent with every other spawned
// window's single-writer pattern (see windows/windowSync.ts).
export function OrbOverlay() {
  const { voice, activate } = useOrbWindowSync();

  useEffect(() => {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
  }, []);

  return (
    <div className="orb-overlay-window">
      <AssistiveOrb state={voice?.state ?? "idle"} transcript={voice?.transcript} onActivate={activate} />
    </div>
  );
}
