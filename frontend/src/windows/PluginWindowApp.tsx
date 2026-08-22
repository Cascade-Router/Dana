import { Suspense } from "react";
import { getPlugin } from "../plugins/registry";
import type { PluginId } from "../plugins/types";
import { usePluginWindowSync } from "./windowSync";
import "./PluginWindowApp.css";

// Rendered inside a spawned plugin window (url `/#/plugin/:id`, see
// main.tsx). This window owns no state of its own — no WebSocket, no
// secrets store handle — it only mirrors whatever the main window last
// broadcast via useSyncBroadcaster, and forwards user interaction back to
// the main window as events. See windowSync.ts for the full contract.
export function PluginWindowApp({ pluginId }: { pluginId: PluginId }) {
  const plugin = getPlugin(pluginId);
  const { payload, sendSelect } = usePluginWindowSync();

  if (!plugin) {
    return <div className="plugin-window plugin-window--error">Unknown plugin: {pluginId}</div>;
  }

  if (!payload) {
    return <div className="plugin-window plugin-window--loading">Connecting to Dana…</div>;
  }

  const pluginState = payload.plugin?.pluginId === pluginId ? payload.plugin : null;

  return (
    <div className="plugin-window">
      <Suspense fallback={<div className="plugin-window--loading">Loading {plugin.name}…</div>}>
        <plugin.Component
          meshUrl={pluginState?.meshUrl ?? null}
          cameraTarget={pluginState?.cameraTarget ?? null}
          onSelect={sendSelect}
          log={[]}
        />
      </Suspense>
    </div>
  );
}
