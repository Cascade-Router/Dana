import type { ComponentType, LazyExoticComponent } from "react";
import type { CameraTarget, CanvasSelection, ServerEvent } from "../lib/useChatSocket";

export type PluginId = "cad" | "workspace" | "coder";

// Shared props every plugin component receives, whether it's rendered
// inline in the main window or as the sole contents of a spawned window.
// Callers own where the data comes from (a live socket, or a synced copy
// relayed over Tauri IPC) — the plugin component itself doesn't care.
//
// This shape is still CAD-flavored (meshUrl/cameraTarget/onSelect/log) from
// when CAD was the only plugin. WorkspacePlugin (and any future plugin
// with different needs) simply accepts and ignores the fields it doesn't
// use — valid and type-safe, but worth generalizing (e.g. a per-plugin
// generic prop type) once a third plugin actually needs something this
// shape doesn't offer, rather than growing this union speculatively now.
export type PluginComponentProps = {
  meshUrl: string | null;
  cameraTarget: CameraTarget | null;
  onSelect: (selection: CanvasSelection) => void;
  log: ServerEvent[];
  /** The active chat's session_id — CadPlugin's Scoped Mini-Explorer
   * (CadToolbar) needs this to only ever list/open THIS session's own
   * CAD artifacts, never another chat's. `null` before the very first
   * WebSocket "ready" event of a brand-new chat. */
  sessionId: string | null;
};

export type PluginDefinition = {
  id: PluginId;
  name: string;
  description: string;
  /** Single letter/short glyph shown in the plugin tab and window switcher — no icon font dependency. */
  glyph: string;
  accentColor: string;
  Component: LazyExoticComponent<ComponentType<PluginComponentProps>>;
  window: {
    width: number;
    height: number;
  };
};
