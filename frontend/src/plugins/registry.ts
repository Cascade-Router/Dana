import { lazy } from "react";
import type { PluginDefinition } from "./types";

// Add new plugins here. Each one is lazy-loaded so its dependencies (three.js,
// @xyflow/react, ...) never touch the bundle Dana needs for a plain chat session.
export const PLUGINS: PluginDefinition[] = [
  {
    id: "cad",
    name: "CAD",
    description: "3D viewport + execution graph for FreeCAD-driven modeling.",
    glyph: "◆",
    accentColor: "#4f8ff7",
    Component: lazy(() => import("./CadPlugin")),
    window: { width: 1100, height: 760 },
  },
  {
    id: "workspace",
    name: "Workspace",
    description: "Read-only file tree + viewer for the agent's sandboxed workspace.",
    glyph: "▤",
    accentColor: "#34d399",
    Component: lazy(() => import("./WorkspacePlugin")),
    window: { width: 900, height: 640 },
  },
  {
    id: "coder",
    name: "Coder",
    description: "Software Engineering — codebase recon and Aider-driven code tasks, with commit/diff output.",
    glyph: "⌨",
    accentColor: "#f97316",
    Component: lazy(() => import("./CoderPlugin")),
    window: { width: 900, height: 700 },
  },
];

export function getPlugin(id: string): PluginDefinition | undefined {
  return PLUGINS.find((p) => p.id === id);
}
