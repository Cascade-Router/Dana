import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { PLUGINS, getPlugin } from "./registry";
import { openPluginWindow } from "../windows/windowSync";
import type { PluginDefinition, PluginId } from "./types";

type PluginContextValue = {
  plugins: PluginDefinition[];
  /** Plugin rendered inline in the main window's PluginHost; null = chat-only. */
  activePluginId: PluginId | null;
  activatePlugin: (id: PluginId | null) => void;
  /** Pop a plugin out into its own top-level Tauri window instead of inline. */
  openInWindow: (id: PluginId) => void;
};

const PluginContext = createContext<PluginContextValue | null>(null);

export function PluginProvider({ children }: { children: ReactNode }) {
  const [activePluginId, setActivePluginId] = useState<PluginId | null>(null);

  const activatePlugin = useCallback((id: PluginId | null) => setActivePluginId(id), []);

  const openInWindow = useCallback((id: PluginId) => {
    const plugin = getPlugin(id);
    if (plugin) void openPluginWindow(plugin);
  }, []);

  const value = useMemo(
    () => ({ plugins: PLUGINS, activePluginId, activatePlugin, openInWindow }),
    [activePluginId, activatePlugin, openInWindow]
  );

  return <PluginContext.Provider value={value}>{children}</PluginContext.Provider>;
}

export function usePlugins(): PluginContextValue {
  const ctx = useContext(PluginContext);
  if (!ctx) throw new Error("usePlugins() must be used within <PluginProvider>");
  return ctx;
}
