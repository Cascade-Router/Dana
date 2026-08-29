import { useCallback, useEffect, useRef, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { emit, listen, type UnlistenFn } from "@tauri-apps/api/event";
import { WebviewWindow } from "@tauri-apps/api/webviewWindow";
import type { PluginDefinition, PluginId } from "../plugins/types";
import type { CameraTarget, CanvasSelection, VoiceState } from "../lib/useChatSocket";
import type { SecretRecord } from "../secrets/types";

// Every exported function here is guarded with `if (!isTauri()) return` —
// emit/listen/WebviewWindow throw synchronously (not a rejected promise a
// .catch() would catch) when window.__TAURI_INTERNALS__ doesn't exist,
// which is the case for every plain-browser build of this same frontend
// (a static Vercel/GitHub Pages deploy, the HF Space's /ui iframe, or this
// dev server) — only the actual Tauri desktop app has that global.
//
// Multi-window IPC contract. The main window is the single source of truth
// (it owns the WebSocket to dana/api/server.py and the secrets store) —
// plugin windows never open their own socket or store handle. Instead they
// render whatever the main window last broadcast, and hand user actions
// back to the main window as events for it to act on. This keeps state from
// forking across windows: there's exactly one writer for every piece of
// shared state, no matter how many windows are open. The always-on-top orb
// window follows the same rule: it used to open its own independent
// WebSocket purely to receive voice_state broadcasts, which meant a
// voice-finalized transcript got dispatched twice (once per connected
// session) whenever the orb window was open — it now gets voice state
// purely from this sync channel instead.
const SYNC_EVENT = "dana://sync";
const CHILD_READY_EVENT = "dana://child-ready";
const CAD_SELECT_EVENT = "dana://cad-select";
const ORB_ACTIVATE_EVENT = "dana://orb-activate";

export type PluginSyncState = {
  pluginId: PluginId;
  meshUrl: string | null;
  cameraTarget: CameraTarget | null;
  sessionId: string | null;
};

export type VoiceSyncState = {
  state: VoiceState;
  transcript: string;
};

export type SyncPayload = {
  secrets: Record<string, SecretRecord>;
  plugin: PluginSyncState | null;
  voice: VoiceSyncState;
};

function pluginWindowLabel(pluginId: PluginId): string {
  return `plugin-${pluginId}`;
}

/** Opens (or focuses, if already open) a plugin's own top-level window. */
export async function openPluginWindow(plugin: PluginDefinition): Promise<void> {
  if (!isTauri()) return; // no multi-window host outside the desktop app
  const label = pluginWindowLabel(plugin.id);
  const existing = await WebviewWindow.getByLabel(label);
  if (existing) {
    await existing.setFocus();
    return;
  }

  new WebviewWindow(label, {
    url: `/#/plugin/${plugin.id}`,
    title: `Dana — ${plugin.name}`,
    width: plugin.window.width,
    height: plugin.window.height,
    minWidth: 480,
    minHeight: 360,
  });
}

/**
 * Main-window hook: re-broadcasts `state` to every open window whenever it
 * changes, and replies immediately when a child window announces it just
 * mounted (so a window opened after the last state change isn't stuck blank).
 */
export function useSyncBroadcaster(state: SyncPayload): void {
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    if (!isTauri()) return;
    emit(SYNC_EVENT, state).catch(() => {});
  }, [state]);

  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: UnlistenFn | undefined;
    listen(CHILD_READY_EVENT, () => {
      emit(SYNC_EVENT, stateRef.current).catch(() => {});
    }).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, []);
}

/**
 * Main-window hook: forwards selections made in a *spawned* plugin window's
 * canvas back into the real onSelect handler (the one wired to the socket).
 */
export function useForwardedPluginSelect(onSelect: (selection: CanvasSelection) => void): void {
  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: UnlistenFn | undefined;
    listen<CanvasSelection>(CAD_SELECT_EVENT, (event) => onSelect(event.payload)).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, [onSelect]);
}

/** Plugin-window hook: mirrors whatever the main window last broadcast. */
export function usePluginWindowSync(): { payload: SyncPayload | null; sendSelect: (s: CanvasSelection) => void } {
  const [payload, setPayload] = useState<SyncPayload | null>(null);

  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: UnlistenFn | undefined;
    listen<SyncPayload>(SYNC_EVENT, (event) => setPayload(event.payload)).then((fn) => {
      unlisten = fn;
      emit(CHILD_READY_EVENT).catch(() => {});
    });
    return () => unlisten?.();
  }, []);

  const sendSelect = useCallback((selection: CanvasSelection) => {
    if (!isTauri()) return;
    emit(CAD_SELECT_EVENT, selection).catch(() => {});
  }, []);

  return { payload, sendSelect };
}

/**
 * Main-window hook: forwards an orb click/hotkey made in the *spawned* orb
 * window back into the real requestListen()/cancelListen() handler (the one
 * wired to the socket) — mirrors useForwardedPluginSelect's role for CAD.
 */
export function useForwardedOrbActivate(onActivate: () => void): void {
  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: UnlistenFn | undefined;
    listen(ORB_ACTIVATE_EVENT, () => onActivate()).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, [onActivate]);
}

/** Orb-window hook: mirrors the main window's voice state, and relays clicks back to it. */
export function useOrbWindowSync(): { voice: VoiceSyncState | null; activate: () => void } {
  const [voice, setVoice] = useState<VoiceSyncState | null>(null);

  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: UnlistenFn | undefined;
    listen<SyncPayload>(SYNC_EVENT, (event) => setVoice(event.payload.voice)).then((fn) => {
      unlisten = fn;
      emit(CHILD_READY_EVENT).catch(() => {});
    });
    return () => unlisten?.();
  }, []);

  const activate = useCallback(() => {
    if (!isTauri()) return;
    emit(ORB_ACTIVATE_EVENT).catch(() => {});
  }, []);

  return { voice, activate };
}
