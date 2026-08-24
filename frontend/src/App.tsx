import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { ChatSidebar } from "./components/ChatSidebar";
import { EnvViewerWidget } from "./components/EnvViewerWidget";
import { TerminalDrawer } from "./components/TerminalDrawer";
import { PluginProvider, usePlugins } from "./plugins/PluginContext";
import { SecretsProvider, useSecrets } from "./secrets/SecretsContext";
import { SecretsMenu } from "./secrets/SecretsMenu";
import { resolveApiUrl } from "./lib/apiBase";
import { useChat } from "./lib/useChat";
import type { ChatMessage } from "./lib/useChatSocket";
import { useOrbActivation } from "./lib/useOrbActivation";
import {
  useForwardedOrbActivate,
  useForwardedPluginSelect,
  useSyncBroadcaster,
  type SyncPayload,
} from "./windows/windowSync";
import "./App.css";

function AppShell() {
  const { plugins, activePluginId, activatePlugin } = usePlugins();
  const { entries } = useSecrets();
  const [secretsOpen, setSecretsOpen] = useState(false);
  const [envViewerOpen, setEnvViewerOpen] = useState(false);

  // Split-view (❐): "split" is today's existing chat+plugin layout (now at
  // an explicit 45/55 ratio instead of the old fixed sidebar width) —
  // "full" hides the chat pane entirely so the active plugin alone fills
  // app__body. Persists across tab switches (switching from CAD to Coder
  // while in "full" keeps Coder full too) rather than resetting per-tab,
  // since that's the simpler, more predictable default and nothing in the
  // spec calls for per-tab memory.
  const [paneMode, setPaneMode] = useState<"split" | "full">("split");
  const toggleSplitView = useCallback(
    (id: (typeof plugins)[number]["id"]) => {
      if (activePluginId !== id) {
        activatePlugin(id);
        setPaneMode("split");
        return;
      }
      setPaneMode((m) => (m === "split" ? "full" : "split"));
    },
    [activePluginId, activatePlugin]
  );

  // BYOK: the backend session only ever sees the value for known,
  // non-"custom" services (openai/anthropic/elevenlabs) — dana.api.server
  // only knows what to do with those two today (model_provider.py,
  // cad_vision.py); an arbitrary user-labeled "custom" key has nowhere to
  // go yet, so it stays frontend-only rather than being sent for no reason.
  const apiKeys = useMemo(
    () => Object.fromEntries(entries.filter((e) => e.service !== "custom").map((e) => [e.service, e.value])),
    [entries]
  );

  // Capability routing: today only one plugin can be inline-active at a
  // time (activePluginId is singular — see PluginContext), so this is
  // always a 0- or 1-element array, but the wire format is a list so the
  // backend/session model isn't locked to "at most one plugin ever."
  const activePlugins = useMemo(() => (activePluginId ? [activePluginId] : []), [activePluginId]);

  // Local Chat Session Persistence: `requestedSessionId` is what
  // useChatSocket connects /ws/chat with (?session_id=...) — null lets the
  // server generate a fresh one. `initialMessages` is that session's
  // already-fetched history (GET /api/sessions/{id}), set right before
  // requestedSessionId changes so the new connection seeds its chat UI
  // with it immediately (see useChatSocket's connect effect).
  const [requestedSessionId, setRequestedSessionId] = useState<string | null>(null);
  const [initialMessages, setInitialMessages] = useState<ChatMessage[]>([]);

  const startNewChat = useCallback(() => {
    setInitialMessages([]);
    setRequestedSessionId(null);
  }, []);

  const openSession = useCallback((id: string) => {
    fetch(resolveApiUrl(`/api/sessions/${id}`))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const history: ChatMessage[] = (data.session?.messages ?? []).map(
          (m: { role: string; content: string }) => ({
            role: m.role === "user" ? "user" : "assistant",
            content: m.content,
          })
        );
        setInitialMessages(history);
        setRequestedSessionId(id);
      })
      .catch(() => {
        // Best-effort — a stale/deleted session id (e.g. removed from
        // another window) just no-ops instead of crashing the chat UI.
      });
  }, []);

  const {
    connection,
    messages,
    log,
    meshUrl,
    cameraTarget,
    voiceState,
    liveActivity,
    turnActive,
    sessionId,
    sendMessage,
    abortTurn,
    sendSelection,
    respondHitl,
    requestListen,
    cancelListen,
  } = useChat(apiKeys, activePlugins, requestedSessionId, initialMessages);

  const activePlugin = plugins.find((p) => p.id === activePluginId) ?? null;
  const isPluginFullScreen = activePlugin !== null && paneMode === "full";

  // Model indicator badge: polls the exact same source of truth
  // _call_llm_once resolves its own provider from (dana.core.model_provider.
  // tool_calling_provider, surfaced on /api/health as "provider") — a
  // status display, never a second guess at what a turn will actually do.
  const [provider, setProvider] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetch(resolveApiUrl("/api/health"))
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (!cancelled && data) setProvider(typeof data.provider === "string" ? data.provider : null);
        })
        .catch(() => {
          if (!cancelled) setProvider(null);
        });
    };
    poll();
    const interval = window.setInterval(poll, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);
  const providerLabel = provider === "gateway" ? "Cascade Proxy: Active" : provider ? `${provider}: Active` : "Offline";

  // Single broadcast point: any plugin/orb window currently open re-renders
  // from this payload the instant it changes. Nothing downstream reaches
  // back into the socket or the secrets store directly.
  const syncPayload: SyncPayload = useMemo(
    () => ({
      secrets: Object.fromEntries(entries.map((e) => [e.id, e])),
      plugin: activePluginId ? { pluginId: activePluginId, meshUrl, cameraTarget } : null,
      voice: { state: voiceState.state, transcript: voiceState.transcript },
    }),
    [entries, activePluginId, meshUrl, cameraTarget, voiceState]
  );
  useSyncBroadcaster(syncPayload);
  useForwardedPluginSelect(sendSelection);

  // One activation policy shared by the dedicated always-on-top orb
  // window's clicks (relayed over Tauri IPC — see useForwardedOrbActivate)
  // and the in-app hotkey: idle -> start listening, listening -> cancel.
  // The main window itself renders no orb of its own — see OrbOverlay.tsx
  // for the single place the orb actually appears, so the two windows
  // never show it twice at once.
  const activateOrb = useOrbActivation(voiceState.state, requestListen, cancelListen);
  useForwardedOrbActivate(activateOrb);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "l") {
        e.preventDefault();
        activateOrb();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activateOrb]);

  return (
    <div className={`app ${activePlugin ? "app--with-plugin" : ""} ${isPluginFullScreen ? "app--plugin-fullscreen" : ""}`}>
      <div className="app__topbar">
        <button
          type="button"
          className={`app__tab ${activePluginId === null ? "app__tab--active" : ""}`}
          onClick={() => activatePlugin(null)}
        >
          Chat
        </button>
        {plugins.map((plugin) => (
          <div key={plugin.id} className="app__tab-group">
            <button
              type="button"
              className={`app__tab ${activePluginId === plugin.id ? "app__tab--active" : ""}`}
              style={{ "--tab-accent": plugin.accentColor } as React.CSSProperties}
              onClick={() => activatePlugin(plugin.id)}
            >
              <span className="app__tab-glyph">{plugin.glyph}</span>
              {plugin.name}
            </button>
            <button
              type="button"
              className={`app__split-toggle ${activePluginId === plugin.id && paneMode === "full" ? "app__split-toggle--full" : ""}`}
              title={
                activePluginId === plugin.id && paneMode === "full"
                  ? `Show split view (Chat + ${plugin.name})`
                  : `Full-screen ${plugin.name}`
              }
              onClick={() => toggleSplitView(plugin.id)}
            >
              ❐
            </button>
          </div>
        ))}
        <div className="app__topbar-spacer" />
        <div className="app__model-badge" title={`Active tool-calling provider: ${provider ?? "unknown"}`}>
          ⚡ {providerLabel}
        </div>
        <button
          type="button"
          className="app__secrets-btn"
          title="Environment Variables"
          onClick={() => setEnvViewerOpen(true)}
        >
          🔑
        </button>
        <button type="button" className="app__secrets-btn" title="Secrets" onClick={() => setSecretsOpen(true)}>
          ⚙
        </button>
      </div>

      <div className="app__body">
        {!isPluginFullScreen && (
          <div className="app__left">
            <ChatSidebar activeSessionId={sessionId} onNewChat={startNewChat} onSelectSession={openSession} />
            <div className="app__chat-column">
              <ChatPanel
                connection={connection}
                messages={messages}
                liveActivity={liveActivity}
                turnActive={turnActive}
                onSend={sendMessage}
                onAbort={abortTurn}
                onHitlRespond={respondHitl}
              />
            </div>
          </div>
        )}
        {activePlugin && (
          <div className="app__right">
            <Suspense fallback={<div className="app__plugin-loading">Loading {activePlugin.name}…</div>}>
              <activePlugin.Component
                meshUrl={meshUrl}
                cameraTarget={cameraTarget}
                onSelect={sendSelection}
                log={log}
              />
            </Suspense>
          </div>
        )}
      </div>

      {secretsOpen && <SecretsMenu onClose={() => setSecretsOpen(false)} />}
      {envViewerOpen && <EnvViewerWidget onClose={() => setEnvViewerOpen(false)} />}
      <TerminalDrawer log={log} />
    </div>
  );
}

export default function App() {
  return (
    <SecretsProvider>
      <PluginProvider>
        <AppShell />
      </PluginProvider>
    </SecretsProvider>
  );
}
