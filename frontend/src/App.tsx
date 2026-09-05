import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { Brain, ClipboardList, Key, Lock, Zap } from "lucide-react";
import { ChatPanel } from "./components/ChatPanel";
import { CostBar } from "./components/CostBar";
import { PlanChecklist } from "./components/PlanChecklist";
import { ChatSidebar } from "./components/ChatSidebar";
import { EnvViewerWidget } from "./components/EnvViewerWidget";
import { TerminalDrawer } from "./components/TerminalDrawer";
import { MemoryViewer } from "./components/MemoryViewer";
import { WebDemoBanner } from "./components/WebDemoBanner";
import { PluginProvider, usePlugins } from "./plugins/PluginContext";
import { SecretsProvider, useSecrets } from "./secrets/SecretsContext";
import { SecretsMenu } from "./secrets/SecretsMenu";
import { apiFetch, IS_GRADIO_MODE } from "./lib/apiBase";
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
  const [planOpen, setPlanOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);

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
    apiFetch(`/api/sessions/${id}`)
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
    costState,
    planState,
    memoryState,
    topologyGraph,
    liveActivity,
    turnActive,
    sessionId,
    autoApprove,
    sendMessage,
    abortTurn,
    sendSelection,
    respondHitl,
    requestListen,
    cancelListen,
    setAutoApprove,
  } = useChat(apiKeys, activePlugins, requestedSessionId, initialMessages);

  const activePlugin = plugins.find((p) => p.id === activePluginId) ?? null;
  const isPluginFullScreen = activePlugin !== null && paneMode === "full";
  // CadPlugin's InspectorDock now hosts Active Plan/Terminal itself (see
  // CadPlugin.tsx) — the global floating badge/overlay/FAB below would just
  // duplicate those over the same viewport, so they're suppressed only
  // while CAD is the active tab. Workspace/coder plugins don't render an
  // InspectorDock, so they keep the global chrome unchanged.
  const isCadActive = activePlugin?.id === "cad";

  // Model indicator badge: polls the exact same source of truth
  // _call_llm_once resolves its own provider from (dana.core.model_provider.
  // tool_calling_provider, surfaced on /api/health as "provider") — a
  // status display, never a second guess at what a turn will actually do.
  const [provider, setProvider] = useState<string | null>(null);
  // Slow Generation Warning System: the local model name, only meaningful
  // once `provider === "ollama"` — see /api/health's own comment. Polled
  // alongside `provider` rather than a second endpoint.
  const [localModel, setLocalModel] = useState<string | null>(null);
  useEffect(() => {
    // No /api/health on the pure-Gradio HF backend — providerLabel below
    // hardcodes a status for this mode instead of polling for one.
    if (IS_GRADIO_MODE) return;
    let cancelled = false;
    const poll = () => {
      apiFetch("/api/health")
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (!cancelled && data) {
            setProvider(typeof data.provider === "string" ? data.provider : null);
            setLocalModel(typeof data.local_model === "string" ? data.local_model : null);
          }
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
  const providerLabel = IS_GRADIO_MODE
    ? "HF ZeroGPU: Online"
    : provider
      ? `${provider}: Active`
      : "Offline";

  // Slow Generation System (heavier local models, run for exact-artifact
  // accuracy over speed, can take 2-3 minutes per turn — see
  // dana.core.react_dispatch's _LOCAL_TOOL_CALL_TIMEOUT_SEC): shown only
  // while a turn is actually in flight against the local Ollama provider,
  // so the user sees it exactly while it's relevant — "still working," not
  // "the app is frozen" — and it disappears the moment a reply lands.
  const isLocalInferenceActive = provider === "ollama";
  const showSlowGenerationWarning = isLocalInferenceActive && turnActive;

  // Single broadcast point: any plugin/orb window currently open re-renders
  // from this payload the instant it changes. Nothing downstream reaches
  // back into the socket or the secrets store directly.
  const syncPayload: SyncPayload = useMemo(
    () => ({
      secrets: Object.fromEntries(entries.map((e) => [e.id, e])),
      plugin: activePluginId ? { pluginId: activePluginId, meshUrl, cameraTarget, sessionId } : null,
      voice: { state: voiceState.state, transcript: voiceState.transcript },
    }),
    [entries, activePluginId, meshUrl, cameraTarget, sessionId, voiceState]
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
      {IS_GRADIO_MODE && <WebDemoBanner />}
      {showSlowGenerationWarning && (
        <div className="app__local-inference-banner" role="status">
          🐢 Local Inference Active — generating with {localModel ?? "the local model"}. This can take a
          few minutes; the app hasn&apos;t frozen.
        </div>
      )}
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
        <button
          type="button"
          className="app__model-badge"
          title={`Active tool-calling provider: ${provider ?? "unknown"} — click to open Settings`}
          onClick={() => setEnvViewerOpen(true)}
        >
          <Zap size={13} strokeWidth={2.25} aria-hidden="true" />
          {providerLabel}
        </button>
        <CostBar cost={costState} />
        {!isCadActive && planState.tasks.length > 0 && (
          <button
            type="button"
            className="app__plan-badge"
            title={`Active Plan: ${planState.objective}`}
            onClick={() => setPlanOpen(true)}
          >
            <ClipboardList size={13} strokeWidth={2.25} aria-hidden="true" />
            {planState.tasks.filter((t) => t.status === "completed").length}/{planState.tasks.length}
          </button>
        )}
        <button
          type="button"
          className="app__icon-btn"
          title="Settings"
          onClick={() => setEnvViewerOpen(true)}
        >
          <Key size={16} strokeWidth={2} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="app__icon-btn"
          title="Core Memory"
          onClick={() => setMemoryOpen(true)}
        >
          <Brain size={16} strokeWidth={2} aria-hidden="true" />
        </button>
        <button type="button" className="app__icon-btn" title="Secrets" onClick={() => setSecretsOpen(true)}>
          <Lock size={16} strokeWidth={2} aria-hidden="true" />
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
                topologyGraph={topologyGraph}
                plan={planState}
                sessionId={sessionId}
              />
            </Suspense>
          </div>
        )}
      </div>

      {secretsOpen && <SecretsMenu onClose={() => setSecretsOpen(false)} />}
      {envViewerOpen && (
        <EnvViewerWidget
          onClose={() => setEnvViewerOpen(false)}
          onSessionsCleared={startNewChat}
          autoApprove={autoApprove}
          onToggleAutoApprove={setAutoApprove}
        />
      )}
      {!isCadActive && planOpen && <PlanChecklist plan={planState} onClose={() => setPlanOpen(false)} />}
      {memoryOpen && <MemoryViewer memory={memoryState} onClose={() => setMemoryOpen(false)} />}
      {!isCadActive && <TerminalDrawer log={log} />}
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
