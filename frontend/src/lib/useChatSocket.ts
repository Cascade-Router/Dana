import { useCallback, useEffect, useRef, useState } from "react";
import { API_WS_BASE, resolveApiUrl, resolveMeshUrl } from "./apiBase";

// dana.plugins.planning.task_board's global plan wire shape — exactly what
// get_active_plan()/create_plan()/mark_task_completed() all share (see
// task_board.py's own "Shape:" comment), so this never needs its own
// translation layer against the backend.
type PlanTask = {
  id: number;
  description: string;
  status: "pending" | "active" | "completed";
};

type PlanWire = { objective: string; tasks: PlanTask[]; current_task_id: number | null };

// Mirrors the JSON shapes dana/api/server.py's `/ws/chat` sends —
// keep these two in sync by hand (no shared schema generation yet).
export type ServerEvent =
  | {
      type: "ready";
      session_id: string;
      driver_state: Record<string, unknown>;
      plugins: { plugins: string[]; tools: unknown[] };
      active_plan: PlanWire;
    }
  | { type: "plan_update"; plan: PlanWire }
  | { type: "user_message"; content: string }
  | { type: "assistant_message"; content: string }
  | { type: "camera_animate"; position: [number, number, number]; target: [number, number, number] }
  | {
      // Still used ONLY for a "parse-N" node (the LLM's own reasoning step,
      // one per ReAct iteration) — a tool dispatch's node_id/label/node_type
      // moved to "tool_dispatch_start" below (WebSocket Consolidation).
      // DAGMonitor.tsx's buildGraph tells the two apart by node_id prefix.
      type: "dag_node_start";
      node_id: string;
      label: string;
      node_type: "agent" | "tool" | "vision";
      inputs: Record<string, unknown>;
    }
  | {
      // Parse-node counterpart to dag_node_start above — see the same note.
      type: "dag_node_complete";
      node_id: string;
      status: "success" | "error";
      output: Record<string, unknown>;
      duration_ms: number;
    }
  | {
      // WebSocket Consolidation: replaces the old dag_node_start(dispatch) +
      // "tool_call" + "tool_start" trio with one event — see dana/api/server.py's
      // _send_tool_dispatch_start. Read by DAGMonitor (node_id/label/node_type)
      // and ChatPanel's Agent Activity feed (tool_name/args_summary) alike.
      type: "tool_dispatch_start";
      node_id: string;
      label: string;
      node_type: "agent" | "tool" | "vision";
      tool_name: string;
      arguments: Record<string, unknown>;
      args_summary: string;
    }
  | {
      // WebSocket Consolidation: replaces the old "tool_complete" +
      // dag_node_complete(dispatch) + "tool_result" trio — see
      // _send_tool_dispatch_end. `output` is the tool's result payload,
      // serialized exactly once (it used to go out twice, byte-identical,
      // as dag_node_complete.output AND tool_result.payload).
      type: "tool_dispatch_end";
      node_id: string;
      tool_id: string;
      status: "success" | "error";
      output: Record<string, unknown>;
      // Kept (not in this consolidation's original field list) because
      // CoderPlugin.tsx's error banner falls back to it when a tool's own
      // failure payload doesn't use the "error" key — see
      // dana/api/server.py's _send_tool_dispatch_end for why.
      message: string;
      duration_ms: number;
      mesh_url: string | null;
    }
  | {
      type: "hitl_approval_required";
      payload: {
        request_id: string;
        action_name: string;
        description: string;
        parameters: Record<string, unknown>;
      };
    }
  | { type: "voice_state"; state: VoiceState; transcript: string }
  | { type: "assistant_audio"; audio_url: string }
  | { type: "server_log"; stream: "stdout" | "stderr"; line: string }
  | {
      type: "usage_update";
      model: string;
      tokens: { prompt: number; completion: number };
      cost_usd: number | null;
      session_total_usd: number;
      by_model: Record<string, number>;
    };

// One session's running LLM cost — mirrors dana/api/server.py's
// session["cost_tracking"] (accumulated in _broadcast_usage_update, one
// "usage_update" event per next_react_turn iteration). `activeModel` is
// just the MOST RECENT model seen this session, not necessarily the one
// with the largest slice of `byModel` (a session that switches models
// keeps every model's own accumulated cost in `byModel`, so CostBar can
// still render a fair proportional segment for each).
export type CostState = {
  activeModel: string | null;
  sessionTotalUsd: number;
  byModel: Record<string, number>;
};

const INITIAL_COST_STATE: CostState = { activeModel: null, sessionTotalUsd: 0, byModel: {} };

// Camel-cased convenience shape PlanChecklist actually renders — see
// PlanWire above for the raw wire shape this is built from (both "ready"'s
// initial active_plan seed and every later "plan_update").
export type PlanState = {
  objective: string;
  tasks: PlanTask[];
  currentTaskId: number | null;
};

const INITIAL_PLAN_STATE: PlanState = { objective: "", tasks: [], currentTaskId: null };

function toPlanState(wire: PlanWire): PlanState {
  return { objective: wire.objective, tasks: wire.tasks, currentTaskId: wire.current_task_id };
}

export type VoiceState = "idle" | "listening" | "processing" | "speaking";

export type HitlRequest = {
  requestId: string;
  actionName: string;
  description: string;
  parameters: Record<string, unknown>;
  resolution?: "approved" | "cancelled";
};

// One "tool_dispatch_start"/"tool_dispatch_end" pair from dana/api/server.py's
// _send_tool_dispatch_start/_send_tool_dispatch_end — the plugin-agnostic
// Agent Activity feed ChatPanel renders inline, distinct from the DAG
// Monitor's "dag_node_*"/"tool_dispatch_*" node graph (which only the
// CadPlugin ever renders). "running" until the matching tool_dispatch_end
// arrives (see the reducer in useChatSocket below).
type AgentActivityStatus = "running" | "success" | "error";

export type AgentActivity = {
  id: string;
  toolName: string;
  argsSummary: string;
  status: AgentActivityStatus;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  imageUrl?: string;
  attachments?: string[];
  activity?: AgentActivity[];
  hitl?: HitlRequest;
};

const MAX_LOG_LINES = 500;

export type ConnectionState = "connecting" | "open" | "closed";

export type CameraTarget = { position: [number, number, number]; target: [number, number, number] };

export type CanvasSelection = {
  meshId: string;
  centroid: [number, number, number];
  normal: [number, number, number];
};

export type ApiKeys = Record<string, string>;

// BYOK: `apiKeys` is a snapshot of the SecretsMenu's keys, keyed by
// ServiceId ("openai", "anthropic", ...). Sent to dana/api/server.py as
// "update_secrets" the moment the socket opens, and again any time the
// caller passes a new object (App.tsx re-derives this from SecretsContext,
// so an edit in SecretsMenu flows through automatically — no separate
// wiring needed there). The backend stores it on session["api_keys"] and
// never logs it; see dana/api/server.py's ws_chat handler.
//
// Capability routing: `activePlugins` is which plugin tab(s) are active
// (App.tsx derives this from PluginContext's activePluginId). Sent as
// "update_context" the same way apiKeys is sent as "update_secrets" — on
// open, and again whenever the caller passes a new array. The backend
// normalizes ids (e.g. "cad" -> "freecad") and stores the result on
// session["active_plugins"] to decide which tools/system-prompt sections
// dana.core.react_dispatch injects into that session's ReAct loop.
// Local Chat Session Persistence: `requestedSessionId` is which saved chat
// (dana.api.sessions) to resume — passed straight through as /ws/chat's
// `?session_id=` query param; `null`/undefined lets the server generate a
// fresh one (a brand-new chat). Changing it (the caller's ChatSidebar
// picking a different session, or "New Chat" going back to null)
// reconnects the socket from scratch — see the connect effect's dependency
// array below. `initialMessages` is the already-fetched history
// (GET /api/sessions/{id}, done by the caller BEFORE swapping
// requestedSessionId) to seed the chat UI with the instant the new
// connection opens, so there's no empty-then-populated flash.
export function useChatSocket(
  apiKeys: ApiKeys = {},
  activePlugins: string[] = [],
  requestedSessionId: string | null = null,
  initialMessages: ChatMessage[] = []
) {
  const socketRef = useRef<WebSocket | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  // The server-RESOLVED session id, echoed back on every "ready" (see
  // onmessage below) — read-only output for callers (e.g. to highlight the
  // active row in ChatSidebar). Deliberately NOT what drives reconnection:
  // that's `requestedSessionId` alone, so echoing this id back never
  // triggers a second, redundant reconnect of the very socket that just
  // reported it.
  const [sessionId, setSessionId] = useState<string | null>(null);
  // Read inside the connect effect at (re)connect time only — a plain ref
  // (same pattern as apiKeysRef/activePluginsRef below) so the effect
  // itself only needs to depend on `requestedSessionId`, not on this array
  // reference too.
  const initialMessagesRef = useRef<ChatMessage[]>(initialMessages);
  initialMessagesRef.current = initialMessages;
  const [log, setLog] = useState<ServerEvent[]>([]);
  // Per-session Terminal History cache, keyed by requestedSessionId. `log`
  // itself stays a flat array (single active connection's live feed) — this
  // is what lets switching BACK to a session already visited in this tab
  // restore what it had instead of the blank slate a fresh `setLog([])`
  // would otherwise leave (see the sessionChanged block below, the only
  // place this is read from or written to).
  const logCacheRef = useRef<Record<string, ServerEvent[]>>({});
  const [driverState, setDriverState] = useState<Record<string, unknown> | null>(null);
  const [costState, setCostState] = useState<CostState>(INITIAL_COST_STATE);
  const [planState, setPlanState] = useState<PlanState>(INITIAL_PLAN_STATE);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [cameraTarget, setCameraTarget] = useState<CameraTarget | null>(null);
  const [voiceState, setVoiceState] = useState<{ state: VoiceState; transcript: string }>({
    state: "idle",
    transcript: "",
  });
  const pendingImageUrlRef = useRef<string | null>(null);
  const replyAudioRef = useRef<HTMLAudioElement | null>(null);
  // The CURRENT turn's in-flight Agent Activity feed — mutated via this ref
  // (not just the mirrored `liveActivity` state below) so "tool_dispatch_end"
  // and the "assistant_message" handler always see the latest entries even
  // though onmessage is a single closure created once in the effect below
  // (a stale-closure hazard the same pendingImageUrlRef pattern above
  // already avoids for the CAD-viewport-capture case). Drained onto the
  // finished ChatMessage's own `activity` field the moment an
  // "assistant_message" arrives, then reset to [] for the next turn.
  const liveActivityRef = useRef<AgentActivity[]>([]);
  const [liveActivity, setLiveActivity] = useState<AgentActivity[]>([]);
  // Global Abort: true from the moment a turn starts (sendMessage, or a
  // tool_dispatch_start/hitl_approval_required arriving from a non-typed trigger
  // like a voice turn) until its final "assistant_message" lands — exactly
  // the window ChatPanel's "Stop Generating" button should be visible for.
  const [turnActive, setTurnActive] = useState(false);
  const apiKeysRef = useRef<ApiKeys>(apiKeys);
  apiKeysRef.current = apiKeys;
  const activePluginsRef = useRef<string[]>(activePlugins);
  activePluginsRef.current = activePlugins;

  // Tracks the session_id THIS hook instance last actually connected as —
  // compared (not just relied on the effect's own dependency array) so the
  // reset below only fires on a genuine session CHANGE, never on a
  // same-session reconnect (a dropped/suspended socket — e.g. a background
  // browser tab's connection getting closed by the browser/OS and this
  // effect re-running to re-establish it) reusing the identical
  // requestedSessionId. Initialized to requestedSessionId itself (not
  // null/undefined) so the very first connect of a brand-new hook instance
  // is never mistaken for a "change" either — first mount already starts
  // from each state's own empty initial value, nothing to additionally
  // clear there.
  const previousSessionIdRef = useRef<string | null>(requestedSessionId);

  useEffect(() => {
    const sessionChanged = previousSessionIdRef.current !== requestedSessionId;
    const previousSessionId = previousSessionIdRef.current;
    previousSessionIdRef.current = requestedSessionId;

    if (sessionChanged) {
      // Stash the outgoing session's log before clearing `log` below, then
      // seed it back in from cache if the incoming session was already
      // visited this tab — otherwise this genuinely is a first visit and an
      // empty array is correct (nothing to restore).
      if (previousSessionId) {
        logCacheRef.current[previousSessionId] = log;
      }
      // Every session CHANGE (never a same-session reconnect — see
      // sessionChanged above) starts this chat's client-side state clean —
      // seeded from whatever history the caller already fetched for THIS
      // requestedSessionId (empty for a brand-new chat), never carrying
      // over the PREVIOUS session's messages/activity/turn state.
      setMessages(initialMessagesRef.current);
      setLiveActivity([]);
      liveActivityRef.current = [];
      setTurnActive(false);
      // Terminal History isolation: `log` (server_log/tool_dispatch_start/
      // tool_dispatch_end/etc. WS events) is genuinely per-session data — without swapping it
      // out here it kept accumulating across a chat switch, so the Terminal
      // History panel showed a PREVIOUS session's backend activity mixed in
      // with the newly-selected one. Swapped via logCacheRef (above) rather
      // than a bare reset so switching BACK to an already-visited session
      // restores what it had instead of going blank. Deliberately NOT resetting
      // consoleCapture.ts's own browser-console buffer here (TerminalDrawer's
      // OTHER log source, read via useSyncExternalStore, combined into the
      // same panel) — that one is browser/OS-level, not tied to any
      // particular chat session, and a debugging tool clearing recent real
      // console errors just because the user switched chats would be
      // surprising, not helpful.
      setLog(requestedSessionId ? logCacheRef.current[requestedSessionId] ?? [] : []);
      setMeshUrl(null);
      setDriverState(null);
      setCameraTarget(null);
      setCostState(INITIAL_COST_STATE);
      // Reseeded the instant the new connection's own "ready" arrives (see
      // below) — the Task Planner is a single GLOBAL plan, not per-session,
      // so this is only ever the brief gap between reset and that reseed,
      // same as driverState/meshUrl above.
      setPlanState(INITIAL_PLAN_STATE);
    }

    const query = requestedSessionId ? `?session_id=${encodeURIComponent(requestedSessionId)}` : "";
    const socket = new WebSocket(`${API_WS_BASE}/ws/chat${query}`);
    socketRef.current = socket;

    socket.onopen = () => {
      setConnection("open");
      // Whatever SecretsContext/PluginContext have resolved by the time the
      // socket opens — if either is still loading/hasn't been set yet, the
      // effects below re-send once they have.
      socket.send(JSON.stringify({ type: "update_secrets", keys: apiKeysRef.current }));
      socket.send(JSON.stringify({ type: "update_context", active_plugins: activePluginsRef.current }));
    };
    socket.onclose = () => setConnection("closed");

    socket.onmessage = (event) => {
      const data: ServerEvent = JSON.parse(event.data);
      // Terminal History's live feed — "server_log" mirrors every backend
      // print()/traceback (see dana/api/server.py's _BroadcastStream), which
      // can be noisy over a long session, so this is capped to the most
      // recent MAX_LOG_LINES rather than growing unbounded for the life of
      // the connection. "ready" is excluded here — it fires on every
      // connect/reconnect carrying the full driver_state payload (already
      // consumed into its own state below), and was crowding real tool
      // activity out of the capped buffer on a long session with several
      // reconnects.
      if (data.type !== "ready") {
        setLog((prev) => [...prev, data].slice(-MAX_LOG_LINES));
      }

      switch (data.type) {
        case "ready":
          setDriverState(data.driver_state);
          setSessionId(data.session_id);
          setPlanState(toPlanState(data.active_plan));
          break;
        case "assistant_message": {
          const imageUrl = pendingImageUrlRef.current ?? undefined;
          pendingImageUrlRef.current = null;
          // Drain this turn's live activity feed onto the finished message
          // — it becomes part of the permanent history (rendered
          // collapsed — see ChatPanel), and the live feed resets to [] so
          // the next turn starts clean.
          const activity = liveActivityRef.current.length ? liveActivityRef.current : undefined;
          liveActivityRef.current = [];
          setLiveActivity([]);
          setTurnActive(false);
          setMessages((prev) => [...prev, { role: "assistant", content: data.content, imageUrl, activity }]);
          break;
        }
        case "tool_dispatch_start": {
          // WebSocket Consolidation: replaces the old "tool_start" case —
          // same Agent Activity entry, sourced from this one event's own
          // tool_name/args_summary fields instead of a separate message.
          setTurnActive(true);
          const entry: AgentActivity = {
            id: `${data.tool_name}-${liveActivityRef.current.length}-${Math.random().toString(36).slice(2)}`,
            toolName: data.tool_name,
            argsSummary: data.args_summary,
            status: "running",
          };
          liveActivityRef.current = [...liveActivityRef.current, entry];
          setLiveActivity(liveActivityRef.current);
          break;
        }
        case "tool_dispatch_end": {
          // WebSocket Consolidation: replaces the old "tool_complete" +
          // "tool_result" cases — marks the matching Agent Activity entry's
          // status AND extracts meshUrl/image_url from the same one event's
          // `output` (the old tool_result's `payload`, unchanged shape) and
          // `mesh_url` fields, instead of two separate messages.
          //
          // Matches the most recent still-"running" entry for this tool_id
          // — matching by name+status (not e.g. an id round-tripped from
          // the server, which this event doesn't carry) is enough here
          // since a turn dispatches tool calls one at a time.
          const current = liveActivityRef.current;
          let targetIndex = -1;
          for (let i = current.length - 1; i >= 0; i--) {
            if (current[i].toolName === data.tool_id && current[i].status === "running") {
              targetIndex = i;
              break;
            }
          }
          if (targetIndex !== -1) {
            const next = current.slice();
            next[targetIndex] = { ...next[targetIndex], status: data.status };
            liveActivityRef.current = next;
            setLiveActivity(next);
          }
          if (data.mesh_url) setMeshUrl(resolveMeshUrl(data.mesh_url));
          const imageUrl = data.output?.image_url;
          if (typeof imageUrl === "string") pendingImageUrlRef.current = resolveApiUrl(imageUrl);
          break;
        }
        case "camera_animate":
          setCameraTarget({ position: data.position, target: data.target });
          break;
        case "usage_update":
          setCostState({
            activeModel: data.model,
            sessionTotalUsd: data.session_total_usd,
            byModel: data.by_model,
          });
          break;
        case "plan_update":
          setPlanState(toPlanState(data.plan));
          break;
        case "voice_state":
          setVoiceState({ state: data.state, transcript: data.transcript });
          break;
        case "assistant_audio": {
          // Server already broadcast voice_state "speaking" alongside this
          // message (see dana/api/server.py's _speak_reply) — playing it
          // back is this client's job; telling the backend playback ended
          // (so it can go back to "idle" and re-arm VoiceService for the
          // next push-to-talk cycle) is done via the "ended"/"error"
          // handlers below, not by guessing a duration up front.
          replyAudioRef.current?.pause();
          const audio = new Audio(resolveApiUrl(data.audio_url));
          replyAudioRef.current = audio;
          const notifyDone = () => {
            if (socketRef.current?.readyState === WebSocket.OPEN) {
              socketRef.current.send(JSON.stringify({ type: "audio_playback_complete" }));
            }
          };
          audio.addEventListener("ended", notifyDone, { once: true });
          audio.addEventListener("error", notifyDone, { once: true });
          audio.play().catch(notifyDone); // e.g. autoplay blocked — don't strand voice_state on "speaking"
          break;
        }
        case "hitl_approval_required": {
          setTurnActive(true);
          const p = data.payload;
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "",
              hitl: {
                requestId: p.request_id,
                actionName: p.action_name,
                description: p.description,
                parameters: p.parameters,
              },
            },
          ]);
          break;
        }
      }
    };

    return () => socket.close();
    // Reconnects from scratch whenever the CALLER explicitly changes which
    // session to connect as (ChatSidebar's "New Chat"/pick-a-session) —
    // deliberately NOT keyed on `sessionId` (the server-echoed value set
    // above), which would otherwise cause an immediate, redundant second
    // reconnect the instant a brand-new chat's generated id comes back.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedSessionId]);

  useEffect(() => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "update_secrets", keys: apiKeys }));
  }, [apiKeys]);

  useEffect(() => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "update_context", active_plugins: activePlugins }));
  }, [activePlugins]);

  // `attachments` are already-resized data URIs from ChatPanel's attachment
  // picker (see its handleFiles/resizeImageToDataUri) — sent alongside
  // `text` in the same "chat_message" payload dana/api/server.py's ws_chat
  // reads off `data.get("attachments")`, then dana.core.react_dispatch's
  // build_user_message turns into the OpenAI-wire multimodal content array
  // for that turn. An empty/omitted array keeps the plain-string payload
  // shape every existing test/backend path already expects.
  //
  // `includeDesktopContext` is Implicit Screen Awareness's toggle state
  // (ChatPanel's monitor button) — sent as `include_desktop_context` so
  // ws_chat captures the desktop itself (dana.plugins.os.desktop_vision's
  // _capture_primary_monitor_jpeg_b64, reused as-is, never duplicated here)
  // and appends it to this turn's attachments BEFORE the ReAct loop starts,
  // rather than the agent spending a whole turn dispatching (and pausing
  // for HITL approval on) analyze_desktop_screen. Omitted entirely when
  // false/undefined, matching cleanAttachments' same "don't send a falsy
  // field" convention below.
  const sendMessage = useCallback((text: string, attachments?: string[], includeDesktopContext?: boolean) => {
    const trimmed = text.trim();
    const cleanAttachments = attachments?.filter(Boolean) ?? [];
    if ((!trimmed && cleanAttachments.length === 0) || socketRef.current?.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed, ...(cleanAttachments.length ? { attachments: cleanAttachments } : {}) },
    ]);
    setTurnActive(true);
    socketRef.current.send(
      JSON.stringify({
        text: trimmed,
        ...(cleanAttachments.length ? { attachments: cleanAttachments } : {}),
        ...(includeDesktopContext ? { include_desktop_context: true } : {}),
      })
    );
  }, []);

  // Global Abort — the "Stop Generating" button. dana/api/server.py's
  // ws_chat either cancels a pending HITL/visual-capture approval outright
  // (replying immediately) or sets session["abort_requested"] for an
  // actively-iterating turn to pick up at its own next checkpoint — either
  // way a real "assistant_message" ("Generation aborted by user.") follows
  // shortly. `setTurnActive(false)` here is the "optimistically update the
  // UI" half: the Stop button (and the live activity spinner it's next to)
  // disappear immediately on click rather than waiting on that round trip.
  const abortTurn = useCallback(() => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "abort_turn" }));
    setTurnActive(false);
  }, []);

  // AssistiveOrb click/hotkey entry points — see dana/services/voice_service.py's
  // push-to-talk docstring. "listen" is a no-op backend-side unless
  // currently idle; "cancel" only does anything while actively listening.
  const requestListen = useCallback(() => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "voice_control", action: "listen" }));
  }, []);

  const cancelListen = useCallback(() => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "voice_control", action: "cancel" }));
  }, []);

  const sendSelection = useCallback((selection: CanvasSelection) => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(
      JSON.stringify({
        type: "canvas_selection",
        payload: { mesh_id: selection.meshId, centroid: selection.centroid, normal: selection.normal },
      })
    );
  }, []);

  const respondHitl = useCallback(
    (requestId: string, approved: boolean, parameters?: Record<string, unknown>) => {
      if (socketRef.current?.readyState !== WebSocket.OPEN) return;
      socketRef.current.send(
        JSON.stringify({
          type: "hitl_response",
          payload: { request_id: requestId, approved, ...(parameters ? { parameters } : {}) },
        })
      );
      setMessages((prev) =>
        prev.map((m) =>
          m.hitl?.requestId === requestId
            ? { ...m, hitl: { ...m.hitl, resolution: approved ? "approved" : "cancelled" } }
            : m
        )
      );
    },
    []
  );

  return {
    connection,
    messages,
    log,
    driverState,
    costState,
    planState,
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
  };
}
