import { useCallback, useEffect, useState } from "react";
import { connectGradioClient, sendGradioChatMessage } from "./gradioChatClient";
import {
  type ApiKeys,
  type ChatMessage,
  type ConnectionState,
  type MemoryState,
  type ServerEvent,
  type TopologyGraph,
  useChatSocket,
} from "./useChatSocket";

// Forces a compile error the moment this hook's return shape drifts from
// useChatSocket's — useChat.ts's `cond ? useGradioChat : useChatSocket`
// needs both branches structurally compatible for every consumer's
// destructuring to keep type-checking.
type ChatHookResult = ReturnType<typeof useChatSocket>;

const EMPTY_MEMORY_STATE: MemoryState = {};
// No Topological Lineage Graph telemetry from app.py's bare Gradio "chat"
// endpoint (it only ever emitted the old erratic dag_node_start/
// tool_dispatch_* execution-path events into `log`, which DAGMonitor no
// longer reads) — permanently empty, same convention as EMPTY_MEMORY_STATE.
const EMPTY_TOPOLOGY_GRAPH: TopologyGraph = { nodes: {}, edges: [] };

// The Gradio-backend counterpart to useChatSocket — same call signature and
// return shape (see useChat.ts, which picks one or the other ONCE at module
// load based on VITE_HF_SPACE_URL) so App.tsx never branches on which
// protocol is live. What differs is real: app.py's plain gr.Blocks "chat"
// endpoint has no BYOK routing, no plugin/capability routing, no session
// resume, no voice, no live tool-activity or DAG telemetry, and no
// interactive HITL approval (auto-resolved server-side instead — see
// app.py's _GradioSocket). Everything for those is either ignored on the
// way in or stays permanently empty/no-op on the way out.
//
// meshUrl IS real here (unlike the rest of the above): app.py's "chat"
// endpoint returns `(text, mesh_file)` per turn — see gradioChatClient.ts's
// GradioChatReply — so this updates the exact same `meshUrl` state
// useChatSocket exposes, and CadPlugin/Viewer3D pick it up completely
// unchanged, the same way they already do for the WS path's mesh_url.
//
// `log` IS real too: app.py's _GradioSocket captures dag_node_start/
// dag_node_complete into graph_out (data[5]) — gradioChatClient.ts's
// GradioChatReply.dagEvents. DAGMonitor.tsx no longer reads `log` at all
// (it renders the deterministic Topological Lineage Graph instead — see
// EMPTY_TOPOLOGY_GRAPH above), but `log` itself is still kept/returned here
// for TerminalDrawer's sake, which still consumes the raw event stream.
export function useGradioChat(
  _apiKeys: ApiKeys = {},
  _activePlugins: string[] = [],
  _requestedSessionId: string | null = null,
  initialMessages: ChatMessage[] = []
): ChatHookResult {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [turnActive, setTurnActive] = useState(false);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [log, setLog] = useState<ServerEvent[]>([]);
  const spaceUrl = import.meta.env.VITE_HF_SPACE_URL as string;

  useEffect(() => {
    let cancelled = false;
    connectGradioClient(spaceUrl)
      .then(() => {
        if (!cancelled) setConnection("open");
      })
      .catch(() => {
        if (!cancelled) setConnection("closed");
      });
    return () => {
      cancelled = true;
    };
  }, [spaceUrl]);

  const sendMessage = useCallback(
    (text: string) => {
      // No attachment/desktop-context support against the bare Gradio
      // backend — silently dropped rather than erroring, same as any other
      // WS-only feature here.
      const trimmed = text.trim();
      if (!trimmed) return;
      setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
      setTurnActive(true);
      sendGradioChatMessage(spaceUrl, trimmed)
        .then((reply) => {
          setMessages((prev) => [...prev, { role: "assistant", content: reply.text }]);
          // Only overwrite on an actual new mesh — a turn that didn't touch
          // CAD (meshUrl: null) shouldn't blank out whatever's already in
          // the viewport, same expectation the WS path's tool_result
          // handler has (it only ever calls setMeshUrl when mesh_url is
          // truthy, never to explicitly clear it).
          if (reply.meshUrl) setMeshUrl(reply.meshUrl);
          // Append, never replace — DAGMonitor's buildGraph() expects the
          // FULL session history (it re-derives turn numbering from
          // "parse-N" node ids across the whole log), the same way the WS
          // path's `log` only ever grows across a session.
          console.log("[useGradioChat] reply.dagEvents received:", reply.dagEvents.length, reply.dagEvents);
          if (reply.dagEvents.length > 0) {
            setLog((prev) => {
              const next = [...prev, ...reply.dagEvents];
              console.log("[useGradioChat] log state updated:", prev.length, "->", next.length);
              return next;
            });
          }
        })
        .catch(() => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: "Couldn't reach the Hugging Face backend — please try again." },
          ]);
        })
        .finally(() => setTurnActive(false));
    },
    [spaceUrl]
  );

  // No in-flight turn to interrupt server-side (predict() is a single
  // request/response round trip, not a resumable multi-step loop from the
  // client's point of view) — this just clears the local "generating" flag.
  const abortTurn = useCallback(() => setTurnActive(false), []);
  const noop = useCallback(() => {}, []);

  return {
    connection,
    messages,
    log,
    driverState: null,
    // No usage/cost telemetry from app.py's bare Gradio "chat" endpoint (no
    // token/model accounting there at all) — stays permanently empty, same
    // convention as driverState/cameraTarget above; CostBar's own
    // `!activeModel && segments.length === 0` check renders nothing for it.
    costState: { activeModel: null, sessionTotalUsd: 0, byModel: {} },
    // No Task Planner telemetry from app.py's bare Gradio "chat" endpoint
    // either — same permanently-empty convention as costState above.
    planState: { objective: "", tasks: [], currentTaskId: null },
    // No Core Memory telemetry from app.py's bare Gradio "chat" endpoint
    // either — same permanently-empty convention as costState/planState above.
    memoryState: EMPTY_MEMORY_STATE,
    topologyGraph: EMPTY_TOPOLOGY_GRAPH,
    meshUrl,
    cameraTarget: null,
    voiceState: { state: "idle" as const, transcript: "" },
    liveActivity: [],
    turnActive,
    sessionId: null,
    // Always true here: app.py's _GradioSocket already auto-approves every
    // hitl_approval_required unconditionally server-side (see its own
    // docstring) — the toggle has nothing to turn on/off on this path, so
    // it's just reported as permanently-on rather than a real control.
    autoApprove: true,
    sendMessage,
    abortTurn,
    sendSelection: noop,
    respondHitl: noop,
    requestListen: noop,
    cancelListen: noop,
    setAutoApprove: noop,
  };
}
