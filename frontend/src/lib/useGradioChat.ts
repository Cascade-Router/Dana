import { useCallback, useEffect, useState } from "react";
import { connectGradioClient, sendGradioChatMessage } from "./gradioChatClient";
import { type ApiKeys, type ChatMessage, type ConnectionState, useChatSocket } from "./useChatSocket";

// Forces a compile error the moment this hook's return shape drifts from
// useChatSocket's — useChat.ts's `cond ? useGradioChat : useChatSocket`
// needs both branches structurally compatible for every consumer's
// destructuring to keep type-checking.
type ChatHookResult = ReturnType<typeof useChatSocket>;

// The Gradio-backend counterpart to useChatSocket — same call signature and
// return shape (see useChat.ts, which picks one or the other ONCE at module
// load based on VITE_HF_SPACE_URL) so App.tsx never branches on which
// protocol is live. What differs is real: app.py's plain gr.ChatInterface
// has no BYOK routing, no plugin/capability routing, no session resume, no
// CAD viewport/mesh export, no voice, no live tool-activity or DAG telemetry,
// and no interactive HITL approval (auto-resolved server-side instead — see
// app.py's _GradioSocket). Everything for those is either ignored on the
// way in or stays permanently empty/no-op on the way out; components that
// already guard on e.g. `meshUrl != null` before rendering the CAD viewport
// behave correctly here for free, they just never see it become non-null.
export function useGradioChat(
  _apiKeys: ApiKeys = {},
  _activePlugins: string[] = [],
  _requestedSessionId: string | null = null,
  initialMessages: ChatMessage[] = []
): ChatHookResult {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [turnActive, setTurnActive] = useState(false);
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
          setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
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
    log: [],
    driverState: null,
    meshUrl: null,
    cameraTarget: null,
    voiceState: { state: "idle" as const, transcript: "" },
    liveActivity: [],
    turnActive,
    sessionId: null,
    sendMessage,
    abortTurn,
    sendSelection: noop,
    respondHitl: noop,
    requestListen: noop,
    cancelListen: noop,
  };
}
