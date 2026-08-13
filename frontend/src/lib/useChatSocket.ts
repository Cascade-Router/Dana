import { useCallback, useEffect, useRef, useState } from "react";
import { API_WS_BASE, resolveMeshUrl } from "./apiBase";

// Mirrors the JSON shapes dana/api/server.py's `/ws/chat` sends —
// keep these two in sync by hand (no shared schema generation yet).
export type ServerEvent =
  | { type: "ready"; driver_state: Record<string, unknown>; plugins: { plugins: string[]; tools: unknown[] } }
  | { type: "user_message"; content: string }
  | { type: "tool_call"; tool_id: string; arguments: Record<string, unknown> }
  | {
      type: "tool_result";
      tool_id: string;
      ok: boolean;
      payload: Record<string, unknown>;
      message: string;
      duration_ms: number;
      mesh_url: string | null;
    }
  | { type: "assistant_message"; content: string };

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type ConnectionState = "connecting" | "open" | "closed";

export function useChatSocket() {
  const socketRef = useRef<WebSocket | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [log, setLog] = useState<ServerEvent[]>([]);
  const [driverState, setDriverState] = useState<Record<string, unknown> | null>(null);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);

  useEffect(() => {
    const socket = new WebSocket(`${API_WS_BASE}/ws/chat`);
    socketRef.current = socket;

    socket.onopen = () => setConnection("open");
    socket.onclose = () => setConnection("closed");

    socket.onmessage = (event) => {
      const data: ServerEvent = JSON.parse(event.data);
      setLog((prev) => [...prev, data]);

      switch (data.type) {
        case "ready":
          setDriverState(data.driver_state);
          break;
        case "assistant_message":
          setMessages((prev) => [...prev, { role: "assistant", content: data.content }]);
          break;
        case "tool_result":
          if (data.mesh_url) setMeshUrl(resolveMeshUrl(data.mesh_url));
          break;
      }
    };

    return () => socket.close();
  }, []);

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || socketRef.current?.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    socketRef.current.send(JSON.stringify({ text: trimmed }));
  }, []);

  return { connection, messages, log, driverState, meshUrl, sendMessage };
}
