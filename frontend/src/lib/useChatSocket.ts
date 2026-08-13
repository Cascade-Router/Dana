import { useCallback, useEffect, useRef, useState } from "react";
import { API_WS_BASE, resolveApiUrl, resolveMeshUrl } from "./apiBase";

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
  | { type: "assistant_message"; content: string }
  | { type: "camera_animate"; position: [number, number, number]; target: [number, number, number] }
  | {
      type: "dag_node_start";
      node_id: string;
      label: string;
      node_type: "agent" | "tool" | "vision";
      inputs: Record<string, unknown>;
    }
  | {
      type: "dag_node_complete";
      node_id: string;
      status: "success" | "error";
      output: Record<string, unknown>;
      duration_ms: number;
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
  | { type: "voice_state"; state: VoiceState; transcript: string };

export type VoiceState = "idle" | "listening" | "processing" | "speaking";

export type HitlRequest = {
  requestId: string;
  actionName: string;
  description: string;
  parameters: Record<string, unknown>;
  resolution?: "approved" | "cancelled";
};

export type ChatMessage = { role: "user" | "assistant"; content: string; imageUrl?: string; hitl?: HitlRequest };

export type ConnectionState = "connecting" | "open" | "closed";

export type CameraTarget = { position: [number, number, number]; target: [number, number, number] };

export type CanvasSelection = {
  meshId: string;
  centroid: [number, number, number];
  normal: [number, number, number];
};

export function useChatSocket() {
  const socketRef = useRef<WebSocket | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [log, setLog] = useState<ServerEvent[]>([]);
  const [driverState, setDriverState] = useState<Record<string, unknown> | null>(null);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [cameraTarget, setCameraTarget] = useState<CameraTarget | null>(null);
  const [voiceState, setVoiceState] = useState<{ state: VoiceState; transcript: string }>({
    state: "idle",
    transcript: "",
  });
  const pendingImageUrlRef = useRef<string | null>(null);

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
        case "assistant_message": {
          const imageUrl = pendingImageUrlRef.current ?? undefined;
          pendingImageUrlRef.current = null;
          setMessages((prev) => [...prev, { role: "assistant", content: data.content, imageUrl }]);
          break;
        }
        case "tool_result": {
          if (data.mesh_url) setMeshUrl(resolveMeshUrl(data.mesh_url));
          const imageUrl = data.payload?.image_url;
          if (typeof imageUrl === "string") pendingImageUrlRef.current = resolveApiUrl(imageUrl);
          break;
        }
        case "camera_animate":
          setCameraTarget({ position: data.position, target: data.target });
          break;
        case "voice_state":
          setVoiceState({ state: data.state, transcript: data.transcript });
          break;
        case "hitl_approval_required": {
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
  }, []);

  const sendMessage = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || socketRef.current?.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    socketRef.current.send(JSON.stringify({ text: trimmed }));
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
    meshUrl,
    cameraTarget,
    voiceState,
    sendMessage,
    sendSelection,
    respondHitl,
  };
}
