import { useEffect, useRef, useState } from "react";
import type { ServerEvent } from "../lib/useChatSocket";
import "./TerminalDrawer.css";

function formatLine(event: ServerEvent): string {
  switch (event.type) {
    case "ready":
      return `[ready] driver_state=${JSON.stringify(event.driver_state)}`;
    case "tool_call":
      return `[tool_call] ${event.tool_id}(${JSON.stringify(event.arguments)})`;
    case "tool_start":
      return `[tool_start] ${event.tool_name} ${event.args_summary}`;
    case "tool_complete":
      return `[tool_complete] ${event.tool_name} status=${event.status}`;
    case "tool_result":
      return `[tool_result] ${event.tool_id} ok=${event.ok} duration_ms=${event.duration_ms} ${event.message}`;
    case "assistant_message":
      return `[assistant] ${event.content}`;
    case "user_message":
      return `[user] ${event.content}`;
    case "server_log":
      return `[${event.stream}] ${event.line}`;
    default:
      return JSON.stringify(event);
  }
}

type Props = {
  log: ServerEvent[];
};

// Quick Debug UI — a floating toggle button (bottom-right, always on top of
// whatever tab/plugin is active) that opens an overlay panel streaming every
// WebSocket event this session has seen, INCLUDING raw backend stdout/stderr
// via "server_log" (see dana/api/server.py's _BroadcastStream). Deliberately
// unstyled beyond "readable" — this is a debugging aid, not product UI.
export function TerminalDrawer({ log }: Props) {
  const [open, setOpen] = useState(false);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [log, open]);

  return (
    <>
      <button
        type="button"
        className="terminal-drawer__fab"
        onClick={() => setOpen((o) => !o)}
        title="Debug Terminal"
      >
        {open ? "✕" : "🐞"} Debug Terminal ({log.length})
      </button>
      {open && (
        <div className="terminal-drawer__panel">
          <div className="terminal-drawer__header">
            Debug Terminal — live backend/WS log
          </div>
          <div className="terminal-drawer__body" ref={bodyRef}>
            {log.length === 0 && <div className="terminal-drawer__empty">No events yet.</div>}
            {log.map((event, i) => (
              <div key={i} className={`terminal-drawer__line terminal-drawer__line--${event.type}`}>
                {formatLine(event)}
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
