import { useState } from "react";
import type { ServerEvent } from "../lib/useChatSocket";
import "./TerminalDrawer.css";

function formatLine(event: ServerEvent): string {
  switch (event.type) {
    case "ready":
      return `[ready] driver_state=${JSON.stringify(event.driver_state)}`;
    case "tool_call":
      return `[tool_call] ${event.tool_id}(${JSON.stringify(event.arguments)})`;
    case "tool_result":
      return `[tool_result] ${event.tool_id} ok=${event.ok} duration_ms=${event.duration_ms} ${event.message}`;
    case "assistant_message":
      return `[assistant] ${event.content}`;
    case "user_message":
      return `[user] ${event.content}`;
    default:
      return JSON.stringify(event);
  }
}

type Props = {
  log: ServerEvent[];
};

export function TerminalDrawer({ log }: Props) {
  const [open, setOpen] = useState(true);

  return (
    <div className={`terminal-drawer ${open ? "terminal-drawer--open" : ""}`}>
      <button className="terminal-drawer__toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} Tool Dispatch Log ({log.length})
      </button>
      {open && (
        <div className="terminal-drawer__body">
          {log.length === 0 && <div className="terminal-drawer__empty">No dispatch events yet.</div>}
          {log.map((event, i) => (
            <div key={i} className={`terminal-drawer__line terminal-drawer__line--${event.type}`}>
              {formatLine(event)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
