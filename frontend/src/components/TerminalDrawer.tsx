import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { getCapturedLog, installConsoleCapture, subscribeCapturedLog } from "../lib/consoleCapture";
import type { ServerEvent } from "../lib/useChatSocket";
import "./TerminalDrawer.css";

// Idempotent (see consoleCapture's own `installed` guard) — calling it here
// too, rather than trusting import order elsewhere, means this component
// never renders "No events yet." while console capture just hasn't started
// yet regardless of which other module happened to import it first.
installConsoleCapture();

// "ready" is deliberately not a case here — useChatSocket.ts's onmessage
// excludes it from the `log` array it feeds this component (see that
// exclusion's own comment for why), so it never reaches this function in
// practice; the `default` branch below still covers it defensively if
// that ever changes.
function formatLine(event: ServerEvent): string {
  switch (event.type) {
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

  // Reactive read of the shared browser-console buffer (consoleCapture.ts)
  // — populated regardless of WS vs Gradio mode, unlike `log` above (which
  // is empty on the Gradio build; see useGradioChat.ts). useSyncExternalStore
  // re-renders this component on every captured console call without any
  // extra plumbing through App.tsx/useChat.
  const consoleLog = useSyncExternalStore(subscribeCapturedLog, getCapturedLog);
  const combinedLog = useMemo(() => [...log, ...consoleLog], [log, consoleLog]);

  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [combinedLog, open]);

  return (
    <>
      <button
        type="button"
        className="terminal-drawer__fab"
        onClick={() => setOpen((o) => !o)}
        title="Terminal History"
      >
        {open ? "✕" : "🐞"} Terminal History ({combinedLog.length})
      </button>
      {open && (
        <div className="terminal-drawer__panel">
          <div className="terminal-drawer__header">
            Terminal History — live backend/WS log + browser console
          </div>
          <div className="terminal-drawer__body" ref={bodyRef}>
            {combinedLog.length === 0 && <div className="terminal-drawer__empty">No events yet.</div>}
            {combinedLog.map((event, i) => (
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
