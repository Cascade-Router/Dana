import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { getCapturedLog, installConsoleCapture, subscribeCapturedLog } from "../lib/consoleCapture";
import type { ServerEvent } from "../lib/useChatSocket";
import "./TerminalDrawer.css";

// Idempotent (see consoleCapture's own `installed` guard) — called here too
// rather than trusting import order, same as TerminalDrawer.tsx itself.
installConsoleCapture();

// Kept identical to TerminalDrawer.tsx's own formatLine — see that file's
// comment for why "ready" is dead code in the default branch.
function formatLine(event: ServerEvent): string {
  switch (event.type) {
    case "tool_dispatch_start":
      return `[tool_dispatch_start] ${event.tool_name}(${JSON.stringify(event.arguments)}) ${event.args_summary}`;
    case "tool_dispatch_end":
      return `[tool_dispatch_end] ${event.tool_id} status=${event.status} duration_ms=${event.duration_ms} ${event.message}`;
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

// InspectorDock's "Terminal" tab — same combined WS-log + browser-console
// feed as TerminalDrawer.tsx's floating FAB, minus the FAB/panel chrome the
// dock's tab bar now provides. Reuses TerminalDrawer's CSS classes directly.
export function TerminalTab({ log }: Props) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const consoleLog = useSyncExternalStore(subscribeCapturedLog, getCapturedLog);
  const combinedLog = useMemo(() => [...log, ...consoleLog], [log, consoleLog]);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [combinedLog]);

  return (
    <div className="terminal-drawer__body" ref={bodyRef}>
      {combinedLog.length === 0 && <div className="terminal-drawer__empty">No events yet.</div>}
      {combinedLog.map((event, i) => (
        <div key={i} className={`terminal-drawer__line terminal-drawer__line--${event.type}`}>
          {formatLine(event)}
        </div>
      ))}
    </div>
  );
}
