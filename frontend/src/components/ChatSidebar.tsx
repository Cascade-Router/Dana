import { useCallback, useEffect, useState } from "react";
import type { MouseEvent } from "react";
import { apiFetch, IS_GRADIO_MODE } from "../lib/apiBase";
import "./ChatSidebar.css";

type SessionMeta = { id: string; title: string; updated_at: string };

type Props = {
  /** The server-resolved id of the currently-open chat (useChatSocket's own `sessionId`) — used only to highlight the active row. */
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
};

// Local Chat Session Persistence's browse/resume/delete UI — a thin REST
// client over dana/api/sessions.py's /api/sessions endpoints. Owns none of
// the actual chat state itself (App.tsx's useChatSocket call does that);
// this component only ever reports WHICH session the user wants active,
// via onNewChat/onSelectSession.
export function ChatSidebar({ activeSessionId, onNewChat, onSelectSession }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [sessions, setSessions] = useState<SessionMeta[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    if (IS_GRADIO_MODE) return; // no /api/sessions on the pure-Gradio HF backend — see the static fallback below
    setError(null);
    apiFetch("/api/sessions")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setSessions(data.sessions ?? []))
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  // Refetches whenever the active session changes — the two moments a
  // session's metadata (a brand-new row, or its updated_at bumping to the
  // top) is actually likely to have changed. No polling: a manual refresh
  // button covers the "another window just saved a turn" gap, matching the
  // same lean convention WorkspacePlugin already uses.
  useEffect(() => {
    load();
  }, [load, activeSessionId]);

  const handleDelete = useCallback(
    (id: string, event: MouseEvent) => {
      event.stopPropagation(); // never also trigger the row's own onSelectSession
      apiFetch(`/api/sessions/${id}`, { method: "DELETE" })
        .then(() => {
          if (id === activeSessionId) onNewChat();
          load();
        })
        .catch((err) => setError(String(err instanceof Error ? err.message : err)));
    },
    [activeSessionId, onNewChat, load]
  );

  if (!expanded) {
    return (
      <div className="chat-sidebar chat-sidebar--collapsed">
        <button
          type="button"
          className="chat-sidebar__icon-btn"
          title="Show chat history"
          onClick={() => setExpanded(true)}
        >
          ▸
        </button>
      </div>
    );
  }

  return (
    <div className="chat-sidebar">
      <div className="chat-sidebar__header">
        <span>Chats</span>
        <div className="chat-sidebar__header-actions">
          {!IS_GRADIO_MODE && (
            <button type="button" className="chat-sidebar__icon-btn" onClick={load} title="Refresh">
              ↻
            </button>
          )}
          <button
            type="button"
            className="chat-sidebar__icon-btn"
            onClick={() => setExpanded(false)}
            title="Collapse"
          >
            ◂
          </button>
        </div>
      </div>

      <button type="button" className="chat-sidebar__new-chat" onClick={onNewChat}>
        + New Chat
      </button>

      <div className="chat-sidebar__list">
        {IS_GRADIO_MODE ? (
          <div
            className="chat-sidebar__item chat-sidebar__item--static"
            title="History isn't available in cloud mode"
          >
            <span className="chat-sidebar__item-title">Cloud Session (Active)</span>
          </div>
        ) : (
          <>
            {error && <div className="chat-sidebar__error">{error}</div>}
            {!sessions && !error && <div className="chat-sidebar__placeholder">Loading…</div>}
            {sessions && sessions.length === 0 && (
              <div className="chat-sidebar__placeholder">No saved chats yet.</div>
            )}
            {sessions?.map((s) => (
              <div
                key={s.id}
                className={`chat-sidebar__item ${s.id === activeSessionId ? "chat-sidebar__item--active" : ""}`}
                onClick={() => onSelectSession(s.id)}
                title={s.title}
              >
                <span className="chat-sidebar__item-title">{s.title}</span>
                <button
                  type="button"
                  className="chat-sidebar__item-delete"
                  title="Delete this chat"
                  onClick={(e) => handleDelete(s.id, e)}
                >
                  ×
                </button>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
