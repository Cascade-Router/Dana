import { FormEvent, useState } from "react";
import type { ChatMessage, ConnectionState } from "../lib/useChatSocket";
import "./ChatPanel.css";

const QUICK_PROMPTS = [
  "Build a box 60x40x20",
  "Create a cylinder radius 10 height 30",
  "Resync the workspace",
  "System status",
];

type Props = {
  connection: ConnectionState;
  messages: ChatMessage[];
  onSend: (text: string) => void;
};

export function ChatPanel({ connection, messages, onSend }: Props) {
  const [draft, setDraft] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    onSend(draft);
    setDraft("");
  };

  return (
    <div className="chat-panel">
      <div className="chat-panel__header">
        <h1>Dana</h1>
        <span className={`chat-panel__status chat-panel__status--${connection}`}>{connection}</span>
      </div>

      <div className="chat-panel__messages">
        {messages.length === 0 && (
          <p className="chat-panel__empty">
            Ask Dana to do something (e.g. &quot;build a box 60x40x20&quot;).
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-panel__bubble chat-panel__bubble--${m.role}`}>
            {m.content}
          </div>
        ))}
      </div>

      <div className="chat-panel__quick-prompts">
        {QUICK_PROMPTS.map((p) => (
          <button key={p} type="button" onClick={() => onSend(p)}>
            {p}
          </button>
        ))}
      </div>

      <form className="chat-panel__composer" onSubmit={submit}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask Dana to do something..."
        />
        <button type="submit" disabled={connection !== "open"}>
          Send
        </button>
      </form>
    </div>
  );
}
