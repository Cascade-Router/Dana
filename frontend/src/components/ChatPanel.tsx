import { FormEvent, useState } from "react";
import type { ChatMessage, ConnectionState, HitlRequest } from "../lib/useChatSocket";
import "./ChatPanel.css";

const QUICK_PROMPTS = [
  "Build a box 60x40x20",
  "Create a cylinder radius 10 height 30",
  "Resync the workspace",
  "System status",
];

function HitlCard({
  hitl,
  onRespond,
}: {
  hitl: HitlRequest;
  onRespond: (requestId: string, approved: boolean, parameters?: Record<string, unknown>) => void;
}) {
  const [modifying, setModifying] = useState(false);
  const [draftParams, setDraftParams] = useState(() => JSON.stringify(hitl.parameters, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);

  if (hitl.resolution) {
    return (
      <div className={`hitl-card hitl-card--${hitl.resolution}`}>
        <div className="hitl-card__title">{hitl.actionName}</div>
        <div className="hitl-card__resolution">
          {hitl.resolution === "approved" ? "Approved — running." : "Cancelled — no changes made."}
        </div>
      </div>
    );
  }

  const submitModified = () => {
    try {
      const parsed = JSON.parse(draftParams);
      setParseError(null);
      onRespond(hitl.requestId, true, parsed);
    } catch {
      setParseError("Parameters must be valid JSON.");
    }
  };

  return (
    <div className="hitl-card hitl-card--pending">
      <div className="hitl-card__title">Approval required: {hitl.actionName}</div>
      <div className="hitl-card__description">{hitl.description}</div>

      {modifying ? (
        <>
          <textarea
            className="hitl-card__params-editor"
            value={draftParams}
            onChange={(e) => setDraftParams(e.target.value)}
            rows={Math.min(10, draftParams.split("\n").length + 1)}
          />
          {parseError && <div className="hitl-card__error">{parseError}</div>}
          <div className="hitl-card__actions">
            <button type="button" className="hitl-card__proceed" onClick={submitModified}>
              Send Modified
            </button>
            <button type="button" className="hitl-card__cancel" onClick={() => setModifying(false)}>
              Back
            </button>
          </div>
        </>
      ) : (
        <>
          <pre className="hitl-card__params">{JSON.stringify(hitl.parameters, null, 2)}</pre>
          <div className="hitl-card__actions">
            <button type="button" className="hitl-card__proceed" onClick={() => onRespond(hitl.requestId, true)}>
              Proceed
            </button>
            <button type="button" className="hitl-card__modify" onClick={() => setModifying(true)}>
              Modify
            </button>
            <button type="button" className="hitl-card__cancel" onClick={() => onRespond(hitl.requestId, false)}>
              Cancel
            </button>
          </div>
        </>
      )}
    </div>
  );
}

type Props = {
  connection: ConnectionState;
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onHitlRespond: (requestId: string, approved: boolean, parameters?: Record<string, unknown>) => void;
};

export function ChatPanel({ connection, messages, onSend, onHitlRespond }: Props) {
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
        {messages.map((m, i) =>
          m.hitl ? (
            <div key={i} className="chat-panel__bubble chat-panel__bubble--assistant chat-panel__bubble--hitl">
              <HitlCard hitl={m.hitl} onRespond={onHitlRespond} />
            </div>
          ) : (
            <div key={i} className={`chat-panel__bubble chat-panel__bubble--${m.role}`}>
              {m.imageUrl && (
                <img className="chat-panel__thumbnail" src={m.imageUrl} alt="CAD viewport capture" />
              )}
              {m.content}
            </div>
          )
        )}
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
