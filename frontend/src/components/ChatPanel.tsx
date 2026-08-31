import { FormEvent, useRef, useState } from "react";
import type { AgentActivity, ChatMessage, ConnectionState, HitlRequest } from "../lib/useChatSocket";
import "./ChatPanel.css";

const QUICK_PROMPTS = [
  "Build a box 60x40x20",
  "Create a cylinder radius 10 height 30",
  "Resync the workspace",
  "System status",
];

// Accepted upload types — mirrors the <input accept=""> below; checked
// again in handleFiles since a drag-drop or a permissive OS picker can
// still hand us a file the accept filter would normally have excluded.
const ACCEPTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

// Raw phone/camera/screenshot images can comfortably exceed the WebSocket
// frame size dana/api/server.py's ws_chat reads in one receive_json() call —
// every attachment is downscaled to fit within this box (aspect preserved)
// and re-encoded as JPEG before it ever touches the socket.
const MAX_ATTACHMENT_DIMENSION = 1024;

// Soft cap on attachments per turn — generous enough for a multi-view
// technical drawing (front/top/side) without letting one turn balloon into
// dozens of full-size images.
const MAX_ATTACHMENTS = 4;

type Attachment = { id: string; dataUrl: string; name: string };

// Browser-side compression: decode via <img>, draw into a <canvas> capped at
// MAX_ATTACHMENT_DIMENSION on its longest edge, re-encode as JPEG. Runs
// entirely client-side — no server round-trip needed just to shrink a photo
// before it's attached to a chat turn.
function resizeImageToDataUri(file: File, maxDimension = MAX_ATTACHMENT_DIMENSION): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file"));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error(`Could not decode image "${file.name}"`));
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDimension || height > maxDimension) {
          const scale = Math.min(maxDimension / width, maxDimension / height);
          width = Math.max(1, Math.round(width * scale));
          height = Math.max(1, Math.round(height * scale));
        }
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas 2D context unavailable"));
          return;
        }
        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}

// Friendly icon/label per tool_id for the Agent Activity feed (see
// AgentActivityFeed below) — a tool_id with no entry here (mostly
// create_freecad_*/other CAD tools, which already have full visibility via
// the CadPlugin's DAG Monitor) falls back to a generic gear + its raw
// tool_name, never breaking the feed for an unmapped id.
const TOOL_ACTIVITY_LABELS: Record<string, { icon: string; label: string }> = {
  search_web: { icon: "🔎", label: "Searching the web" },
  read_webpage: { icon: "🌐", label: "Reading a webpage" },
  run_python_script: { icon: "🐍", label: "Executing Python script" },
  write_file: { icon: "📝", label: "Writing a file" },
  read_file: { icon: "📄", label: "Reading a file" },
  list_directory: { icon: "📁", label: "Listing files" },
  update_core_memory: { icon: "🧠", label: "Updating memory" },
  analyze_workspace_image: { icon: "🖼️", label: "Analyzing an image" },
  take_canvas_screenshot: { icon: "📸", label: "Capturing the viewport" },
  load_capability: { icon: "🔓", label: "Loading a capability" },
  system_state: { icon: "⚙️", label: "Checking system status" },
  check_plugin_registry: { icon: "🧩", label: "Checking plugin registry" },
  query_engineering_standard: { icon: "📐", label: "Looking up a standard" },
  manipulate_camera: { icon: "🎥", label: "Moving the camera" },
  analyze_codebase: { icon: "🔎", label: "Reading the codebase" },
  execute_code_task: { icon: "🛠️", label: "Running a code task" },
};
const DEFAULT_ACTIVITY_ICON = "⚙️";

function activityLine(activity: AgentActivity): string {
  const meta = TOOL_ACTIVITY_LABELS[activity.toolName];
  const label = meta?.label ?? activity.toolName;
  return activity.argsSummary ? `${label}: ${activity.argsSummary}` : label;
}

function ActivityRow({ activity }: { activity: AgentActivity }) {
  const icon =
    activity.status === "success" ? "✅" : activity.status === "error" ? "❌" : TOOL_ACTIVITY_LABELS[activity.toolName]?.icon ?? DEFAULT_ACTIVITY_ICON;
  return (
    <div className={`chat-panel__activity-row chat-panel__activity-row--${activity.status}`}>
      <span className="chat-panel__activity-icon">{icon}</span>
      <span className="chat-panel__activity-label">{activityLine(activity)}</span>
      {activity.status === "running" && <span className="chat-panel__activity-spinner" aria-hidden="true" />}
    </div>
  );
}

// Plugin-agnostic tool-execution transparency, rendered inline in the chat
// stream itself — unlike the CadPlugin's DAG Monitor, this is visible with
// no plugin tab open at all (see dana/api/server.py's _execute_and_continue
// for the "tool_start"/"tool_complete" events this renders). `live` drops
// the collapsing <details> wrapper so an in-flight turn's activity is
// always visible while it's actually happening, not hidden behind a click.
function AgentActivityFeed({ activity, live = false }: { activity: AgentActivity[]; live?: boolean }) {
  if (activity.length === 0) return null;
  const rows = activity.map((a) => <ActivityRow key={a.id} activity={a} />);
  if (live) {
    return <div className="chat-panel__activity">{rows}</div>;
  }
  return (
    <details className="chat-panel__activity-details">
      <summary>
        {activity.length} tool call{activity.length === 1 ? "" : "s"}
      </summary>
      <div className="chat-panel__activity">{rows}</div>
    </details>
  );
}

// execute_code_task's own params shape (dana/plugins/coder_plugin/manifest.json)
// is always {task_description, files} — surfaced explicitly here (which
// files Aider is about to touch, and why) rather than as opaque JSON, since
// this is the one HITL prompt in the app where "what exactly gets mutated"
// isn't obvious from the action name alone.
function CodeTaskApprovalDetails({ parameters, diff }: { parameters: Record<string, unknown>; diff?: string }) {
  const [showDiff, setShowDiff] = useState(true); // Default to showing diff for safety
  const files = Array.isArray(parameters.files) ? parameters.files.filter((f): f is string => typeof f === "string") : [];
  const taskDescription = typeof parameters.task_description === "string" ? parameters.task_description : "";
  const hasDiff = diff && diff.trim().length > 0;
  
  return (
    <div className="hitl-card__code-task">
      <div className="hitl-card__code-task-badge">⚠️ Aider will edit files and create a git commit</div>
      {taskDescription && <div className="hitl-card__code-task-desc">{taskDescription}</div>}
      {files.length > 0 && (
        <>
          <div className="hitl-card__code-task-label">Files to be modified</div>
          <ul className="hitl-card__code-task-files">
            {files.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </>
      )}
      {hasDiff && (
        <div className="hitl-card__diff-section">
          <div className="hitl-card__diff-header">
            <span className="hitl-card__diff-label">📋 Current File State (Before Changes)</span>
            <button
              type="button"
              className="hitl-card__diff-toggle"
              onClick={() => setShowDiff(!showDiff)}
            >
              {showDiff ? "Hide Preview" : "Show Preview"}
            </button>
          </div>
          {showDiff && (
            <pre className="hitl-card__diff">
              <code>{diff}</code>
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

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
          {hitl.actionName === "execute_code_task" ? (
            <CodeTaskApprovalDetails parameters={hitl.parameters} diff={hitl.diff} />
          ) : (
            <pre className="hitl-card__params">{JSON.stringify(hitl.parameters, null, 2)}</pre>
          )}
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
  liveActivity: AgentActivity[];
  turnActive: boolean;
  onSend: (text: string, attachments?: string[], includeDesktopContext?: boolean) => void;
  onAbort: () => void;
  onHitlRespond: (requestId: string, approved: boolean, parameters?: Record<string, unknown>) => void;
};

export function ChatPanel({ connection, messages, liveActivity, turnActive, onSend, onAbort, onHitlRespond }: Props) {
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [attachError, setAttachError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Implicit Screen Awareness — persists across sends (a standing opt-in,
  // not a one-shot flag) until the user clicks it off again; see
  // useChatSocket.sendMessage's `includeDesktopContext` param for how this
  // reaches dana/api/server.py's ws_chat.
  const [desktopContextEnabled, setDesktopContextEnabled] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!draft.trim() && attachments.length === 0) return;
    onSend(draft, attachments.map((a) => a.dataUrl), desktopContextEnabled);
    setDraft("");
    setAttachments([]);
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setAttachError(null);
    const room = MAX_ATTACHMENTS - attachments.length;
    if (room <= 0) {
      setAttachError(`Up to ${MAX_ATTACHMENTS} images per message.`);
      return;
    }
    const picked = Array.from(files).slice(0, room);
    const accepted = picked.filter((f) => ACCEPTED_IMAGE_TYPES.has(f.type));
    if (accepted.length < picked.length) {
      setAttachError("Only PNG, JPEG, and WebP images are supported.");
    }
    const resized = await Promise.all(
      accepted.map(async (file) => {
        try {
          const dataUrl = await resizeImageToDataUri(file);
          return { id: `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2)}`, dataUrl, name: file.name };
        } catch {
          setAttachError(`Could not process "${file.name}".`);
          return null;
        }
      })
    );
    setAttachments((prev) => [...prev, ...resized.filter((a): a is Attachment => a !== null)]);
  };

  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
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
              {m.activity && m.activity.length > 0 && <AgentActivityFeed activity={m.activity} />}
              {m.imageUrl && (
                <img className="chat-panel__thumbnail" src={m.imageUrl} alt="CAD viewport capture" />
              )}
              {m.attachments && m.attachments.length > 0 && (
                <div className="chat-panel__sent-attachments">
                  {m.attachments.map((src, j) => (
                    <img key={j} className="chat-panel__thumbnail" src={src} alt="attached image" />
                  ))}
                </div>
              )}
              {m.content}
            </div>
          )
        )}
        {liveActivity.length > 0 && (
          <div className="chat-panel__bubble chat-panel__bubble--assistant chat-panel__bubble--activity-live">
            <AgentActivityFeed activity={liveActivity} live />
          </div>
        )}
      </div>

      <div className="chat-panel__quick-prompts">
        {QUICK_PROMPTS.map((p) => (
          <button key={p} type="button" onClick={() => onSend(p, undefined, desktopContextEnabled)}>
            {p}
          </button>
        ))}
      </div>

      {attachments.length > 0 && (
        <div className="chat-panel__attachment-preview">
          {attachments.map((a) => (
            <div key={a.id} className="chat-panel__attachment-thumb">
              <img src={a.dataUrl} alt={a.name} />
              <button
                type="button"
                className="chat-panel__attachment-remove"
                title="Remove"
                onClick={() => removeAttachment(a.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      {attachError && <div className="chat-panel__attachment-error">{attachError}</div>}

      {desktopContextEnabled && (
        <div className="chat-panel__desktop-context-note">
          🖥️ Screen Awareness is on — your screen is attached to every message you send.
        </div>
      )}

      {turnActive && (
        <div className="chat-panel__abort-row">
          <button type="button" className="chat-panel__abort-btn" onClick={onAbort}>
            ■ Stop Generating
          </button>
        </div>
      )}

      <form className="chat-panel__composer" onSubmit={submit}>
        <input
          ref={fileInputRef}
          type="file"
          hidden
          accept="image/png,image/jpeg,image/webp"
          multiple
          onChange={(e) => {
            void handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          className="chat-panel__attach-btn"
          title="Attach an image"
          disabled={connection !== "open" || attachments.length >= MAX_ATTACHMENTS}
          onClick={() => fileInputRef.current?.click()}
        >
          📎
        </button>
        <button
          type="button"
          className={`chat-panel__desktop-toggle ${desktopContextEnabled ? "chat-panel__desktop-toggle--active" : ""}`}
          title={
            desktopContextEnabled
              ? "Screen Awareness is on — click to stop attaching your screen"
              : "Enable Screen Awareness — attach your screen to every message"
          }
          aria-pressed={desktopContextEnabled}
          disabled={connection !== "open"}
          onClick={() => setDesktopContextEnabled((prev) => !prev)}
        >
          🖥️
        </button>
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
