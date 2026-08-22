import { useMemo, useState } from "react";
import type { ServerEvent } from "../lib/useChatSocket";
import type { PluginComponentProps } from "./types";
import "./CoderPlugin.css";

const CODER_TOOL_IDS = new Set(["analyze_codebase", "execute_code_task"]);

type CoderRun = {
  key: string;
  toolId: "analyze_codebase" | "execute_code_task";
  arguments: Record<string, unknown>;
  result: {
    ok: boolean;
    payload: Record<string, unknown>;
    message: string;
    durationMs: number;
  } | null;
};

// Pairs each "software_engineering" domain tool_call with the tool_result
// that follows it — dana/api/server.py dispatches one tool call at a time
// per turn (see ChatPanel's AgentActivityFeed matching comment), so a
// tool_call is always immediately followed by its own tool_result, never
// interleaved with another call to the same or a different tool.
function buildRuns(log: ServerEvent[]): CoderRun[] {
  const runs: CoderRun[] = [];
  let pending: CoderRun | null = null;

  for (const event of log) {
    if (event.type === "tool_call" && CODER_TOOL_IDS.has(event.tool_id)) {
      pending = {
        key: `${runs.length}:${event.tool_id}`,
        toolId: event.tool_id as CoderRun["toolId"],
        arguments: event.arguments,
        result: null,
      };
      runs.push(pending);
    } else if (event.type === "tool_result" && pending && event.tool_id === pending.toolId && !pending.result) {
      pending.result = {
        ok: event.ok,
        payload: event.payload,
        message: event.message,
        durationMs: event.duration_ms,
      };
    }
  }
  return runs;
}

function FileList({ files }: { files: unknown }) {
  const list = Array.isArray(files) ? files.filter((f): f is string => typeof f === "string") : [];
  if (list.length === 0) return null;
  return (
    <ul className="coder-plugin__file-list">
      {list.map((f) => (
        <li key={f}>{f}</li>
      ))}
    </ul>
  );
}

function OutputBlock({ label, text }: { label: string; text: unknown }) {
  if (typeof text !== "string" || text.trim().length === 0) return null;
  return (
    <div className="coder-plugin__output-block">
      <div className="coder-plugin__output-label">{label}</div>
      <pre className="coder-plugin__output-pre">{text}</pre>
    </div>
  );
}

function RunCard({ run }: { run: CoderRun }) {
  const [open, setOpen] = useState(false);
  const status = run.result === null ? "running" : run.result.ok ? "success" : "error";
  const isExecute = run.toolId === "execute_code_task";

  return (
    <div className={`coder-plugin__card coder-plugin__card--${status}`}>
      <button type="button" className="coder-plugin__card-header" onClick={() => setOpen((o) => !o)}>
        <span className="coder-plugin__card-caret">{open ? "▾" : "▸"}</span>
        <span className="coder-plugin__card-icon">{isExecute ? "🛠️" : "🔎"}</span>
        <span className="coder-plugin__card-title">{isExecute ? "execute_code_task" : "analyze_codebase"}</span>
        <span className={`coder-plugin__card-status coder-plugin__card-status--${status}`}>
          {status === "running" ? "running…" : status === "success" ? "committed" : "failed"}
        </span>
      </button>

      {open && (
        <div className="coder-plugin__card-body">
          {isExecute ? (
            <>
              <div className="coder-plugin__field-label">Task</div>
              <div className="coder-plugin__task-text">{String(run.arguments.task_description ?? "")}</div>
              <div className="coder-plugin__field-label">Target files</div>
              <FileList files={run.arguments.files} />
            </>
          ) : (
            <>
              <div className="coder-plugin__field-label">Query</div>
              <div className="coder-plugin__task-text">{String(run.arguments.query ?? "")}</div>
              <FileList files={run.arguments.files} />
            </>
          )}

          {run.result && (
            <>
              {!run.result.ok && (
                <div className="coder-plugin__error-banner">{String(run.result.payload.error ?? run.result.message)}</div>
              )}
              <OutputBlock label="stdout" text={run.result.payload.stdout} />
              <OutputBlock label="stderr" text={run.result.payload.stderr} />
              <OutputBlock label="matches" text={run.result.payload.matches} />
              <div className="coder-plugin__meta">{run.result.durationMs}ms</div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Software Engineering / Coder plugin tab — read-only transparency view over
// the coder_plugin's two tools (dana/plugins/coder_plugin/manifest.json,
// domain "software_engineering"): analyze_codebase (read-only recon) and
// execute_code_task (Aider-driven mutation + auto-commit, HITL-gated before
// it ever dispatches — see ChatPanel's HitlCard for the approval step
// itself). This plugin only renders what already streamed over the socket
// (PluginComponentProps' `log`), the same way DAGMonitor/TerminalDrawer do —
// no separate REST endpoint, since dana/api/server.py already broadcasts
// every tool_call/tool_result pair to every connected plugin.
export default function CoderPlugin({ log }: PluginComponentProps) {
  const runs = useMemo(() => buildRuns(log), [log]);

  return (
    <div className="coder-plugin">
      <div className="coder-plugin__header">
        <span>Coder — Software Engineering</span>
        <span className="coder-plugin__count">{runs.length} run{runs.length === 1 ? "" : "s"}</span>
      </div>
      <div className="coder-plugin__body">
        {runs.length === 0 && (
          <div className="coder-plugin__placeholder">
            No codebase recon or code tasks yet this session. Ask Dana to look at or change some code.
          </div>
        )}
        {runs.map((run) => (
          <RunCard key={run.key} run={run} />
        ))}
      </div>
    </div>
  );
}
