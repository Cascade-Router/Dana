import { useCallback, useEffect, useState } from "react";
import { resolveApiUrl } from "../lib/apiBase";
import type { PluginComponentProps } from "./types";
import "./PlannerPlugin.css";

type TaskStatus = "pending" | "active" | "completed";
type Task = { id: number; description: string; status: TaskStatus };
type Plan = { objective: string; tasks: Task[]; current_task_id: number | null };

// Simple interval polling to keep the checklist synced with whatever the
// agent's own create_plan/mark_task_completed tool calls just changed —
// there is no dedicated WebSocket event for plan updates (yet), so this
// plugin manages its own state via a plain REST poll instead, the same
// tradeoff ServicesPlugin already makes for its log viewer.
const POLL_INTERVAL_MS = 3000;

const STATUS_MARKER: Record<TaskStatus, string> = { completed: "✓", active: "▶", pending: "○" };

// Task Planner / Executive Function's user-facing counterpart
// (dana/api/planner.py -> dana.plugins.planning.task_board) — read-only
// visibility into the EXACT SAME checklist the agent renders into its own
// system prompt every turn (dana.core.react_dispatch.build_system_prompt's
// "## Current Active Plan" block), so what the user sees here is always
// exactly what the model itself is currently anchored on. Ignores the
// CAD-shaped PluginComponentProps every plugin currently receives (same
// pattern as WorkspacePlugin/SkillsPlugin/ServicesPlugin) — this plugin
// manages its own state via REST calls instead. Deliberately NO edit/kill
// control here (unlike ServicesPlugin's "Kill" button): the plan is
// mutated ONLY by the agent's own create_plan/mark_task_completed tool
// calls — see dana/api/planner.py's own docstring for why a human
// silently rewriting the agent's executive-function state has no
// equivalent safe use case.
export default function PlannerPlugin(_props: PluginComponentProps) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    fetch(resolveApiUrl("/api/planner"))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setPlan(data.plan ?? null);
        setError(null);
      })
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  useEffect(() => {
    load();
    const id = window.setInterval(load, POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const hasTasks = (plan?.tasks.length ?? 0) > 0;

  return (
    <div className="planner-plugin">
      <div className="planner-plugin__header">
        <span>Active Plan</span>
        <button type="button" className="planner-plugin__refresh" onClick={load} title="Refresh">
          ↻
        </button>
      </div>

      <div className="planner-plugin__body">
        {error && <div className="planner-plugin__error">{error}</div>}
        {!error && !plan && <div className="planner-plugin__placeholder">Loading…</div>}
        {!error && plan && !hasTasks && (
          <div className="planner-plugin__placeholder">
            No active plan. Ask Dana to tackle a multi-step project and she&apos;ll lay one out
            here.
          </div>
        )}

        {!error && plan && hasTasks && (
          <>
            <div className="planner-plugin__objective">{plan.objective}</div>
            <ul className="planner-plugin__checklist">
              {plan.tasks.map((task) => (
                <li key={task.id} className={`planner-plugin__task planner-plugin__task--${task.status}`}>
                  <span className="planner-plugin__task-marker" aria-hidden="true">
                    {STATUS_MARKER[task.status]}
                  </span>
                  <span className="planner-plugin__task-description">{task.description}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
