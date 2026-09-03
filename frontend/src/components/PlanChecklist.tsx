import type { PlanState } from "../lib/useChatSocket";
import "./PlanChecklist.css";

type Props = {
  plan: PlanState;
  onClose: () => void;
};

// Renders dana/api/server.py's "plan_update" broadcasts (and "ready"'s own
// initial active_plan seed) — dana.plugins.planning.task_board's global
// Task Planner state, the exact same plan create_plan/mark_task_completed
// mutate and dana.core.react_dispatch.build_system_prompt's "## Current
// Active Plan" system-prompt block renders for the LLM. This is the
// user-facing half of that same state: visual proof the structural
// create_plan override (dana/api/server.py's _looks_multi_step) actually
// fired, without having to read server stderr for it.
//
// Same collapsible-overlay convention as EnvViewerWidget/SecretsMenu (a
// topbar toggle button + backdrop-click-to-close panel) — see App.tsx's
// planOpen state.
export function PlanChecklist({ plan, onClose }: Props) {
  const { objective, tasks } = plan;

  return (
    <div className="plan-checklist">
      <div className="plan-checklist__backdrop" onClick={onClose} />
      <div className="plan-checklist__panel" role="dialog" aria-label="Active Plan">
        <div className="plan-checklist__header">
          <h2>Active Plan</h2>
          <button type="button" className="plan-checklist__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {tasks.length === 0 ? (
          <div className="plan-checklist__empty">No active plan yet — one appears here the moment a multi-step request creates one.</div>
        ) : (
          <>
            <div className="plan-checklist__objective" title={objective}>
              {objective}
            </div>
            <ul className="plan-checklist__list">
              {tasks.map((task) => (
                <li key={task.id} className={`plan-checklist__item plan-checklist__item--${task.status}`}>
                  <span className="plan-checklist__marker" aria-hidden="true">
                    {task.status === "completed" ? "✓" : task.status === "active" ? "▶" : "○"}
                  </span>
                  <span className="plan-checklist__text">{task.description}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
