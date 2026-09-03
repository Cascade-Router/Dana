import type { PlanState } from "../lib/useChatSocket";
import "./PlanChecklist.css";

type Props = {
  plan: PlanState;
};

// InspectorDock's "Active Plan" tab — same plan_update-driven render as
// PlanChecklist.tsx's floating overlay (see that file's own comment for the
// exact backend state this mirrors), minus the backdrop/close chrome the
// dock's tab bar now provides. Reuses PlanChecklist's CSS classes directly.
export function PlanTab({ plan }: Props) {
  const { objective, tasks } = plan;

  if (tasks.length === 0) {
    return (
      <div className="plan-checklist__empty">
        No active plan yet — one appears here the moment a multi-step request creates one.
      </div>
    );
  }

  return (
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
  );
}
