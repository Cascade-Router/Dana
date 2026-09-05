import { useState } from "react";
import "./WebDemoBanner.css";

const DISMISS_KEY = "dana:webDemoBanner:dismissed";

function readDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    return false;
  }
}

// Shown only against the sandboxed Hugging Face Space backend (IS_GRADIO_MODE
// — see App.tsx), never a real desktop/WS session: that's the one case where
// FreeCAD is dana.platform.mock's headless trimesh driver (no native FreeCAD
// rendering) and CoderPlugin/Aider's execute_code_task is unavailable at all
// (app.py's _harden_tool_registry pops it before the Space ever accepts a
// turn — see that function's own docstring). Dismissal persists across
// reloads via localStorage, same convention as InspectorDock's collapse
// state, but does NOT persist across a page navigation within the same
// load — there is no server-side session flag for this, purely a per-browser
// "I've seen this" marker.
export function WebDemoBanner() {
  const [dismissed, setDismissed] = useState(readDismissed);

  if (dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      // best-effort only — a blocked/private localStorage just means this
      // reappears next reload, nothing else depends on it
    }
  };

  return (
    <div className="web-demo-banner" role="status">
      <span className="web-demo-banner__text">
        <strong>Dana Web Demo</strong> — you're viewing the sandboxed headless demo. Native desktop
        features like local FreeCAD rendering and Aider self-healing are simulated or unavailable here.{" "}
        <a
          href="https://github.com/Cascade-Router/Dana"
          target="_blank"
          rel="noopener noreferrer"
          className="web-demo-banner__link"
        >
          Get the desktop app on GitHub ↗
        </a>
      </span>
      <button
        type="button"
        className="web-demo-banner__dismiss"
        title="Dismiss"
        aria-label="Dismiss"
        onClick={dismiss}
      >
        ✕
      </button>
    </div>
  );
}
