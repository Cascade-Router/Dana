import { useState, type ReactNode } from "react";
import "./InspectorDock.css";

export type InspectorTab = {
  id: string;
  label: string;
  /** Single letter/short glyph, matching PluginDefinition's own "no icon
   * font dependency" convention. */
  glyph: string;
  badge?: string | number;
  content: ReactNode;
};

type Props = {
  tabs: InspectorTab[];
  defaultTabId?: string;
};

const COLLAPSE_KEY = "dana:inspectorDock:collapsed";

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === "1";
  } catch {
    return false;
  }
}

// Consolidates what used to be three independent floating overlays (Active
// Plan, Terminal History, Topology Graph) fighting over the same space atop
// the CAD viewport into one docked, tabbed panel — see CadPlugin.tsx. All
// tabs' content stays mounted (toggled via `hidden`, not conditional
// rendering) so switching tabs never tears down the Topology graph's
// ReactFlow canvas (losing its fitView/selection) or the Terminal's scroll
// position.
export function InspectorDock({ tabs, defaultTabId }: Props) {
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const [activeId, setActiveId] = useState(defaultTabId ?? tabs[0]?.id);

  const setCollapsedPersist = (next: boolean) => {
    setCollapsed(next);
    try {
      localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
    } catch {
      // best-effort only — a blocked/private localStorage just means the
      // dock reopens expanded next launch, nothing else depends on it
    }
  };

  if (tabs.length === 0) return null;

  if (collapsed) {
    return (
      <div className="inspector-dock inspector-dock--collapsed">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className="inspector-dock__rail-btn"
            title={tab.label}
            onClick={() => {
              setActiveId(tab.id);
              setCollapsedPersist(false);
            }}
          >
            <span className="inspector-dock__glyph" aria-hidden="true">
              {tab.glyph}
            </span>
            {tab.badge !== undefined && <span className="inspector-dock__badge">{tab.badge}</span>}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="inspector-dock">
      <div className="inspector-dock__tabbar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`inspector-dock__tab ${activeId === tab.id ? "inspector-dock__tab--active" : ""}`}
            onClick={() => setActiveId(tab.id)}
          >
            <span className="inspector-dock__glyph" aria-hidden="true">
              {tab.glyph}
            </span>
            {tab.label}
            {tab.badge !== undefined && <span className="inspector-dock__tab-badge">{tab.badge}</span>}
          </button>
        ))}
        <button
          type="button"
          className="inspector-dock__collapse-btn"
          title="Collapse inspector"
          onClick={() => setCollapsedPersist(true)}
        >
          ▸
        </button>
      </div>
      <div className="inspector-dock__body">
        {tabs.map((tab) => (
          <div key={tab.id} className="inspector-dock__tab-content" hidden={activeId !== tab.id}>
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
}
