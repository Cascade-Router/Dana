import { useState } from "react";
import type { MemoryState } from "../lib/useChatSocket";
import "./MemoryViewer.css";

type MemoryViewerProps = {
  memory: MemoryState;
  onClose: () => void;
};

export function MemoryViewer({ memory, onClose }: MemoryViewerProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  const memoryEntries = Object.entries(memory);
  const hasMemory = memoryEntries.length > 0;

  return (
    <div className="memory-viewer">
      <div className="memory-viewer__backdrop" onClick={onClose} />
      <div className={`memory-viewer__panel ${isCollapsed ? "memory-viewer__panel--collapsed" : ""}`}>
        <div className="memory-viewer__header">
          <div className="memory-viewer__title">
            <span className="memory-viewer__icon">🧠</span>
            <h3>Core Memory</h3>
            <span className="memory-viewer__count">
              {hasMemory ? `${memoryEntries.length} item${memoryEntries.length === 1 ? "" : "s"}` : "Empty"}
            </span>
          </div>
          <div className="memory-viewer__controls">
            <button
              type="button"
              className="memory-viewer__toggle"
              onClick={() => setIsCollapsed(!isCollapsed)}
              title={isCollapsed ? "Expand" : "Collapse"}
            >
              {isCollapsed ? "▲" : "▼"}
            </button>
            <button
              type="button"
              className="memory-viewer__close"
              onClick={onClose}
              title="Close"
            >
              ✕
            </button>
          </div>
        </div>
        
        {!isCollapsed && (
          <div className="memory-viewer__content">
            {hasMemory ? (
              <div className="memory-viewer__entries">
                {memoryEntries.map(([section, content]) => (
                  <div key={section} className="memory-viewer__entry">
                    <div className="memory-viewer__section">{section}</div>
                    <div className="memory-viewer__content-text">{content}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="memory-viewer__empty">
                <p>No core memory stored yet.</p>
                <p className="memory-viewer__hint">
                  The agent will automatically save important facts and preferences here as you interact.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
