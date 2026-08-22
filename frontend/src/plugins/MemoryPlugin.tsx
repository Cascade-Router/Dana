import { useCallback, useEffect, useState } from "react";
import { resolveApiUrl } from "../lib/apiBase";
import type { PluginComponentProps } from "./types";
import "./MemoryPlugin.css";

type MemorySection = { section: string; content: string };

function toSections(memory: Record<string, string>): MemorySection[] {
  return Object.entries(memory)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([section, content]) => ({ section, content }));
}

function toMemory(sections: MemorySection[]): Record<string, string> {
  return Object.fromEntries(sections.map((s) => [s.section, s.content]));
}

// Direct user visibility into, and control over, the agent's persistent
// Core Memory (dana/plugins/memory/core_memory.py via /api/memory) — the
// SAME on-disk file the agent's own update_core_memory ReAct tool writes,
// and the same content every session's system prompt reads back. Ignores
// the CAD-shaped PluginComponentProps every plugin currently receives
// (same pattern as WorkspacePlugin) — this plugin manages its own state
// via REST calls instead.
//
// The backend's POST /api/memory is a full overwrite (see
// dana.plugins.memory.core_memory.replace_core_memory), so every mutating
// action here (save/delete/add) always sends the WHOLE current section
// list back, not a per-section patch.
export default function MemoryPlugin(_props: PluginComponentProps) {
  const [sections, setSections] = useState<MemorySection[] | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busySection, setBusySection] = useState<string | null>(null);
  const [newSection, setNewSection] = useState("");
  const [newContent, setNewContent] = useState("");

  const load = useCallback(() => {
    setError(null);
    fetch(resolveApiUrl("/api/memory"))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const loaded = toSections(data.memory ?? {});
        setSections(loaded);
        setDrafts(Object.fromEntries(loaded.map((s) => [s.section, s.content])));
      })
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const persist = useCallback((next: MemorySection[]) => {
    return fetch(resolveApiUrl("/api/memory"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(toMemory(next)),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const saved = toSections(data.memory ?? {});
        setSections(saved);
        setDrafts(Object.fromEntries(saved.map((s) => [s.section, s.content])));
      })
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  const saveSection = useCallback(
    async (section: string) => {
      if (!sections) return;
      setBusySection(section);
      const next = sections.map((s) =>
        s.section === section ? { ...s, content: drafts[section] ?? s.content } : s
      );
      await persist(next);
      setBusySection(null);
    },
    [sections, drafts, persist]
  );

  const deleteSection = useCallback(
    async (section: string) => {
      if (!sections) return;
      setBusySection(section);
      await persist(sections.filter((s) => s.section !== section));
      setBusySection(null);
    },
    [sections, persist]
  );

  const addSection = useCallback(async () => {
    const name = newSection.trim();
    if (!name || !sections) return;
    if (sections.some((s) => s.section === name)) {
      setError(`A section named "${name}" already exists.`);
      return;
    }
    setBusySection(name);
    await persist([...sections, { section: name, content: newContent }]);
    setNewSection("");
    setNewContent("");
    setBusySection(null);
  }, [newSection, newContent, sections, persist]);

  return (
    <div className="memory-plugin">
      <div className="memory-plugin__header">
        <span>Core Memory</span>
        <button type="button" className="memory-plugin__refresh" onClick={load} title="Refresh">
          ↻
        </button>
      </div>

      <div className="memory-plugin__body">
        {error && <div className="memory-plugin__error">{error}</div>}
        {!sections && !error && <div className="memory-plugin__placeholder-text">Loading…</div>}
        {sections && sections.length === 0 && (
          <div className="memory-plugin__placeholder-text">
            Dana hasn&apos;t saved anything to Core Memory yet.
          </div>
        )}

        {sections?.map((s) => {
          const draft = drafts[s.section] ?? s.content;
          const dirty = draft !== s.content;
          const busy = busySection === s.section;
          return (
            <div key={s.section} className="memory-plugin__card">
              <div className="memory-plugin__card-title">{s.section}</div>
              <textarea
                className="memory-plugin__textarea"
                value={draft}
                onChange={(e) => setDrafts((prev) => ({ ...prev, [s.section]: e.target.value }))}
                rows={Math.min(8, Math.max(2, draft.split("\n").length))}
              />
              <div className="memory-plugin__card-actions">
                <button type="button" disabled={!dirty || busy} onClick={() => saveSection(s.section)}>
                  {busy ? "Saving…" : "Save"}
                </button>
                <button
                  type="button"
                  className="memory-plugin__delete"
                  disabled={busy}
                  onClick={() => deleteSection(s.section)}
                >
                  Delete
                </button>
              </div>
            </div>
          );
        })}

        <div className="memory-plugin__card memory-plugin__card--add">
          <div className="memory-plugin__card-title">Add section</div>
          <input
            className="memory-plugin__input"
            value={newSection}
            onChange={(e) => setNewSection(e.target.value)}
            placeholder="Section name (e.g. user_preferences)"
          />
          <textarea
            className="memory-plugin__textarea"
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="Content"
            rows={3}
          />
          <div className="memory-plugin__card-actions">
            <button
              type="button"
              disabled={!newSection.trim() || busySection === newSection.trim()}
              onClick={addSection}
            >
              {busySection === newSection.trim() ? "Adding…" : "Add"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
