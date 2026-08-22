import { useCallback, useEffect, useState } from "react";
import { resolveApiUrl } from "../lib/apiBase";
import type { PluginComponentProps } from "./types";
import "./SkillsPlugin.css";

type Skill = { name: string; description: string; code: string };

// Autonomous Skill Acquisition's transparency + editing UI — direct user
// visibility into (and control over) every skill the agent has taught
// itself via the save_new_skill ReAct tool (dana/api/skills.py ->
// dana.core.react_dispatch's "user_skills" capability domain). Ignores the
// CAD-shaped PluginComponentProps every plugin currently receives (same
// pattern as WorkspacePlugin/MemoryPlugin) — this plugin manages its own
// state via REST calls instead.
//
// Editing is a plain <textarea> (no Monaco/CodeMirror — see this plugin's
// task constraints): PUT /api/skills/{name} writes the FULL edited source
// verbatim and hot-reloads the registry, returning a 400 with the exact
// parse/validation error if the edit doesn't load cleanly (dana.core.
// skill_loader.validate_skill_file) — surfaced inline, not as a crash.
export default function SkillsPlugin(_props: PluginComponentProps) {
  const [skills, setSkills] = useState<Skill[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [editingName, setEditingName] = useState<string | null>(null);
  const [draftCode, setDraftCode] = useState("");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(() => {
    setError(null);
    fetch(resolveApiUrl("/api/skills"))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setSkills(data.skills ?? []))
      .catch((err) => setError(String(err instanceof Error ? err.message : err)));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleDelete = useCallback(
    (name: string) => {
      setBusy(name);
      fetch(resolveApiUrl(`/api/skills/${name}`), { method: "DELETE" })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(() => {
          if (editingName === name) setEditingName(null);
          load();
        })
        .catch((err) => setError(String(err instanceof Error ? err.message : err)))
        .finally(() => setBusy(null));
    },
    [editingName, load]
  );

  const startEdit = useCallback((skill: Skill) => {
    setEditingName(skill.name);
    setDraftCode(skill.code);
    setSaveError(null);
  }, []);

  const cancelEdit = useCallback(() => {
    setEditingName(null);
    setSaveError(null);
  }, []);

  const handleSave = useCallback(
    (name: string) => {
      setSaving(true);
      setSaveError(null);
      fetch(resolveApiUrl(`/api/skills/${name}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: draftCode }),
      })
        .then(async (res) => {
          if (!res.ok) {
            const body = await res.json().catch(() => ({}));
            throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then(() => {
          setEditingName(null);
          load();
        })
        .catch((err) => setSaveError(String(err instanceof Error ? err.message : err)))
        .finally(() => setSaving(false));
    },
    [draftCode, load]
  );

  return (
    <div className="skills-plugin">
      <div className="skills-plugin__header">
        <span>Learned Skills</span>
        <button type="button" className="skills-plugin__refresh" onClick={load} title="Refresh">
          ↻
        </button>
      </div>

      <div className="skills-plugin__body">
        {error && <div className="skills-plugin__error">{error}</div>}
        {!skills && !error && <div className="skills-plugin__placeholder">Loading…</div>}
        {skills && skills.length === 0 && (
          <div className="skills-plugin__placeholder">Dana hasn&apos;t taught herself any skills yet.</div>
        )}

        {skills?.map((s) => {
          const isEditing = editingName === s.name;
          return (
            <div key={s.name} className="skills-plugin__card">
              <div className="skills-plugin__card-header">
                <span className="skills-plugin__card-name">{s.name}</span>
                <div className="skills-plugin__card-actions">
                  {!isEditing && (
                    <button type="button" className="skills-plugin__edit" onClick={() => startEdit(s)}>
                      Edit
                    </button>
                  )}
                  <button
                    type="button"
                    className="skills-plugin__delete"
                    disabled={busy === s.name}
                    onClick={() => handleDelete(s.name)}
                  >
                    {busy === s.name ? "Deleting…" : "Delete"}
                  </button>
                </div>
              </div>

              {s.description && <div className="skills-plugin__card-description">{s.description}</div>}

              {isEditing ? (
                <div className="skills-plugin__editor">
                  <textarea
                    className="skills-plugin__textarea"
                    value={draftCode}
                    onChange={(e) => setDraftCode(e.target.value)}
                    rows={Math.min(24, Math.max(8, draftCode.split("\n").length))}
                    spellCheck={false}
                  />
                  {saveError && <div className="skills-plugin__save-error">{saveError}</div>}
                  <div className="skills-plugin__editor-actions">
                    <button type="button" disabled={saving} onClick={() => handleSave(s.name)}>
                      {saving ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      className="skills-plugin__cancel"
                      disabled={saving}
                      onClick={cancelEdit}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <details className="skills-plugin__code-details">
                  <summary>View source</summary>
                  <pre className="skills-plugin__code">{s.code}</pre>
                </details>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
