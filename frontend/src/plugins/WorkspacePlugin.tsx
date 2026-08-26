import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { apiFetch, IS_GRADIO_MODE, resolveApiUrl } from "../lib/apiBase";
import { fetchGradioArtifacts, type GradioArtifact } from "../lib/gradioChatClient";
import type { PluginComponentProps } from "./types";
import "./WorkspacePlugin.css";

type WorkspaceNode =
  | { type: "directory"; name: string; path: string; children: WorkspaceNode[] }
  | { type: "file"; name: string; path: string; size: number };

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg", "bmp"]);

function isImagePath(path: string): boolean {
  const ext = path.split(".").pop()?.toLowerCase();
  return !!ext && IMAGE_EXTENSIONS.has(ext);
}

// Each path segment is encoded individually — the slashes themselves are
// the /file/{file_path:path} route's own segment separators, not part of
// any single segment's content.
function workspaceFileUrl(path: string): string {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return resolveApiUrl(`/api/workspace/file/${encoded}`);
}

function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

const _SPACE_URL = import.meta.env.VITE_HF_SPACE_URL as string;

// Gradio mode has no REST API at all (see apiBase.ts's IS_GRADIO_MODE) —
// the tree/mount browser below is entirely REST-backed and would just show
// perpetual "Failed to load" errors there, so this tab shows the SAME
// generated-artifact list app.py's hidden "artifacts" endpoint already
// exposes to CadToolbar's Export dropdown (fetchGradioArtifacts), just
// hydrated as a proper list here instead of a dropdown menu.
function GradioWorkspaceArtifacts() {
  const [artifacts, setArtifacts] = useState<GradioArtifact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchGradioArtifacts(_SPACE_URL)
      .then((files) => setArtifacts(files))
      .catch((err) => {
        console.error("[WorkspacePlugin] fetchGradioArtifacts failed:", err);
        setError(String(err instanceof Error ? err.message : err));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="workspace-plugin workspace-plugin--gradio">
      <div className="workspace-plugin__sidebar-header">
        <span>Generated Artifacts</span>
        <button type="button" className="workspace-plugin__refresh" onClick={refresh} title="Refresh">
          ↻
        </button>
      </div>
      <div className="workspace-plugin__artifacts">
        {loading && artifacts.length === 0 && <div className="workspace-plugin__placeholder-text">Loading…</div>}
        {error && <div className="workspace-plugin__error">Failed to load: {error}</div>}
        {!loading && !error && artifacts.length === 0 && (
          <div className="workspace-plugin__placeholder-text">No artifacts generated yet.</div>
        )}
        {artifacts.map((a) => (
          <div key={a.url} className="workspace-plugin__artifact-row">
            <div className="workspace-plugin__artifact-info">
              <span className="workspace-plugin__artifact-name" title={a.filename}>
                {a.filename}
              </span>
              <span className="workspace-plugin__artifact-meta">
                {a.format ? a.format.toUpperCase() : "FILE"} · {formatFileSize(a.size_bytes)}
              </span>
            </div>
            <a
              className="workspace-plugin__artifact-download"
              href={a.url}
              target="_blank"
              rel="noreferrer"
              download={a.filename}
            >
              ⬇ Download
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}

type TreeRowProps = {
  node: WorkspaceNode;
  depth: number;
  selectedPath: string | null;
  collapsed: Set<string>;
  onToggle: (path: string) => void;
  onSelectFile: (path: string) => void;
};

function TreeRow({ node, depth, selectedPath, collapsed, onToggle, onSelectFile }: TreeRowProps) {
  const indent = { paddingLeft: `${8 + depth * 14}px` };

  if (node.type === "directory") {
    const isOpen = depth === 0 || !collapsed.has(node.path);
    return (
      <div>
        <button
          type="button"
          className="workspace-plugin__row workspace-plugin__row--dir"
          style={indent}
          onClick={() => onToggle(node.path)}
        >
          <span className="workspace-plugin__caret">{isOpen ? "▾" : "▸"}</span>
          {node.name}
        </button>
        {isOpen &&
          (node.children.length > 0 ? (
            node.children.map((child) => (
              <TreeRow
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                collapsed={collapsed}
                onToggle={onToggle}
                onSelectFile={onSelectFile}
              />
            ))
          ) : (
            <div className="workspace-plugin__empty" style={{ paddingLeft: `${8 + (depth + 1) * 14}px` }}>
              empty
            </div>
          ))}
      </div>
    );
  }

  return (
    <button
      type="button"
      className={`workspace-plugin__row workspace-plugin__row--file ${
        selectedPath === node.path ? "workspace-plugin__row--selected" : ""
      }`}
      style={indent}
      onClick={() => onSelectFile(node.path)}
      title={node.path}
    >
      {node.name}
    </button>
  );
}

// Read-only viewer for the agent's own sandbox (AGENT_WORKSPACE_DIR) — the
// second plugin proving out the multi-plugin architecture alongside
// CadPlugin. Deliberately ignores the CAD-shaped props every plugin
// component currently receives (see types.ts's PluginComponentProps note)
// — this plugin manages its own state via the /api/workspace/* REST
// endpoints instead. Strictly read-only by design: there is no write/edit
// path here at all; every actual file mutation goes through the agent's
// own os_tools ReAct tools (write_file, run_python_script), never this UI.
export default function WorkspacePlugin(_props: PluginComponentProps) {
  // IS_GRADIO_MODE is a build-time constant (import.meta.env.VITE_HF_SPACE_URL
  // never changes across a render — see useChat.ts's identical reasoning),
  // so branching before any hooks run below is safe: this condition can
  // never flip within one running build, unlike a normal conditional hook
  // call would be.
  if (IS_GRADIO_MODE) {
    return <GradioWorkspaceArtifacts />;
  }
  return <RestWorkspaceTree />;
}

function RestWorkspaceTree() {
  const [tree, setTree] = useState<WorkspaceNode | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);

  // Dynamic Workspace Mounting — registered external directories the user
  // has explicitly granted the agent access to (dana/api/workspace.py's
  // mounts.json registry, via list_directory/read_file/write_file's
  // allowed_mounts). `mounts` mirrors that SAME on-disk registry (GET
  // /api/workspace/mounts), not just this session's own additions, so a
  // mount registered from another window still shows up here on refresh.
  const [mounts, setMounts] = useState<string[]>([]);
  const [mountError, setMountError] = useState<string | null>(null);
  const [mounting, setMounting] = useState(false);

  const loadTree = useCallback(() => {
    setTreeError(null);
    apiFetch("/api/workspace/tree")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setTree(data.tree))
      .catch((err) => setTreeError(String(err instanceof Error ? err.message : err)));
  }, []);

  const loadMounts = useCallback(() => {
    apiFetch("/api/workspace/mounts")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setMounts(Array.isArray(data.mounted_directories) ? data.mounted_directories : []))
      .catch(() => {}); // best-effort — the mount button/tree still work without this list
  }, []);

  useEffect(() => {
    loadTree();
    loadMounts();
  }, [loadTree, loadMounts]);

  // Native OS folder picker (Tauri's dialog plugin) -> the resulting
  // absolute path is registered via POST /api/workspace/mount, which
  // validates it exists and appends it to the SAME mounts.json registry
  // dana.plugins.os.file_system.resolve_sandboxed_path's allowed_mounts
  // reads on every os_tools call — no separate trust store of its own.
  const handleMountFolder = useCallback(async () => {
    setMountError(null);
    setMounting(true);
    try {
      const selected = await open({ directory: true, multiple: false });
      if (!selected || Array.isArray(selected)) return; // cancelled, or an unexpected multi-select result
      const res = await apiFetch("/api/workspace/mount", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: selected }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      setMounts(Array.isArray(data.mounted_directories) ? data.mounted_directories : []);
    } catch (err) {
      setMountError(String(err instanceof Error ? err.message : err));
    } finally {
      setMounting(false);
    }
  }, []);

  const onToggle = useCallback((path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const onSelectFile = useCallback((path: string) => {
    setSelectedPath(path);
    setContent(null);
    setContentError(null);

    if (isImagePath(path)) return; // the <img> tag below fetches it directly

    setLoadingContent(true);
    fetch(workspaceFileUrl(path))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((text) => setContent(text))
      .catch((err) => setContentError(String(err instanceof Error ? err.message : err)))
      .finally(() => setLoadingContent(false));
  }, []);

  return (
    <div className="workspace-plugin">
      <div className="workspace-plugin__sidebar">
        <div className="workspace-plugin__sidebar-header">
          <span>Workspace</span>
          <button type="button" className="workspace-plugin__refresh" onClick={loadTree} title="Refresh">
            ↻
          </button>
        </div>
        <div className="workspace-plugin__tree">
          {treeError && <div className="workspace-plugin__error">Failed to load: {treeError}</div>}
          {!treeError && !tree && <div className="workspace-plugin__placeholder-text">Loading…</div>}
          {tree && (
            <TreeRow
              node={tree}
              depth={0}
              selectedPath={selectedPath}
              collapsed={collapsed}
              onToggle={onToggle}
              onSelectFile={onSelectFile}
            />
          )}
        </div>

        <div className="workspace-plugin__mounts">
          <button
            type="button"
            className="workspace-plugin__mount-btn"
            onClick={handleMountFolder}
            disabled={mounting}
            title="Grant Dana access to an external folder on this computer"
          >
            {mounting ? "Mounting…" : "+ Mount Local Folder"}
          </button>
          {mountError && <div className="workspace-plugin__error">{mountError}</div>}
          {mounts.length > 0 && (
            <ul className="workspace-plugin__mount-list" title="Directories Dana may also read/write">
              {mounts.map((m) => (
                <li key={m} className="workspace-plugin__mount-item" title={m}>
                  {m}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="workspace-plugin__viewer">
        {!selectedPath && <div className="workspace-plugin__placeholder">Select a file to view it.</div>}
        {selectedPath && isImagePath(selectedPath) && (
          <img className="workspace-plugin__image" src={workspaceFileUrl(selectedPath)} alt={selectedPath} />
        )}
        {selectedPath && !isImagePath(selectedPath) && (
          <>
            {loadingContent && <div className="workspace-plugin__placeholder">Loading…</div>}
            {contentError && <div className="workspace-plugin__error">Failed to load: {contentError}</div>}
            {content !== null && <pre className="workspace-plugin__content">{content}</pre>}
          </>
        )}
      </div>
    </div>
  );
}
