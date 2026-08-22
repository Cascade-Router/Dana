import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { resolveApiUrl } from "../lib/apiBase";
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
    fetch(resolveApiUrl("/api/workspace/tree"))
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setTree(data.tree))
      .catch((err) => setTreeError(String(err instanceof Error ? err.message : err)));
  }, []);

  const loadMounts = useCallback(() => {
    fetch(resolveApiUrl("/api/workspace/mounts"))
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
      const res = await fetch(resolveApiUrl("/api/workspace/mount"), {
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
