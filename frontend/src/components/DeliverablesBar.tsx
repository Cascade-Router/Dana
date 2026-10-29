import { useEffect, useMemo, useRef, useState } from "react";
import { resolveApiUrl } from "../lib/apiBase";
import { resolveArtifactUrl, type CadArtifact } from "../lib/useCadArtifacts";
import type { ServerEvent } from "../lib/useChatSocket";
import "./DeliverablesBar.css";

// One-click access to what a session has produced — macro, STEP/STL,
// blueprint, BOM — with zero LLM calls. Everything is derived from the
// session-scoped artifact list (dana/api/cad.py) and the event log
// (tool_dispatch_end / python_script_exported), so it also works for a
// resumed session whose terminal_log was restored from disk.

type Props = {
  sessionId: string | null;
  artifacts: CadArtifact[];
  log: ServerEvent[];
  onExportMacro?: () => void;
  onShowBlueprint: () => void;
};

type BomOutput = {
  name?: string;
  path?: string;
  part_count?: number;
  total_mass_g?: number;
  total_cost?: number;
  currency?: string;
  assembly_center_of_mass_mm?: number[];
  statically_stable?: boolean | null;
  stability_margin_mm?: number | null;
  stability_warning?: string | null;
};

const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;

function latestArtifact(artifacts: CadArtifact[], formats: string[]): CadArtifact | undefined {
  // useCadArtifacts lists newest first.
  return artifacts.find((a) => formats.includes(a.format.toLowerCase()));
}

function latestBom(log: ServerEvent[]): BomOutput | null {
  for (let i = log.length - 1; i >= 0; i--) {
    const e = log[i];
    if (e.type === "tool_dispatch_end" && e.tool_id === "generate_assembly_bom" && e.status === "success") {
      return e.output as BomOutput;
    }
  }
  return null;
}

function fmt(value: number | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function DeliverablesBar({ sessionId, artifacts, log, onExportMacro, onShowBlueprint }: Props) {
  const [bomOpen, setBomOpen] = useState(false);
  const [macroPending, setMacroPending] = useState(false);

  const step = latestArtifact(artifacts, ["step", "stp"]);
  const mesh = latestArtifact(artifacts, ["stl", "glb"]);
  const pdf = latestArtifact(artifacts, ["pdf"]);
  const svg = latestArtifact(artifacts, ["svg"]);
  const bom = useMemo(() => latestBom(log), [log]);
  const bomCsv = bom?.path ? artifacts.find((a) => a.filename === basename(bom.path!)) : undefined;

  // Download the macro only for an export THIS click asked for — never for a
  // python_script_exported event replayed from a restored log.
  // Compared by identity, not count: the log is capped, so counts can stall.
  const seenMacroEvent = useRef<ServerEvent | undefined>(undefined);
  useEffect(() => {
    const latest = [...log].reverse().find((e) => e.type === "python_script_exported");
    if (macroPending && latest && latest !== seenMacroEvent.current && latest.type === "python_script_exported") {
      window.open(resolveApiUrl(latest.download_url), "_blank");
      setMacroPending(false);
    }
    seenMacroEvent.current = latest;
  }, [log, macroPending]);

  // A "Nothing to export yet" reply means no event will come; don't stay stuck.
  useEffect(() => {
    if (!macroPending) return;
    const timer = window.setTimeout(() => setMacroPending(false), 15000);
    return () => window.clearTimeout(timer);
  }, [macroPending]);

  const open = (artifact: CadArtifact | undefined) => {
    if (artifact) window.open(resolveArtifactUrl(artifact, sessionId), "_blank");
  };

  return (
    <div className="deliverables-bar">
      <span className="deliverables-bar__label">Deliverables</span>
      {onExportMacro && (
        <button
          type="button"
          className="deliverables-bar__btn"
          title="Replayable FreeCAD macro of this session (0 tokens)"
          disabled={macroPending}
          onClick={() => {
            setMacroPending(true);
            onExportMacro();
          }}
        >
          {macroPending ? "Exporting…" : "Macro .py"}
        </button>
      )}
      <button type="button" className="deliverables-bar__btn" disabled={!step} onClick={() => open(step)}>
        STEP
      </button>
      <button
        type="button"
        className="deliverables-bar__btn"
        disabled={!mesh}
        onClick={() => open(mesh)}
        title={mesh ? "Download the latest mesh" : "No mesh yet"}
      >
        {mesh ? mesh.format.toUpperCase() : "STL"}
      </button>
      <div className="deliverables-bar__group">
        <button type="button" className="deliverables-bar__btn" disabled={!svg} onClick={onShowBlueprint}>
          Blueprint
        </button>
        <button
          type="button"
          className="deliverables-bar__btn deliverables-bar__btn--icon"
          disabled={!pdf}
          title={pdf ? "Download the latest blueprint PDF" : "No blueprint PDF yet"}
          onClick={() => open(pdf)}
        >
          PDF
        </button>
      </div>
      <div className="deliverables-bar__popover-anchor">
        <button
          type="button"
          className="deliverables-bar__btn"
          disabled={!bom}
          onClick={() => setBomOpen((v) => !v)}
        >
          BOM ▾
        </button>
        {bomOpen && bom && (
          <div className="deliverables-bar__popover" role="dialog" aria-label="Bill of materials summary">
            <div className="deliverables-bar__popover-title">{bom.name ?? "Assembly"}</div>
            <dl className="deliverables-bar__stats">
              <dt>Parts</dt>
              <dd>{bom.part_count ?? "—"}</dd>
              <dt>Mass</dt>
              <dd>{fmt(bom.total_mass_g)} g</dd>
              <dt>Cost</dt>
              <dd>
                {fmt(bom.total_cost, 2)} {bom.currency ?? ""}
              </dd>
              <dt>Center of mass</dt>
              <dd>
                {bom.assembly_center_of_mass_mm
                  ? `(${bom.assembly_center_of_mass_mm.map((v) => fmt(v)).join(", ")}) mm`
                  : "—"}
              </dd>
              <dt>Tip-over</dt>
              <dd
                className={
                  bom.statically_stable === false
                    ? "deliverables-bar__bad"
                    : bom.statically_stable
                      ? "deliverables-bar__good"
                      : undefined
                }
              >
                {bom.statically_stable == null
                  ? "not checked"
                  : `${bom.statically_stable ? "stable" : "tips over"} (margin ${fmt(bom.stability_margin_mm ?? undefined)} mm)`}
              </dd>
            </dl>
            {bom.stability_warning && <p className="deliverables-bar__warning">{bom.stability_warning}</p>}
            <button
              type="button"
              className="deliverables-bar__btn"
              disabled={!bomCsv}
              onClick={() => open(bomCsv)}
            >
              Download CSV
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
