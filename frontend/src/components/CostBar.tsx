import { useMemo } from "react";
import type { CostState } from "../lib/useChatSocket";

// Fixed, deterministic palette — a model's segment color is hashed from its
// own name (not "first model seen this session"), so it never shifts if
// by_model's key order changes across usage_update events.
const SEGMENT_COLORS = ["#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#fb7185", "#22d3ee", "#facc15"];

function colorForModel(model: string): string {
  let hash = 0;
  for (let i = 0; i < model.length; i++) {
    hash = (hash * 31 + model.charCodeAt(i)) >>> 0;
  }
  return SEGMENT_COLORS[hash % SEGMENT_COLORS.length];
}

function formatUsd(value: number): string {
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

type Props = {
  cost: CostState;
};

// Renders dana/api/server.py's "usage_update" broadcasts (session["cost_tracking"],
// accumulated once per next_react_turn iteration off dana/core/pricing.py's
// OpenRouter price table) — see useChatSocket's CostState/`case "usage_update"`
// for where this data actually comes from. Purely presentational: takes the
// already-derived CostState as a prop rather than opening its own socket, the
// same pattern every other App.tsx-level display (e.g. the model-provider
// badge) already follows.
export function CostBar({ cost }: Props) {
  const { activeModel, sessionTotalUsd, byModel } = cost;

  const segments = useMemo(() => {
    const entries = Object.entries(byModel).filter(([, value]) => value > 0);
    const total = entries.reduce((sum, [, value]) => sum + value, 0);
    if (total <= 0) return [];
    return entries
      .map(([model, value]) => ({ model, value, pct: (value / total) * 100 }))
      .sort((a, b) => b.value - a.value);
  }, [byModel]);

  // Nothing to show yet — no turn carrying priced usage has completed this
  // session (a local Ollama-only session, e.g., never populates byModel
  // since dana.core.pricing has no entry for it — see estimate_cost_usd).
  if (!activeModel && segments.length === 0) return null;

  return (
    <div className="cost-bar" title="Session LLM cost (OpenRouter pricing estimate)">
      <span className="cost-bar__model">{activeModel ?? "—"}</span>
      <div className="cost-bar__track">
        {segments.length === 0 ? (
          <div className="cost-bar__empty" />
        ) : (
          segments.map((seg) => (
            <div
              key={seg.model}
              className="cost-bar__segment"
              style={{ width: `${seg.pct}%`, background: colorForModel(seg.model) }}
              title={`${seg.model}: ${formatUsd(seg.value)}`}
            />
          ))
        )}
      </div>
      <span className="cost-bar__total">{formatUsd(sessionTotalUsd)}</span>
    </div>
  );
}
