"""Export Donna's production LangGraph ReAct topology as Mermaid.

Uses LangGraph's native ``CompiledGraph.get_graph().draw_mermaid()`` — never
asks an LLM to guess the structure.

Usage:
  python -m dana.tools.export_graph
  python -m dana.tools.export_graph --out docs/architecture/donna_architecture.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    try:
        from dana.paths import PROJECT_ROOT

        return Path(PROJECT_ROOT)
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[2]


async def _stub_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Topology-only stub — export never invokes nodes."""
    return {
        "messages": list(state.get("messages") or []),
        "iterations": int(state.get("iterations") or 0),
        "last_obs": str(state.get("last_obs") or ""),
        "final_raw": "",
        "halt": True,
    }


async def _stub_tools(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": list(state.get("messages") or []),
        "iterations": int(state.get("iterations") or 0),
        "last_obs": str(state.get("last_obs") or ""),
        "final_raw": "",
        "halt": True,
    }


def compile_production_react_app() -> Any:
    """Build the compiled ReAct app with the same wiring as live Donna."""
    from dana.agentic_react_graph import compile_donna_react_graph

    return compile_donna_react_graph(_stub_agent, _stub_tools)


def export_mermaid(app: Any) -> str:
    """Return native Mermaid source from ``app.get_graph()``."""
    graph = app.get_graph()
    draw = getattr(graph, "draw_mermaid", None)
    if not callable(draw):
        raise RuntimeError(
            "LangGraph graph object has no draw_mermaid(); "
            "upgrade langgraph or check get_graph() API."
        )
    text = draw()
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("draw_mermaid() returned empty output")
    return text


def try_export_png(app: Any, dest: Path) -> Path | None:
    """Best-effort PNG via ``draw_mermaid_png`` when dependencies are present."""
    graph = app.get_graph()
    draw_png = getattr(graph, "draw_mermaid_png", None)
    if not callable(draw_png):
        return None
    try:
        raw = draw_png()
    except Exception as exc:  # noqa: BLE001
        print(f"[export_graph] PNG skipped ({exc})", file=sys.stderr)
        return None
    if not raw:
        return None
    dest.write_bytes(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
    return dest


def build_markdown(mermaid: str) -> str:
    """Wrap Mermaid in a markdown doc with routing-audit notes."""
    return f"""# Donna LangGraph Architecture

Native export from the production ReAct ``StateGraph`` via
``CompiledGraph.get_graph().draw_mermaid()``
(``dana.agentic_react_graph.compile_donna_react_graph``).

## Graph

```mermaid
{mermaid.strip()}
```

## Topology notes (for routing audits)

| Edge / path | Meaning |
|---|---|
| `START -> agent` | Every developer/vision/research turn enters the ReAct router/synthesis node. |
| `agent -> tools` | Conditional: last AI message has ``tool_calls`` (native bind_tools or recovered JSON). |
| `agent -> END` | Conditional: ``halt`` or no tool calls (spoken final answer). |
| `tools -> agent` | Continue ReAct after tool observations (unless ``halt``). |
| `tools -> END` | Max-iters / forced halt after tools. |

### Outside this diagram (still part of live routing)

These policies run **before** or **inside** node bodies - they do not add extra
LangGraph nodes, but matter when auditing starvation / vision bugs:

- **Mode foresight** (`dana.agentic.get_donna_mode`): chat bypasses this graph;
  vision/research keep ReAct and may force ``analyze_visual_context`` into the
  bind set via ``merge_bound_tool_ids``.
- **Broker foresight** (`IntentBroker.parse_utterance`): may seed a forced tool
  into the agent node before the first LLM step.
- **Explicit tool merge**: tool ids spelled in the user text are always_include'd
  so mode overrides cannot starve e.g. ``draft_cursor_prompt``.
- **Vision JIT**: ``analyze_visual_context`` executes inside the ``tools`` node
  (direct YOLO path), not as a separate graph node.

Regenerate with:

```bash
python -m dana.tools.export_graph
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Donna's production LangGraph ReAct topology as Mermaid."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown output path (default: <repo>/docs/architecture/donna_architecture.md)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG path (default: <repo>/docs/architecture/donna_architecture.png when drawable)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG export even if draw_mermaid_png is available",
    )
    args = parser.parse_args(argv)

    root = _project_root()
    arch_dir = root / "docs" / "architecture"
    out_md = args.out or (arch_dir / "donna_architecture.md")
    out_png = args.png or (arch_dir / "donna_architecture.png")

    print("[export_graph] Compiling production ReAct graph…", flush=True)
    app = compile_production_react_app()
    print("[export_graph] Drawing Mermaid via get_graph().draw_mermaid()…", flush=True)
    mermaid = export_mermaid(app)

    # Sanity: production conditional paths must appear in the native dump.
    lowered = mermaid.lower()
    for needle in ("agent", "tools"):
        if needle not in lowered:
            print(
                f"[export_graph] ERROR: expected node `{needle}` missing from Mermaid",
                file=sys.stderr,
            )
            return 2

    out_md.write_text(build_markdown(mermaid), encoding="utf-8")
    print(f"[export_graph] Wrote {out_md}", flush=True)

    if not args.no_png:
        png_path = try_export_png(app, out_png)
        if png_path is not None:
            print(f"[export_graph] Wrote {png_path}", flush=True)
        else:
            print(
                "[export_graph] PNG not written (draw_mermaid_png unavailable "
                "or missing renderer deps)",
                flush=True,
            )

    print("[export_graph] Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
