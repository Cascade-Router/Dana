"""Dry-run Phase 5: compress ~2k-token Python into dense summary + Chroma retrieval.

Prototypes MemoryCompressor + idle_compressed collection without patching production.
"""

from __future__ import annotations

import ast
import hashlib
import re
import tempfile
import time
from pathlib import Path

from dana.memory.vault import FakeEmbeddings


def _approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text or "")))


def compress_text_to_latent_summary(raw_text: str, *, max_tokens: int = 180) -> dict:
    """Prototype: AST-aware dense structural summary (<~200 tokens) + embedding."""
    text = (raw_text or "").strip()
    if not text:
        return {"summary": "", "embedding": [], "tokens": 0, "kind": "empty"}

    kind = "text"
    parts: list[str] = []
    try:
        tree = ast.parse(text)
        kind = "python"
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node).strip())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                ret = ast.unparse(node.returns) if node.returns else None
                sig = f"def {node.name}({', '.join(args)})"
                if ret:
                    sig += f" -> {ret}"
                # First meaningful statements (skip docstring)
                body_bits: list[str] = []
                for stmt in node.body[:6]:
                    if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, ast.Constant
                    ):
                        continue
                    body_bits.append(ast.unparse(stmt).strip().splitlines()[0][:80])
                parts.append(sig + (" | " + "; ".join(body_bits) if body_bits else ""))
            elif isinstance(node, ast.ClassDef):
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                parts.append(f"class {node.name}: methods={methods[:12]}")
        if imports:
            parts.insert(0, "imports: " + "; ".join(imports[:12]))
    except SyntaxError:
        # Lightweight C++/generic structural scrape
        if re.search(r"\b(?:class|struct|namespace|template)\b", text):
            kind = "cpp_or_code"
        sigs = re.findall(
            r"(?:^|\n)\s*(?:[\w:<>\*&]+\s+)+(\w+)\s*\(([^;{}]{0,120})\)\s*(?:const)?\s*[{;]",
            text,
        )
        for name, args in sigs[:20]:
            parts.append(f"{name}({args.strip()[:80]})")
        if not parts:
            # Fall back to first non-empty dense lines
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts.append(line[:100])
                if len(parts) >= 18:
                    break

    summary = " | ".join(parts)
    # Hard trim to ~max_tokens words
    words = summary.split()
    if len(words) > max_tokens:
        summary = " ".join(words[:max_tokens]) + "…"
    emb = FakeEmbeddings().embed_query(summary)
    return {
        "summary": summary,
        "embedding": emb,
        "tokens": _approx_tokens(summary),
        "kind": kind,
        "raw_tokens": _approx_tokens(text),
    }


def main() -> None:
    # ~2k-token synthetic Python module (structural, not boilerplate-heavy noise)
    lines = [
        '"""Navigation planning helpers for multi-agent coordination."""',
        "from __future__ import annotations",
        "import math",
        "from dataclasses import dataclass",
        "from typing import Iterable, Sequence",
        "",
        "@dataclass",
        "class Pose2D:",
        "    x: float",
        "    y: float",
        "    yaw: float",
        "",
        "class Costmap:",
        "    def __init__(self, width: int, height: int, resolution: float) -> None:",
        "        self.width = width",
        "        self.height = height",
        "        self.resolution = resolution",
        "        self.grid = [[0.0 for _ in range(width)] for _ in range(height)]",
        "",
        "    def inflate(self, radius_m: float) -> None:",
        "        cells = max(1, int(radius_m / self.resolution))",
        "        for y in range(self.height):",
        "            for x in range(self.width):",
        "                if self.grid[y][x] >= 1.0:",
        "                    for dy in range(-cells, cells + 1):",
        "                        for dx in range(-cells, cells + 1):",
        "                            yy, xx = y + dy, x + dx",
        "                            if 0 <= yy < self.height and 0 <= xx < self.width:",
        "                                self.grid[yy][xx] = max(self.grid[yy][xx], 0.5)",
        "",
        "def a_star(start: Pose2D, goal: Pose2D, costmap: Costmap) -> list[Pose2D]:",
        "    open_set = { (int(start.x), int(start.y)) }",
        "    came_from = {}",
        "    g = { (int(start.x), int(start.y)): 0.0 }",
        "    while open_set:",
        "        current = min(open_set, key=lambda n: g.get(n, 1e9))",
        "        if current == (int(goal.x), int(goal.y)):",
        "            path = [current]",
        "            while current in came_from:",
        "                current = came_from[current]",
        "                path.append(current)",
        "            path.reverse()",
        "            return [Pose2D(float(x), float(y), 0.0) for x, y in path]",
        "        open_set.remove(current)",
        "        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):",
        "            nxt = (current[0]+dx, current[1]+dy)",
        "            if not (0 <= nxt[0] < costmap.width and 0 <= nxt[1] < costmap.height):",
        "                continue",
        "            tentative = g[current] + 1.0 + costmap.grid[nxt[1]][nxt[0]]",
        "            if tentative < g.get(nxt, 1e9):",
        "                came_from[nxt] = current",
        "                g[nxt] = tentative",
        "                open_set.add(nxt)",
        "    return []",
        "",
        "def smooth_path(path: Sequence[Pose2D], iterations: int = 20) -> list[Pose2D]:",
        "    if len(path) < 3:",
        "        return list(path)",
        "    pts = [Pose2D(p.x, p.y, p.yaw) for p in path]",
        "    for _ in range(iterations):",
        "        for i in range(1, len(pts) - 1):",
        "            pts[i].x = 0.5 * pts[i].x + 0.25 * (pts[i-1].x + pts[i+1].x)",
        "            pts[i].y = 0.5 * pts[i].y + 0.25 * (pts[i-1].y + pts[i+1].y)",
        "    return pts",
        "",
        "def coordinate_fleet(agents: Iterable[str], goal: Pose2D) -> dict[str, list[Pose2D]]:",
        "    out = {}",
        "    cmap = Costmap(64, 64, 0.05)",
        "    cmap.inflate(0.2)",
        "    for name in agents:",
        "        start = Pose2D(0.0, 0.0, 0.0)",
        "        path = a_star(start, goal, cmap)",
        "        out[name] = smooth_path(path)",
        "    return out",
    ]
    # Pad with helper stubs to reach ~2000 tokens without useless noise
    for i in range(120):
        lines.append(f"def helper_metric_{i}(values: Sequence[float], scale: float = 1.0) -> float:")
        lines.append(
            f"    total = sum(float(v) * scale for v in values) if values else 0.0"
        )
        lines.append(
            f"    return total / max(1, len(list(values))) if values else 0.0  # metric_{i}"
        )
        lines.append("")
    raw = "\n".join(lines)
    raw_tokens = _approx_tokens(raw)
    assert raw_tokens >= 2000, raw_tokens

    compressed = compress_text_to_latent_summary(raw, max_tokens=180)
    assert compressed["tokens"] <= 200, compressed["tokens"]
    assert compressed["kind"] == "python"
    assert "a_star" in compressed["summary"]
    assert "Costmap" in compressed["summary"] or "class Costmap" in compressed["summary"]

    # Temporary Chroma idle_compressed collection + retrieval preference demo
    import chromadb

    tmp = tempfile.mkdtemp(prefix="dana_idle_compressed_")
    try:
        client = chromadb.PersistentClient(path=tmp)
        col = client.get_or_create_collection(
            name="idle_compressed",
            metadata={"hnsw:space": "cosine"},
        )
        emb_model = FakeEmbeddings()
        summary = compressed["summary"]
        doc_id = "idle_" + hashlib.sha1(summary.encode("utf-8")).hexdigest()[:16]
        emb = emb_model.embed_documents([summary])[0]
        col.upsert(
            ids=[doc_id],
            documents=[summary],
            embeddings=[emb],
            metadatas=[
                {
                    "source": "idle_compressed",
                    "kind": compressed["kind"],
                    "raw_tokens": int(compressed["raw_tokens"]),
                    "summary_tokens": int(compressed["tokens"]),
                    "ts": float(time.time()),
                }
            ],
        )
        # Prefer idle_compressed first (Phase 5 retrieval integration prototype)
        q = "a_star path planning costmap inflate"
        q_emb = emb_model.embed_query(q)
        hit = col.query(query_embeddings=[q_emb], n_results=1)
        docs = (hit.get("documents") or [[]])[0]
        assert docs and "a_star" in docs[0]
        print("RAW_TOKENS", compressed["raw_tokens"])
        print("SUMMARY_TOKENS", compressed["tokens"])
        print("SUMMARY", summary[:280].replace("\n", " "))
        print("RETRIEVED_OK", docs[0][:120].replace("\n", " "))
        print("DRYRUN_OK")
        # Release Chroma handles before cleanup (Windows file locks).
        del col, client
    finally:
        try:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
