"""Phase 5 — Latent memory compression for idle research / sandbox outputs.

Compresses raw text into dense structural summaries (<200 tokens) with
embeddings for the Chroma ``idle_compressed`` collection.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any


def _approx_tokens(text: str) -> int:
    return max(0, len(re.findall(r"\S+", text or "")))


def _trim_tokens(text: str, *, max_tokens: int) -> str:
    words = (text or "").split()
    if len(words) <= max_tokens:
        return " ".join(words)
    return " ".join(words[:max_tokens]) + "…"


class MemoryCompressor:
    """AST-aware dense summarizer for code/research text."""

    def __init__(self, *, max_tokens: int = 180) -> None:
        self.max_tokens = max(40, int(max_tokens))

    def compress_text_to_latent_summary(
        self,
        raw_text: str,
        *,
        embeddings: Any | None = None,
    ) -> dict[str, Any]:
        """Extract structural logic into a dense summary + optional embedding.

        Returns ``{summary, embedding, tokens, kind, raw_tokens}``.
        """
        text = (raw_text or "").strip()
        if not text:
            return {
                "summary": "",
                "embedding": [],
                "tokens": 0,
                "kind": "empty",
                "raw_tokens": 0,
            }

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
                    parts.append(self._py_function_summary(node))
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
            if re.search(r"\b(?:class|struct|namespace|template|void|int|bool)\b", text):
                kind = "cpp_or_code"
            parts.extend(self._cpp_or_generic_parts(text))

        if not parts:
            # Research / prose: keep high-signal sentences, drop filler.
            kind = "research" if kind == "text" else kind
            parts = self._prose_parts(text)

        summary = _trim_tokens(" | ".join(parts), max_tokens=self.max_tokens)
        embedding: list[float] = []
        if embeddings is not None and summary:
            try:
                embedding = list(embeddings.embed_query(summary))
            except Exception:  # noqa: BLE001
                embedding = []
        return {
            "summary": summary,
            "embedding": embedding,
            "tokens": _approx_tokens(summary),
            "kind": kind,
            "raw_tokens": _approx_tokens(text),
        }

    def _py_function_summary(self, node: ast.AST) -> str:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        args = [a.arg for a in node.args.args]
        ret = ast.unparse(node.returns) if node.returns else None
        sig = f"def {node.name}({', '.join(args)})"
        if ret:
            sig += f" -> {ret}"
        body_bits: list[str] = []
        for stmt in node.body[:5]:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            try:
                line = ast.unparse(stmt).strip().splitlines()[0][:90]
            except Exception:  # noqa: BLE001
                continue
            if line:
                body_bits.append(line)
        if body_bits:
            return sig + " | " + "; ".join(body_bits)
        return sig

    def _cpp_or_generic_parts(self, text: str) -> list[str]:
        parts: list[str] = []
        sigs = re.findall(
            r"(?:^|\n)\s*(?:[\w:<>\*&]+\s+)+(\w+)\s*\(([^;{}]{0,120})\)\s*(?:const)?\s*[{;]",
            text,
        )
        for name, args in sigs[:24]:
            parts.append(f"{name}({args.strip()[:80]})")
        if parts:
            return parts
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "//", "/*", "*")):
                continue
            parts.append(line[:100])
            if len(parts) >= 18:
                break
        return parts

    def _prose_parts(self, text: str) -> list[str]:
        # Prefer lines that look like findings / APIs / algorithms.
        scored: list[tuple[int, str]] = []
        for raw in re.split(r"[.\n]+", text):
            line = " ".join(raw.split()).strip()
            if len(line) < 24:
                continue
            score = 0
            low = line.lower()
            for key in (
                "algorithm",
                "architecture",
                "pattern",
                "framework",
                "function",
                "class",
                "pipeline",
                "latency",
                "nav2",
                "ros2",
                "pytorch",
                "chromadb",
                "finding",
                "recommend",
            ):
                if key in low:
                    score += 2
            if re.search(r"\b(?:def|class|struct)\b", line):
                score += 3
            scored.append((score, line[:160]))
        scored.sort(key=lambda x: (-x[0], -len(x[1])))
        out = [s for _, s in scored[:16]]
        if out:
            return out
        return [" ".join(text.split())[:400]]


def ingest_idle_compressed(
    raw_text: str,
    *,
    source: str = "idle_research",
    topic: str = "",
) -> str:
    """Compress ``raw_text`` and upsert into CodebaseVault ``idle_compressed``."""
    text = (raw_text or "").strip()
    if not text:
        return "ERROR: empty idle compression input"
    try:
        from dana.memory.vault import get_codebase_vault

        vault = get_codebase_vault()
        compressor = MemoryCompressor(max_tokens=180)
        payload = compressor.compress_text_to_latent_summary(
            text, embeddings=vault.embeddings
        )
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            return "ERROR: compression produced empty summary"
        meta = {
            "source": str(source or "idle_research"),
            "kind": str(payload.get("kind") or "text"),
            "topic": str(topic or "")[:240],
            "raw_tokens": int(payload.get("raw_tokens") or 0),
            "summary_tokens": int(payload.get("tokens") or 0),
            "ts": float(time.time()),
            "filepath": f"idle_compressed/{source}",
        }
        return vault.upsert_compressed(
            summary,
            metadata=meta,
            embedding=list(payload.get("embedding") or []) or None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: idle compression ingest failed: {exc}"


__all__ = ("MemoryCompressor", "ingest_idle_compressed")
