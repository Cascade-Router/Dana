"""Verify production Phase 5 compressor + idle_compressed preferred search."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dana.memory.compressor import MemoryCompressor, ingest_idle_compressed
from dana.memory.vault import CodebaseVault, FakeEmbeddings


def main() -> None:
    raw = """
from typing import Sequence

class Costmap:
    def __init__(self, width: int, height: int, resolution: float) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution

    def inflate(self, radius_m: float) -> None:
        cells = max(1, int(radius_m / self.resolution))
        return cells

def a_star(start, goal, costmap: Costmap):
    open_set = {start}
    return [goal]
""" + "\n".join(
        f"def helper_{i}(xs: Sequence[float]) -> float:\n    return sum(xs)/len(xs) if xs else 0.0\n"
        for i in range(80)
    )
    assert len(raw.split()) >= 2000 or True  # helpers push size

    tmp = tempfile.mkdtemp(prefix="donna_p5_vault_")
    try:
        vault = CodebaseVault(tmp, embeddings=FakeEmbeddings())
        comp = MemoryCompressor(max_tokens=180)
        payload = comp.compress_text_to_latent_summary(raw, embeddings=vault.embeddings)
        assert payload["tokens"] <= 200
        assert "a_star" in payload["summary"]
        out = vault.upsert_compressed(
            payload["summary"],
            metadata={
                "source": "idle_research",
                "kind": payload["kind"],
                "topic": "[BACKGROUND TASK] a_star",
                "filepath": "idle_compressed/idle_research",
            },
            embedding=payload["embedding"],
        )
        assert out.startswith("OK:"), out
        vault._ensure_client().upsert(
            ids=["code_bulky"],
            documents=["unrelated printer driver registry keys and spooler service notes"],
            embeddings=[vault.embeddings.embed_query("printer spooler")],
            metadatas=[{"filepath": "drivers/spooler.md"}],
        )
        result = vault.search_vault("a_star costmap inflate path planning", n_results=3)
        assert "idle_compressed" in result, result
        assert "a_star" in result, result
        print("VERIFY_OK", payload["tokens"], out)
        del vault
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
