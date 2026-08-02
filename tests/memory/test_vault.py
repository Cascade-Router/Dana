"""Hermetic Chroma codebase vault tests (temp dir + FakeEmbeddings)."""

from __future__ import annotations

from pathlib import Path

from dana.memory.vault import CodebaseVault, FakeEmbeddings, ingest_local_directory, search_vault


def test_ingest_and_search_vault(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "widget.py").write_text(
        "def build_purple_widget():\n"
        "    '''Assemble the purple widget assembly for CAMGRASPER.'''\n"
        "    return 'purple-widget-ready'\n",
        encoding="utf-8",
    )
    (src / "notes.md").write_text(
        "# Notes\nThe purple widget assembly lives in widget.py.\n",
        encoding="utf-8",
    )
    # Ignored trees / extensions must not break ingest.
    junk = src / "__pycache__"
    junk.mkdir()
    (junk / "skip.pyc").write_bytes(b"\x00\x01")
    (src / "skip.bin").write_bytes(b"\x00\x01")

    vault_dir = tmp_path / "vault"
    embeddings = FakeEmbeddings(dim=32)
    vault = CodebaseVault(vault_dir, embeddings=embeddings)

    msg = vault.ingest_local_directory(src)
    assert msg.startswith("OK: ingested ")
    assert "chunks into dana_codebase_vault" in msg
    assert "0 chunks" not in msg

    hits = vault.search_vault("purple widget assembly", n_results=3)
    assert hits.startswith("OK: vault search results")
    assert "widget.py" in hits or "purple" in hits.lower()

    # Module helpers also accept injectable path + FakeEmbeddings.
    msg2 = ingest_local_directory(
        src, persist_directory=tmp_path / "vault_b", embeddings=FakeEmbeddings()
    )
    assert "chunks into dana_codebase_vault" in msg2
    empty = search_vault(
        "nothing here yet",
        persist_directory=tmp_path / "vault_empty",
        embeddings=FakeEmbeddings(),
    )
    assert "empty" in empty.lower() or "0 matches" in empty


def test_codebase_vault_skips_missing_dir(tmp_path: Path) -> None:
    vault = CodebaseVault(
        tmp_path / "vault2", embeddings=FakeEmbeddings(dim=16)
    )
    out = vault.ingest_local_directory(tmp_path / "nope")
    assert out.startswith("ERROR:")
