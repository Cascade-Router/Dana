"""Chroma vector sync — purge / re-embed on filesystem events."""

from __future__ import annotations

import time
from pathlib import Path

from dana.memory.vault import CodebaseVault, FakeEmbeddings, normalize_vault_filepath
from dana.memory.vector_sync import (
    VectorIndexSync,
    is_ephemeral_db_noise,
    should_sync_path,
)


def test_should_sync_path_filters(tmp_path: Path) -> None:
    good = tmp_path / "widget.py"
    good.write_text("x = 1\n", encoding="utf-8")
    assert should_sync_path(good)
    assert not should_sync_path(tmp_path / "foo.bin")
    # Skip-dir segment even if suffix is ok.
    nested = tmp_path / ".git" / "hooks"
    nested.mkdir(parents=True)
    skipped = nested / "x.py"
    skipped.write_text("x = 1\n", encoding="utf-8")
    assert not should_sync_path(skipped)


def test_ephemeral_db_noise_silently_ignored(tmp_path: Path) -> None:
    noise_names = [
        "blackboard.db-journal",
        "chroma.sqlite3-journal",
        "state.db-wal",
        "state.db-shm",
        "scratch.tmp",
        "write.lock",
    ]
    for name in noise_names:
        assert is_ephemeral_db_noise(tmp_path / name), name
        assert not should_sync_path(tmp_path / name), name

    vault = CodebaseVault(tmp_path / "vault", embeddings=FakeEmbeddings(dim=8))
    sync = VectorIndexSync(
        vault=vault,
        watch_roots=[tmp_path],
        debounce_s=0.01,
        max_workers=1,
    )
    before = dict(sync.stats)
    for name in noise_names:
        sync.enqueue("modified", tmp_path / name)
        sync.enqueue("deleted", tmp_path / name)
    sync.flush(timeout_s=5)
    assert sync.stats["events"] == before["events"]
    assert sync.stats["reembedded"] == before["reembedded"]
    assert sync.stats["purged"] == before["purged"]


def test_reembed_and_purge_keep_index_fresh(tmp_path: Path) -> None:
    src = tmp_path / "ws"
    src.mkdir()
    target = src / "alpha.py"
    target.write_text("ALPHA = 'one'\n", encoding="utf-8")

    vault = CodebaseVault(tmp_path / "vault", embeddings=FakeEmbeddings(dim=24))
    assert "chunks" in vault.ingest_local_directory(src)

    hits = vault.search_vault("ALPHA one", n_results=3)
    assert "alpha.py" in hits or "one" in hits.lower()

    # Modify → re-embed replaces stale chunk content.
    target.write_text("ALPHA = 'two_updated_marker'\n", encoding="utf-8")
    msg = vault.reembed_file(target)
    assert msg.startswith("OK: re-embedded")
    hits2 = vault.search_vault("two_updated_marker", n_results=3)
    assert "two_updated_marker" in hits2

    # Delete → purge removes the path from retrieval.
    key = normalize_vault_filepath(target)
    target.unlink()
    purged = vault.purge_filepath(key)
    assert purged.startswith("OK: purged")
    hits3 = vault.search_vault("two_updated_marker", n_results=3)
    assert "empty" in hits3.lower() or "0 matches" in hits3


def test_vector_sync_event_pipeline(tmp_path: Path) -> None:
    src = tmp_path / "ws"
    src.mkdir()
    vault = CodebaseVault(tmp_path / "vault", embeddings=FakeEmbeddings(dim=16))
    seen: list[tuple[str, str]] = []

    sync = VectorIndexSync(
        vault=vault,
        watch_roots=[src],
        debounce_s=0.05,
        max_workers=1,
        on_event=lambda kind, key: seen.append((kind, key)),
    )
    # Drive events without requiring a live Observer thread.
    f = src / "beta.py"
    f.write_text("def beta():\n    return 1\n", encoding="utf-8")
    sync.enqueue("created", f)
    sync.flush(timeout_s=10)
    assert sync.stats["reembedded"] >= 1

    f.write_text("def beta():\n    return 42\n", encoding="utf-8")
    sync.enqueue("modified", f)
    sync.flush(timeout_s=10)

    hits = vault.search_vault("return 42", n_results=2)
    assert "42" in hits

    key = normalize_vault_filepath(f)
    f.unlink()
    sync.enqueue("deleted", key)
    sync.flush(timeout_s=10)
    assert sync.stats["purged"] >= 1
    hits2 = vault.search_vault("return 42", n_results=2)
    assert "empty" in hits2.lower() or "0 matches" in hits2
    assert any(k == "deleted" for k, _ in seen) or sync.stats["events"] >= 3


def test_start_stop_watchdog_or_polling(tmp_path: Path) -> None:
    src = tmp_path / "ws"
    src.mkdir()
    (src / "gamma.py").write_text("GAMMA = 1\n", encoding="utf-8")
    vault = CodebaseVault(tmp_path / "vault", embeddings=FakeEmbeddings(dim=12))
    sync = VectorIndexSync(
        vault=vault,
        watch_roots=[src],
        debounce_s=0.05,
        max_workers=1,
    )
    msg = sync.start()
    assert msg.startswith("OK: vector sync started")
    assert "mode=" in msg
    # Give observer a moment; also poke enqueue for hermetic certainty.
    time.sleep(0.2)
    sync.enqueue("modified", src / "gamma.py")
    sync.flush(timeout_s=10)
    assert sync.stats["reembedded"] >= 1 or sync.stats["events"] >= 1
    assert sync.stop(wait=True).startswith("OK: vector sync stopped")
