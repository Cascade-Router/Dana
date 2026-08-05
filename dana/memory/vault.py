"""ChromaDB long-term codebase memory vault (coexists with episodic SQLite store)."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from pathlib import Path
from typing import Any, Protocol

COLLECTION_NAME = "dana_codebase_vault"
IDLE_COLLECTION_NAME = "idle_compressed"
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_SKIP_DIR_NAMES = frozenset({".git", "__pycache__", "node_modules", "build"})
_INGEST_EXTENSIONS = frozenset(
    {".py", ".cpp", ".hpp", ".h", ".c", ".md", ".txt", ".json", ".urdf"}
)

_LOCK = threading.Lock()
_DEFAULT_VAULT: "CodebaseVault | None" = None


def default_vault_dir() -> Path:
    """Project-root ``.dana/vault/`` (injectable via constructor / helpers)."""
    return Path(__file__).resolve().parent.parent.parent / ".dana" / "vault"


class EmbeddingsProtocol(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FakeEmbeddings:
    """Deterministic bag-of-words embeddings for hermetic offline tests."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9_]+", (text or "").lower()):
            vec[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def _build_default_embeddings() -> EmbeddingsProtocol:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=_DEFAULT_MODEL)


def _chunk_id(filepath: str, index: int) -> str:
    digest = hashlib.sha1(f"{filepath}::{index}".encode("utf-8")).hexdigest()[:24]
    return f"chunk_{digest}"


def normalize_vault_filepath(path: str | Path) -> str:
    """Canonical filepath key stored in Chroma metadata (absolute posix)."""
    return Path(path).expanduser().resolve().as_posix()


def _iter_ingest_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in _INGEST_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files)


def _emit_executing(tool: str) -> None:
    try:
        from dana.ui.status_bus import emit_state_change

        emit_state_change("executing", tool=tool)
    except Exception:  # noqa: BLE001
        pass


class CodebaseVault:
    """Persistent Chroma collection for local codebase chunks.

    Chroma client + HuggingFace embeddings are constructed on first
    ingest/search only — import and construction stay cheap.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        *,
        embeddings: EmbeddingsProtocol | None = None,
    ) -> None:
        self.persist_directory = Path(
            persist_directory if persist_directory is not None else default_vault_dir()
        )
        self._embeddings = embeddings
        self._client: Any = None
        self._collection: Any = None
        self._idle_collection: Any = None
        self._client_lock = threading.RLock()

    def _ensure_client(self) -> Any:
        """Open PersistentClient + primary collection on first use (not at import/ctor)."""
        if self._collection is not None:
            return self._collection
        with self._client_lock:
            if self._collection is not None:
                return self._collection
            import chromadb

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection

    def _ensure_idle_collection(self) -> Any:
        """Open the dense ``idle_compressed`` collection (Phase 5)."""
        if self._idle_collection is not None:
            return self._idle_collection
        with self._client_lock:
            if self._idle_collection is not None:
                return self._idle_collection
            self._ensure_client()
            self._idle_collection = self._client.get_or_create_collection(
                name=IDLE_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return self._idle_collection

    @property
    def embeddings(self) -> EmbeddingsProtocol:
        if self._embeddings is None:
            self._embeddings = _build_default_embeddings()
        return self._embeddings

    def ingest_local_directory(self, path: str | Path) -> str:
        """Walk ``path``, chunk source files, upsert into ``dana_codebase_vault``."""
        _emit_executing("ingest_local_directory")
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"ERROR: ingest path is not a directory: {root}"

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for file_path in _iter_ingest_files(root):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not (text or "").strip():
                continue
            rel = normalize_vault_filepath(file_path)
            chunks = splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                if not (chunk or "").strip():
                    continue
                ids.append(_chunk_id(rel, idx))
                documents.append(chunk)
                metadatas.append({"filepath": rel})

        if not documents:
            return "OK: ingested 0 chunks into dana_codebase_vault"

        collection = self._ensure_client()
        embeddings = self.embeddings.embed_documents(documents)
        # Chroma upsert in batches to avoid oversized payloads.
        batch = 64
        for start in range(0, len(documents), batch):
            end = start + batch
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        return f"OK: ingested {len(documents)} chunks into dana_codebase_vault"

    def purge_filepath(self, filepath: str | Path) -> str:
        """Delete all vault chunks whose metadata ``filepath`` matches ``filepath``."""
        key = normalize_vault_filepath(filepath)
        collection = self._ensure_client()
        try:
            # Prefer metadata filter; fall back to id scan when where is unsupported.
            collection.delete(where={"filepath": key})
        except Exception:  # noqa: BLE001
            try:
                existing = collection.get(include=["metadatas"])
                ids = list(existing.get("ids") or [])
                metas = list(existing.get("metadatas") or [])
                drop = [
                    cid
                    for cid, meta in zip(ids, metas)
                    if str((meta or {}).get("filepath") or "") == key
                ]
                if drop:
                    collection.delete(ids=drop)
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: purge failed for {key}: {exc}"
        return f"OK: purged vault chunks for {key}"

    def reembed_file(self, filepath: str | Path) -> str:
        """Strip prior embeddings for ``filepath``, then re-chunk and upsert."""
        path = Path(filepath).expanduser().resolve()
        key = normalize_vault_filepath(path)
        purge_msg = self.purge_filepath(key)
        if not path.is_file():
            return f"{purge_msg}; skip re-embed (file missing)"
        if path.suffix.lower() not in _INGEST_EXTENSIONS:
            return f"{purge_msg}; skip re-embed (unsupported suffix {path.suffix!r})"
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            return f"{purge_msg}; skip re-embed (path in skipped directory)"

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return f"ERROR: re-embed read failed for {key}: {exc}"
        if not (text or "").strip():
            return f"{purge_msg}; skip re-embed (empty file)"

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = [c for c in splitter.split_text(text) if (c or "").strip()]
        if not chunks:
            return f"{purge_msg}; skip re-embed (0 chunks)"

        ids = [_chunk_id(key, idx) for idx in range(len(chunks))]
        metadatas = [{"filepath": key} for _ in chunks]
        embeddings = self.embeddings.embed_documents(chunks)
        collection = self._ensure_client()
        batch = 64
        for start in range(0, len(chunks), batch):
            end = start + batch
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        return f"OK: re-embedded {len(chunks)} chunks for {key}"

    def upsert_compressed(
        self,
        summary: str,
        *,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Upsert one dense summary into ``idle_compressed`` (Phase 5)."""
        text = (summary or "").strip()
        if not text:
            return "ERROR: empty compressed summary"
        meta = dict(metadata or {})
        meta.setdefault("filepath", "idle_compressed")
        meta.setdefault("source", "idle_compressed")
        # Chroma metadata values must be str/int/float/bool.
        clean_meta: dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean_meta[str(key)] = value
            else:
                clean_meta[str(key)] = str(value)
        emb = list(embedding) if embedding else self.embeddings.embed_documents([text])[0]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
        cid = (doc_id or f"idle_{digest}").strip() or f"idle_{digest}"
        collection = self._ensure_idle_collection()
        collection.upsert(
            ids=[cid],
            documents=[text],
            embeddings=[emb],
            metadatas=[clean_meta],
        )
        try:
            self.compact_collections()
        except Exception:  # noqa: BLE001
            pass
        return f"OK: upserted compressed summary into {IDLE_COLLECTION_NAME} id={cid}"

    def compact_collections(self) -> str:
        """Best-effort Chroma flush + sqlite VACUUM after background ingestion.

        Chroma 1.x has no public ``compact()`` API; touching collection counts
        settles segment indexes, then VACUUM reclaims space on ``chroma.sqlite3``.
        """
        try:
            self._ensure_client()
            for getter in (self._ensure_client, self._ensure_idle_collection):
                try:
                    coll = getter()
                    if coll is not None:
                        _ = int(coll.count() or 0)
                except Exception:  # noqa: BLE001
                    continue
            db = Path(self.persist_directory) / "chroma.sqlite3"
            if db.is_file():
                import sqlite3

                con = sqlite3.connect(str(db))
                try:
                    con.execute("VACUUM")
                    con.commit()
                finally:
                    con.close()
            return "OK: vault collections compacted"
        except Exception as exc:  # noqa: BLE001
            return f"WARNING: vault compact skipped ({exc})"

    def search_vault(self, query: str, n_results: int = 5) -> str:
        """Return formatted top chunks; prefer dense ``idle_compressed`` hits first."""
        _emit_executing("search_vault")
        q = (query or "").strip()
        if not q:
            return "ERROR: missing query"
        n = max(1, int(n_results or 5))
        emb = self.embeddings.embed_query(q)
        lines: list[str] = []

        # Phase 5 — prefer high-density idle summaries for fast TTFT context.
        try:
            idle = self._ensure_idle_collection()
            idle_count = int(idle.count() or 0)
        except Exception:  # noqa: BLE001
            idle = None
            idle_count = 0
        if idle is not None and idle_count > 0:
            idle_n = min(n, idle_count)
            idle_result = idle.query(
                query_embeddings=[emb],
                n_results=idle_n,
                include=["documents", "metadatas"],
            )
            idle_docs = (idle_result.get("documents") or [[]])[0]
            idle_metas = (idle_result.get("metadatas") or [[]])[0]
            for doc, meta in zip(idle_docs, idle_metas):
                body = (doc or "").strip()
                if not body:
                    continue
                m = meta or {}
                topic = m.get("topic") or m.get("filepath") or IDLE_COLLECTION_NAME
                lines.append(f"[idle_compressed] {topic}\n{body}")
                if len(lines) >= n:
                    break

        remaining = n - len(lines)
        if remaining > 0:
            collection = self._ensure_client()
            count = int(collection.count() or 0)
            if count > 0:
                code_n = min(remaining, count)
                result = collection.query(
                    query_embeddings=[emb],
                    n_results=code_n,
                    include=["documents", "metadatas"],
                )
                docs = (result.get("documents") or [[]])[0]
                metas = (result.get("metadatas") or [[]])[0]
                for doc, meta in zip(docs, metas):
                    body = (doc or "").strip()
                    if not body:
                        continue
                    fp = (meta or {}).get("filepath") or "(unknown)"
                    lines.append(f"[codebase] {fp}\n{body}")

        if not lines:
            return "OK: vault empty (0 matches)"
        numbered = [f"[{i}] {block}" for i, block in enumerate(lines[:n], start=1)]
        return "OK: vault search results\n\n" + "\n\n".join(numbered)


def get_codebase_vault(
    persist_directory: str | Path | None = None,
    *,
    embeddings: EmbeddingsProtocol | None = None,
) -> CodebaseVault:
    """Process-wide default vault, or a fresh instance when path/embeddings overridden."""
    global _DEFAULT_VAULT
    if persist_directory is not None or embeddings is not None:
        return CodebaseVault(persist_directory, embeddings=embeddings)
    with _LOCK:
        if _DEFAULT_VAULT is None:
            _DEFAULT_VAULT = CodebaseVault()
        return _DEFAULT_VAULT


def ingest_local_directory(
    path: str | Path,
    *,
    persist_directory: str | Path | None = None,
    embeddings: EmbeddingsProtocol | None = None,
) -> str:
    return get_codebase_vault(
        persist_directory, embeddings=embeddings
    ).ingest_local_directory(path)


def search_vault(
    query: str,
    n_results: int = 5,
    *,
    persist_directory: str | Path | None = None,
    embeddings: EmbeddingsProtocol | None = None,
) -> str:
    return get_codebase_vault(
        persist_directory, embeddings=embeddings
    ).search_vault(query, n_results=n_results)


def purge_filepath(
    filepath: str | Path,
    *,
    persist_directory: str | Path | None = None,
    embeddings: EmbeddingsProtocol | None = None,
) -> str:
    return get_codebase_vault(
        persist_directory, embeddings=embeddings
    ).purge_filepath(filepath)


def reembed_file(
    filepath: str | Path,
    *,
    persist_directory: str | Path | None = None,
    embeddings: EmbeddingsProtocol | None = None,
) -> str:
    return get_codebase_vault(
        persist_directory, embeddings=embeddings
    ).reembed_file(filepath)
