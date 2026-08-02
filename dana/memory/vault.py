"""ChromaDB long-term codebase memory vault (coexists with episodic SQLite store)."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from pathlib import Path
from typing import Any, Protocol

COLLECTION_NAME = "dana_codebase_vault"
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
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        """Open PersistentClient + collection on first use (not at import/ctor)."""
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
            rel = str(file_path.as_posix())
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

    def search_vault(self, query: str, n_results: int = 5) -> str:
        """Return formatted top chunks with filepath metadata."""
        _emit_executing("search_vault")
        q = (query or "").strip()
        if not q:
            return "ERROR: missing query"
        n = max(1, int(n_results or 5))
        collection = self._ensure_client()
        count = int(collection.count() or 0)
        if count == 0:
            return "OK: vault empty (0 matches)"
        n = min(n, count)
        emb = self.embeddings.embed_query(q)
        result = collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "metadatas"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        if not docs:
            return "OK: no matching chunks"
        lines: list[str] = []
        for i, doc in enumerate(docs, start=1):
            meta = metas[i - 1] if i - 1 < len(metas) else {}
            fp = (meta or {}).get("filepath") or "(unknown)"
            body = (doc or "").strip()
            lines.append(f"[{i}] {fp}\n{body}")
        return "OK: vault search results\n\n" + "\n\n".join(lines)


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
