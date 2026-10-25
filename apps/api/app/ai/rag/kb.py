"""The knowledge base used at request time: loaded once per process, then shared.

Backend: Postgres (pgvector tables written by scripts/ingest_docs.py --db) when DATABASE_URL is
set, otherwise the local index files in data/docs/index/. The query embedder (bge-small, ~100 MB
of RAM) and, if RAG_MODE = hybrid_rerank, the cross-encoder load on first use, not at startup.
Nothing here calls the network (the model files are read from the local cache).
"""

import threading
from pathlib import Path

from app.ai.rag.embed import DEFAULT_MODEL, get_fastembed
from app.ai.rag.retrieve import FastEmbedReranker, Retriever
from app.ai.rag.store import DenseIndex, InMemoryIndex
from app.core.config import REPO_ROOT, Settings


class KnowledgeBaseUnavailable(RuntimeError):
    pass


def model_cache_dir(settings: Settings) -> Path:
    return settings.rag_model_dir or (REPO_ROOT / ".cache" / "models")


def load_retriever(settings: Settings) -> Retriever:
    index: DenseIndex
    if settings.database_url is not None and settings.rag_backend in ("auto", "postgres"):
        from app.ai.rag.store import PgVectorStore
        from app.db.session import get_engine

        index = PgVectorStore(get_engine())
    else:
        directory = settings.rag_index_dir or (REPO_ROOT / "data" / "docs" / "index")
        if not (directory / "chunks.jsonl").is_file():
            raise KnowledgeBaseUnavailable(
                "the document index is not built; run scripts/ingest_docs.py"
            )
        index = InMemoryIndex.load(directory)
    if not index.chunks():
        raise KnowledgeBaseUnavailable("the knowledge base is empty; run scripts/ingest_docs.py")
    cache = str(model_cache_dir(settings))
    reranker = FastEmbedReranker(cache_dir=cache) if settings.rag_mode == "hybrid_rerank" else None
    return Retriever(index, get_fastembed(DEFAULT_MODEL, cache), reranker)


class KnowledgeBase:
    """Lazy, thread-safe holder (like MarketDataStore). A failed load is retried next time."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._settings.rag_mode

    def get(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    try:
                        self._retriever = load_retriever(self._settings)
                    except KnowledgeBaseUnavailable:
                        raise
                    except Exception as exc:  # DB down, model missing: say so, log upstream
                        raise KnowledgeBaseUnavailable(
                            f"the knowledge base could not be loaded ({type(exc).__name__})"
                        ) from exc
        return self._retriever

    def set(self, retriever: Retriever) -> None:
        self._retriever = retriever
