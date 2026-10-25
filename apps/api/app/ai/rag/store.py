"""Where chunks and their embeddings live.

- InMemoryIndex: numpy matrix of normalised vectors; cosine = dot product. Used in tests, for
  local files (data/docs/index/, gitignored) and as the loaded form of the Postgres store.
- PgVectorStore: tables `rag_documents` and `rag_chunks` (embedding vector(384)) in Postgres;
  dense search runs in SQL with pgvector's cosine distance operator `<=>`.
Both answer `search(query_vector, k, fund_code)` and `chunks()`, so the retriever does not
care which one it has.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.ai.rag.corpus import Chunk
from app.db.base import NAMING_CONVENTION

EMBED_DIM = 384


class DenseIndex(Protocol):
    def chunks(self) -> list[Chunk]: ...
    def search(
        self, query: np.ndarray, k: int, fund_code: str | None = None
    ) -> list[tuple[int, float]]: ...


class InMemoryIndex:
    """Dense search over normalised vectors; returns (chunk index, cosine similarity)."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._chunks = chunks
        self._m = np.asarray(vectors, dtype=np.float32)

    def chunks(self) -> list[Chunk]:
        return self._chunks

    def search(
        self, query: np.ndarray, k: int, fund_code: str | None = None
    ) -> list[tuple[int, float]]:
        if not self._chunks:
            return []
        sims = self._m @ np.asarray(query, dtype=np.float32)
        idx = np.arange(len(self._chunks))
        if fund_code is not None:
            idx = np.array([i for i, c in enumerate(self._chunks) if c.fund_code == fund_code])
            if idx.size == 0:
                return []
        order = idx[np.argsort(-sims[idx], kind="stable")][:k]
        return [(int(i), float(sims[i])) for i in order]

    # --- local files ---

    def save(self, directory: Path, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self._m)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        meta = {"model": model, "dim": int(self._m.shape[1]) if len(self._m) else 0}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "InMemoryIndex":
        lines = (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk(**json.loads(line)) for line in lines if line.strip()]
        return cls(chunks, np.load(directory / "embeddings.npy"))


# --- Postgres (pgvector) ---


class RagBase(DeclarativeBase):
    """Separate metadata: the Vector type exists only on Postgres, and SQLite-based unit tests
    create app.db.base.Base tables only."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class RagDocument(RagBase):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    doc_type: Mapped[str] = mapped_column(String(40))
    fund_code: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    url: Mapped[str] = mapped_column(String(600))
    sha256: Mapped[str] = mapped_column(String(64))
    pages: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)


class RagChunk(RagBase):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        # Approximate nearest-neighbour search on cosine distance (pgvector HNSW).
        Index(
            "ix_rag_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True
    )
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    fund_code: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    model: Mapped[str] = mapped_column(String(80))


class PgVectorStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._cache: list[Chunk] | None = None

    def replace_document(
        self, doc: dict[str, object], chunks: list[Chunk], vectors: np.ndarray, model: str
    ) -> None:
        """Insert or replace one document and all its chunks in one transaction."""
        with Session(self._engine) as s, s.begin():
            s.execute(delete(RagChunk).where(RagChunk.doc_id == doc["id"]))
            s.execute(delete(RagDocument).where(RagDocument.id == doc["id"]))
            s.add(RagDocument(**doc, ingested_at=datetime.now(UTC)))
            s.flush()
            rows = [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "text": c.text,
                    "fund_code": c.fund_code,
                    "embedding": v.tolist(),
                    "model": model,
                }
                for c, v in zip(chunks, vectors, strict=True)
            ]
            for i in range(0, len(rows), 500):
                s.execute(insert(RagChunk), rows[i : i + 500])
        self._cache = None

    def chunks(self) -> list[Chunk]:
        if self._cache is None:
            with Session(self._engine) as s:
                rows = s.execute(
                    select(RagChunk, RagDocument)
                    .join(RagDocument, RagDocument.id == RagChunk.doc_id)
                    .order_by(RagChunk.doc_id, RagChunk.page, RagChunk.id)
                ).all()
            self._cache = [
                Chunk(
                    id=c.id,
                    doc_id=c.doc_id,
                    title=d.title,
                    doc_type=d.doc_type,
                    fund_code=c.fund_code,
                    url=d.url,
                    page=c.page,
                    text=c.text,
                )
                for c, d in rows
            ]
        return self._cache

    def search(
        self, query: np.ndarray, k: int, fund_code: str | None = None
    ) -> list[tuple[int, float]]:
        """Cosine similarity = 1 - (embedding <=> query), computed by pgvector in SQL."""
        position = {c.id: i for i, c in enumerate(self.chunks())}
        dist = RagChunk.embedding.cosine_distance(np.asarray(query, dtype=float).tolist())
        stmt = select(RagChunk.id, dist.label("d")).order_by(dist).limit(k)
        if fund_code is not None:
            stmt = stmt.where(RagChunk.fund_code == fund_code)
        with Session(self._engine) as s:
            rows = s.execute(stmt).all()
        return [(position[r.id], 1.0 - float(r.d)) for r in rows if r.id in position]

    def counts(self) -> tuple[int, int]:
        with Session(self._engine) as s:
            docs = s.scalar(select(func.count()).select_from(RagDocument)) or 0
            chunks = s.scalar(select(func.count()).select_from(RagChunk)) or 0
        return docs, chunks
