"""Retrieval: dense, hybrid (BM25 + dense, reciprocal rank fusion) or hybrid + reranker.

    dense          top-k by cosine similarity of bge-small embeddings
    hybrid         RRF(dense top-C, BM25 top-C), C = candidates
                   RRF score(d) = sum_lists 1 / (rrf_k + rank_list(d)),  rank from 1, rrf_k = 60
    hybrid_rerank  hybrid top-C re-scored by a cross-encoder (query, passage) -> top-k

Each mode is selectable per call, so the evaluation compares them on the same queries. A fund
filter restricts every list to one fund's documents (fund_code = AMFI scheme code).

Why RRF: dense and BM25 scores are on different scales; RRF uses ranks only, needs no score
normalisation and is robust (Cormack, Clarke and Buettcher 2009). rrf_k = 60 is their value.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.ai.rag.bm25 import BM25
from app.ai.rag.corpus import Chunk
from app.ai.rag.embed import Embedder
from app.ai.rag.store import DenseIndex

Mode = Literal["dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("dense", "hybrid", "hybrid_rerank")
RRF_K = 60
DEFAULT_CANDIDATES = 20


class Reranker(Protocol):
    name: str

    def scores(self, query: str, passages: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder (default Xenova/ms-marco-MiniLM-L-6-v2, ~80 MB, ONNX) via fastembed."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model_name=model, cache_dir=cache_dir)

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages)]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # mode-specific: cosine, RRF score, or reranker score
    dense_rank: int | None
    bm25_rank: int | None


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids: score = sum 1/(k + rank). Ties broken by first appearance."""
    scores: dict[int, float] = {}
    order: dict[int, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            order.setdefault(doc, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


class Retriever:
    def __init__(
        self, index: DenseIndex, embedder: Embedder, reranker: Reranker | None = None
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self._chunks = index.chunks()
        self._bm25 = BM25([c.text for c in self._chunks])
        self._by_fund: dict[str | None, set[int]] = {}
        for i, c in enumerate(self._chunks):
            self._by_fund.setdefault(c.fund_code, set()).add(i)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: Mode = "hybrid",
        fund_code: str | None = None,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        if not self._chunks or not query.strip():
            return []
        dense = self.index.search(self.embedder.embed_query(query), candidates, fund_code)
        dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
        if mode == "dense":
            return [Hit(self._chunks[i], s, dense_rank[i], None) for i, s in dense[:k]]
        allowed = self._by_fund.get(fund_code, set()) if fund_code is not None else None
        lexical = self._bm25.top(query, candidates, allowed)
        bm25_rank = {i: r for r, (i, _) in enumerate(lexical, start=1)}
        fused = rrf([[i for i, _ in dense], [i for i, _ in lexical]])[:candidates]
        if mode == "hybrid":
            return [
                Hit(self._chunks[i], s, dense_rank.get(i), bm25_rank.get(i)) for i, s in fused[:k]
            ]
        assert self.reranker is not None
        texts = [self._chunks[i].text for i, _ in fused]
        rescored = sorted(
            zip((i for i, _ in fused), self.reranker.scores(query, texts), strict=True),
            key=lambda x: -x[1],
        )
        return [
            Hit(self._chunks[i], s, dense_rank.get(i), bm25_rank.get(i)) for i, s in rescored[:k]
        ]
