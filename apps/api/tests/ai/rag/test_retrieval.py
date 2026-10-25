"""BM25 (hand-checked), RRF (exact), retrieval modes, fund filter, citations, local index."""

import math
from pathlib import Path

import numpy as np
import pytest

from app.ai.rag.bm25 import BM25, tokenize
from app.ai.rag.citations import extract_citations, passages
from app.ai.rag.embed import HashingEmbedder
from app.ai.rag.retrieve import RRF_K, Retriever, rrf
from app.ai.rag.store import InMemoryIndex
from tests.ai.rag.helpers import CORPUS, chunk

EMB = HashingEmbedder(256)


def retriever(reranker=None) -> Retriever:
    vecs = EMB.embed_documents([c.text for c in CORPUS])
    return Retriever(InMemoryIndex(CORPUS, vecs), EMB, reranker)


def test_tokenizer_keeps_numbers_and_lowercases() -> None:
    assert tokenize("Exit load: 1% within 365 Days") == [
        "exit",
        "load",
        "1",
        "within",
        "365",
        "days",
    ]


def test_bm25_matches_hand_computation() -> None:
    docs = ["exit load one", "exit exit", "benchmark index"]
    bm = BM25(docs, k1=1.5, b=0.75)
    # N = 3, avgdl = (3 + 2 + 2) / 3; "exit" in 2 docs, "load" in 1.
    avgdl = 7 / 3
    idf_exit = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    idf_load = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))

    def term(f: int, dl: int, idf: float) -> float:
        return idf * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    s = bm.scores("exit load")
    assert s[0] == pytest.approx(term(1, 3, idf_exit) + term(1, 3, idf_load), rel=1e-12)
    assert s[1] == pytest.approx(term(2, 2, idf_exit), rel=1e-12)
    assert s[2] == 0.0
    assert [i for i, _ in bm.top("exit load", 5)] == [0, 1]
    assert bm.top("exit", 5, allowed={1}) == [(1, bm.scores("exit")[1])]


def test_rrf_exact_scores() -> None:
    fused = rrf([[10, 20, 30], [30, 10]])
    expected = {
        10: 1 / (RRF_K + 1) + 1 / (RRF_K + 2),
        20: 1 / (RRF_K + 2),
        30: 1 / (RRF_K + 3) + 1 / (RRF_K + 1),
    }
    assert [d for d, _ in fused] == [10, 30, 20]
    for d, s in fused:
        assert s == pytest.approx(expected[d], rel=1e-15)


def test_dense_and_hybrid_find_the_exit_load_passage() -> None:
    r = retriever()
    exit_chunks = {"sid_alpha:p3:0", "sid_beta:p2:0"}
    for mode in ("dense", "hybrid"):
        hits = r.search("What is the exit load?", k=2, mode=mode)
        assert hits[0].chunk.id in exit_chunks, mode
    hybrid = r.search("exit load", k=2, mode="hybrid")
    assert {h.chunk.id for h in hybrid} == exit_chunks  # BM25 matches both on "exit load"
    assert all(h.bm25_rank is not None for h in hybrid)


def test_fund_filter_restricts_every_list() -> None:
    r = retriever()
    for mode in ("dense", "hybrid"):
        hits = r.search("benchmark index", k=5, mode=mode, fund_code="222")
        assert hits and all(h.chunk.fund_code == "222" for h in hits)
    assert r.search("benchmark", k=3, fund_code="999") == []


def test_rerank_reorders_candidates() -> None:
    class ByLength:
        name = "by-length"

        def scores(self, query: str, texts: list[str]) -> list[float]:
            return [float(len(t)) for t in texts]  # longest passage wins

    r = retriever(ByLength())
    hits = r.search("exit load benchmark riskometer", k=3, mode="hybrid_rerank")
    lengths = [len(h.chunk.text) for h in hits]
    assert lengths == sorted(lengths, reverse=True)
    with pytest.raises(ValueError, match="needs a reranker"):
        retriever().search("x", mode="hybrid_rerank")
    with pytest.raises(ValueError, match="mode must be one of"):
        retriever().search("x", mode="sparse")  # type: ignore[arg-type]


def test_empty_query_and_empty_index() -> None:
    assert retriever().search("   ") == []
    empty = Retriever(InMemoryIndex([], np.zeros((0, 256))), EMB)
    assert empty.search("anything") == []


def test_local_index_round_trip(tmp_path: Path) -> None:
    vecs = EMB.embed_documents([c.text for c in CORPUS])
    InMemoryIndex(CORPUS, vecs).save(tmp_path, EMB.name)
    loaded = InMemoryIndex.load(tmp_path)
    assert loaded.chunks() == CORPUS
    q = EMB.embed_query("riskometer levels")
    assert loaded.search(q, 3) == InMemoryIndex(CORPUS, vecs).search(q, 3)


def test_citations_match_retrieved_passages_and_flag_invented_ones() -> None:
    hits = retriever().search("exit load", k=3, mode="hybrid")
    shown = passages(hits)
    ref0 = shown[0]["ref"]
    answer = f"The exit load is 1% [{ref0}]. Again [{ref0.upper()}]. Also [made_up_doc p.9]."
    cites, invented = extract_citations(answer, shown)
    assert [c.document + " p." + str(c.page) for c in cites] == [ref0]
    assert cites[0].title.startswith("Title of") and cites[0].url.startswith("https://")
    assert invented == ["[made_up_doc p.9]"]
    assert extract_citations("no citations here", shown) == ([], [])


def test_long_passages_are_truncated_for_the_model() -> None:
    long = chunk("doc:p1:0", "word " * 400)
    vec = EMB.embed_documents([long.text])
    hit = Retriever(InMemoryIndex([long], vec), EMB).search("word", k=1)
    p = passages(hit)[0]
    assert len(p["text"]) <= 705 and p["text"].endswith("...")


def test_lenticular_bracket_citations_are_recognised() -> None:
    # gpt-oss cites as 【doc p.N】 (seen in the 2026-10-25 evaluation run).
    shown = [{"ref": "sid_x p.7", "document": "SID X", "page": 7, "url": None, "text": "t"}]
    cites, invented = extract_citations("Exit load 2%【sid_x p.7】 and 【other p.1】.", shown)
    assert [(c.document, c.page) for c in cites] == [("sid_x", 7)]
    assert invented == ["[other p.1]"]
