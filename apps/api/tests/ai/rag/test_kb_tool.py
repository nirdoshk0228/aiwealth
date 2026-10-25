"""search_knowledge_base tool, citations through the agent loop, and pgvector on Postgres."""

import json

import numpy as np
import pytest
from sqlalchemy import Engine

from app.ai.agent.loop import run_agent
from app.ai.agent.tools import ToolContext, execute
from app.ai.llm.fake import FakeLLMClient, call, reply
from app.ai.rag.embed import HashingEmbedder
from app.ai.rag.kb import KnowledgeBase, KnowledgeBaseUnavailable
from app.ai.rag.retrieve import Retriever
from app.ai.rag.store import InMemoryIndex, PgVectorStore, RagBase
from app.core.config import Settings
from app.schemas.chat import ChatMessage
from tests.ai.rag.helpers import CORPUS

EMB = HashingEmbedder(384)


def kb() -> Retriever:
    return Retriever(InMemoryIndex(CORPUS, EMB.embed_documents([c.text for c in CORPUS])), EMB)


def ctx(knowledge=kb) -> ToolContext:
    def no_market():
        raise AssertionError("the knowledge-base tool must not need market data")

    return ToolContext(market=no_market, portfolio=None, knowledge=knowledge)


def test_search_returns_refs_and_records_what_the_model_saw() -> None:
    c = ctx()
    out = execute("search_knowledge_base", json.dumps({"query": "exit load"}), c)
    refs = [p["ref"] for p in out["passages"]]
    assert refs[0] in {"sid_alpha p.3", "sid_beta p.2"}
    assert "chunk_id" not in out["passages"][0] and "url" not in out["passages"][0]  # compact
    assert [p["ref"] for p in c.retrieved] == refs  # full records kept for citations
    assert "not found in the documents" in out["instructions"]


def test_fund_name_restricts_to_that_funds_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai.agent import tools

    monkeypatch.setattr(tools, "_fund_code", lambda name: "222" if "beta" in name.lower() else None)
    out = execute(
        "search_knowledge_base", json.dumps({"query": "benchmark", "fund": "Beta"}), ctx()
    )
    assert out["passages"] and all(p["ref"].startswith("sid_beta") for p in out["passages"])
    unknown = execute(
        "search_knowledge_base", json.dumps({"query": "benchmark", "fund": "Nope"}), ctx()
    )
    assert "no universe fund matched" in unknown["note"]


def test_fund_name_resolution_uses_the_universe() -> None:
    from app.ai.agent.tools import _fund_code

    assert _fund_code("Axis Liquid Fund") == "120389"
    assert _fund_code("parag parikh flexi cap") == "122639"
    assert _fund_code("Imaginary Moon Fund") is None
    # Spacing variants of the same scheme name (the universe says "Kotak Flexi Cap Fund").
    assert _fund_code("Kotak Flexicap Fund") == _fund_code("Kotak Flexi Cap") == "120166"
    assert _fund_code("HDFC Midcap") == "118989"


def test_unavailable_knowledge_base_is_a_tool_error() -> None:
    def broken():
        raise KnowledgeBaseUnavailable("the document index is not built")

    assert (
        "not built"
        in execute("search_knowledge_base", '{"query": "exit load"}', ctx(broken))["error"]
    )
    none = ToolContext(market=lambda: None, portfolio=None, knowledge=None)  # type: ignore[arg-type,return-value]
    assert "not available" in execute("search_knowledge_base", '{"query": "abc"}', none)["error"]


def test_citations_flow_through_the_agent_loop() -> None:
    def answer(messages, tools):
        passages = json.loads(messages[-1].content)["passages"]
        return reply(f"The exit load is 1% [{passages[0]['ref']}]; see also [ghost_doc p.1].")

    fake = FakeLLMClient([call("search_knowledge_base", '{"query": "exit load"}'), answer])
    events = list(run_agent(fake, [ChatMessage(role="user", content="Exit load?")], ctx()))
    ans = next(e for e in events if e.type == "answer")
    assert len(ans.citations) == 1 and ans.citations[0].page in (2, 3)
    assert ans.flags == ["citation not among the retrieved passages: [ghost_doc p.1]"]


def test_local_backend_missing_index_is_reported(tmp_path) -> None:
    s = Settings(database_url=None, rag_index_dir=tmp_path, rag_backend="local")  # type: ignore[arg-type]
    with pytest.raises(KnowledgeBaseUnavailable, match="not built"):
        KnowledgeBase(s).get()


@pytest.mark.postgres
def test_pgvector_store_round_trip(pg_engine: Engine) -> None:
    from sqlalchemy import text

    with pg_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    RagBase.metadata.create_all(pg_engine)
    store = PgVectorStore(pg_engine)
    vecs = EMB.embed_documents([c.text for c in CORPUS])
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(CORPUS):
        by_doc.setdefault(c.doc_id, []).append(i)
    for doc, idx in by_doc.items():
        first = CORPUS[idx[0]]
        store.replace_document(
            {
                "id": doc,
                "title": first.title,
                "doc_type": first.doc_type,
                "fund_code": first.fund_code,
                "url": first.url,
                "sha256": "0" * 64,
                "pages": 9,
                "meta": {},
            },
            [CORPUS[i] for i in idx],
            vecs[idx],
            EMB.name,
        )
    assert store.counts() == (3, len(CORPUS))
    assert sorted(c.id for c in store.chunks()) == sorted(c.id for c in CORPUS)
    mem = InMemoryIndex(
        store.chunks(), np.array([vecs[[c.id for c in CORPUS].index(x.id)] for x in store.chunks()])
    )
    q = EMB.embed_query("benchmark index")
    pg_hits = store.search(q, 4)
    mem_hits = mem.search(q, 4)
    # Same similarity at every rank (pgvector stores float32); ids must agree wherever the
    # scores differ, while tied scores (here several exact 0.0) may come in any order.
    for (_, a), (_, b) in zip(pg_hits, mem_hits, strict=True):
        assert a == pytest.approx(b, abs=1e-5)
    distinct = [i for (i, sc) in mem_hits if sum(abs(sc - o) < 1e-6 for _, o in mem_hits) == 1]
    assert distinct and set(distinct) <= {i for i, _ in pg_hits}
    assert all(store.chunks()[i].fund_code == "222" for i, _ in store.search(q, 5, "222"))
    # Replacing a document removes its old chunks.
    store.replace_document(
        {
            "id": "sid_beta",
            "title": "t",
            "doc_type": "sid",
            "fund_code": "222",
            "url": "u",
            "sha256": "1" * 64,
            "pages": 1,
            "meta": {},
        },
        [CORPUS[2]],
        vecs[[2]],
        EMB.name,
    )
    assert store.counts() == (3, len(CORPUS) - 1)
