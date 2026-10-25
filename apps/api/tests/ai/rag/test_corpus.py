"""Manifest, PDF verification and page-preserving chunking."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ai.rag.corpus import (
    CHUNK_CHARS,
    DocEntry,
    _clean,
    chunk_document,
    load_manifest,
    pdf_pages,
    split_page,
    verify,
)
from app.core.config import REPO_ROOT
from tests.ai.rag.helpers import make_pdf

ENTRY = DocEntry(
    id="sid_test",
    title="Test Fund SID",
    doc_type="sid",
    fund_code="123",
    url="https://example.invalid/x.pdf",
    expect="Test Fund",
)


def test_committed_manifest_is_valid_and_has_no_duplicates() -> None:
    m = load_manifest(REPO_ROOT / "data" / "docs_manifest.json")
    assert len(m.documents) >= 10
    assert all(d.url.startswith("https://") for d in m.documents)
    fund_docs = [d for d in m.documents if d.fund_code]
    assert all(d.doc_type in {"sid", "kim", "factsheet", "scheme_summary"} for d in fund_docs)


def test_manifest_rejects_duplicates(tmp_path: Path) -> None:
    doc = ENTRY.model_dump()
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"note": "x", "documents": [doc, doc]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate document ids"):
        load_manifest(p)
    with pytest.raises(ValidationError):
        DocEntry.model_validate(doc | {"url": "http://insecure.example/x.pdf"})


def test_pdf_text_per_page() -> None:
    data = make_pdf(["Test Fund scheme information", "Exit load is 1%"])
    assert pdf_pages(data) == ["Test Fund scheme information", "Exit load is 1%"]


def test_verify_accepts_the_right_document_and_rejects_others() -> None:
    long_page = "Test Fund " + "word " * 200
    ok, reason, pages = verify(ENTRY, make_pdf([long_page, "second page"]))
    assert ok and reason == "ok" and len(pages) == 2
    assert verify(ENTRY, b"<html>not found</html>") == (
        False,
        "not a PDF (the server returned something else)",
        [],
    )
    wrong = verify(ENTRY, make_pdf(["Another scheme " + "word " * 200]))
    assert not wrong[0] and "expected text 'Test Fund' not found" in wrong[1]
    short = verify(ENTRY, make_pdf(["Test Fund"]))
    assert not short[0] and "scanned" in short[1]


def test_clean_removes_control_characters() -> None:
    assert _clean("a\x00b\x07c   d‑e\n\n\n\nf") == "abc d-e\n\nf"


def test_split_page_windows_overlap_and_cover_everything() -> None:
    sentences = [f"Sentence number {i} says something about the scheme." for i in range(60)]
    text = " ".join(sentences)
    pieces = split_page(text)
    assert all(len(p) <= CHUNK_CHARS for p in pieces)
    assert len(pieces) > 3
    # Cuts fall at sentence ends: every piece but the last ends with a full stop.
    assert all(p.endswith(".") for p in pieces[:-1])
    # Overlap: each piece starts inside the previous one.
    for a, b in zip(pieces, pieces[1:], strict=False):
        assert b[:40] in a
    # Nothing is lost: every sentence appears in some piece.
    assert all(any(s in p for p in pieces) for s in sentences)


def test_split_page_handles_text_without_breaks_and_bad_parameters() -> None:
    blob = "x" * 2000
    pieces = split_page(blob, size=900, overlap=100)
    assert [len(p) for p in pieces] == [900, 900, 400]
    with pytest.raises(ValueError):
        split_page("abc", size=100, overlap=100)


def test_chunks_keep_page_numbers_and_metadata() -> None:
    pages = ["A" * 100, "", "tiny", "Exit load details. " * 80]
    chunks = chunk_document(ENTRY, pages)
    assert {c.page for c in chunks} == {1, 4}  # empty and too-short pages dropped
    assert chunks[0].id == "sid_test:p1:0" and chunks[0].fund_code == "123"
    assert all(c.doc_id == "sid_test" and c.url == ENTRY.url for c in chunks)
    assert [c.id for c in chunks if c.page == 4][:2] == ["sid_test:p4:0", "sid_test:p4:1"]
