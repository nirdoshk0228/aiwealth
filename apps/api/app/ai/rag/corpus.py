"""The document corpus: manifest, download and verification, page text, chunking.

Run offline by scripts/ingest_docs.py, never on a request path. Every manifest entry is kept
only if the downloaded file is a PDF whose first pages contain the entry's `expect` phrase, so a
wrong or moved URL cannot slip a different document into the knowledge base.

Chunking keeps page numbers exact for citations: each page is split on its own (a chunk never
spans two pages) into windows of about CHUNK_CHARS characters, cut at sentence or line ends
where possible, with CHUNK_OVERLAP characters of overlap so a sentence cut at a boundary still
appears whole in one of the two neighbours.
"""

import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80  # drop near-empty fragments (page numbers, headers only)
VERIFY_PAGES = 3

DocType = Literal["regulation", "investor_education", "sid", "kim", "factsheet", "scheme_summary"]


class DocEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    doc_type: DocType
    fund_code: str | None = Field(default=None, pattern=r"^\d+$")
    url: str = Field(pattern=r"^https://")
    expect: str = Field(min_length=3, description="Phrase that must appear in the first pages.")


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str
    documents: list[DocEntry]


class LockEntry(BaseModel):
    """What was actually downloaded and checked (data/docs/manifest.lock.json)."""

    id: str
    url: str
    verified: bool
    reason: str
    sha256: str | None = None
    bytes: int | None = None
    pages: int | None = None
    downloaded_at: datetime | None = None


def load_manifest(path: Path) -> Manifest:
    m = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [d.id for d in m.documents]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(f"duplicate document ids: {dups}")
    return m


# --- PDF text ---


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    # Control characters (PDF extraction can yield NUL bytes, which Postgres text rejects).
    text = _CONTROL.sub("", text)
    text = text.replace(" ", " ").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_pages(data: bytes) -> list[str]:
    """Text of each page (index 0 = page 1). Empty string for pages without extractable text
    (scanned images are not OCR'd; they simply contribute nothing)."""
    from pypdf import PdfReader

    # pypdf warns about fonts it cannot fully decode; the text quality is checked by
    # `verify` (and was measured for the corpus), so the warnings are only noise here.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(io.BytesIO(data))
    return [_clean(page.extract_text() or "") for page in reader.pages]


def verify(entry: DocEntry, data: bytes) -> tuple[bool, str, list[str]]:
    """(ok, reason, pages). A document must be a PDF containing `expect` near the start."""
    if not data.startswith(b"%PDF"):
        return False, "not a PDF (the server returned something else)", []
    try:
        pages = pdf_pages(data)
    except Exception as exc:  # malformed PDF: reported, never ingested
        return False, f"unreadable PDF ({type(exc).__name__})", []
    head = re.sub(r"\s+", " ", " ".join(pages[:VERIFY_PAGES])).lower()
    needle = re.sub(r"\s+", " ", entry.expect).lower()
    if needle not in head:
        return False, f"expected text {entry.expect!r} not found in the first pages", pages
    if sum(len(p) for p in pages) < 500:
        return False, "almost no extractable text (scanned PDF?)", pages
    return True, "ok", pages


# --- chunking ---


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>:p<page>:<n>"
    doc_id: str
    title: str
    doc_type: str
    fund_code: str | None
    url: str
    page: int  # 1-based
    text: str


_BREAK = re.compile(r"(?<=[.;:!?])\s+|\n+")


def split_page(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Windows of at most `size` characters, ending at a sentence/line break when one exists in
    the second half of the window, each starting `overlap` characters before the previous end."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cuts = [m.end() for m in _BREAK.finditer(window) if m.end() >= size // 2]
            if cuts:
                end = start + cuts[-1]
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document(entry: DocEntry, pages: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        for n, piece in enumerate(p for p in split_page(text) if len(p) >= MIN_CHUNK_CHARS):
            chunks.append(
                Chunk(
                    id=f"{entry.id}:p{page_no}:{n}",
                    doc_id=entry.id,
                    title=entry.title,
                    doc_type=entry.doc_type,
                    fund_code=entry.fund_code,
                    url=entry.url,
                    page=page_no,
                    text=piece,
                )
            )
    return chunks


def lock_entry(entry: DocEntry, ok: bool, reason: str, data: bytes | None, pages: int) -> LockEntry:
    return LockEntry(
        id=entry.id,
        url=entry.url,
        verified=ok,
        reason=reason,
        sha256=hashlib.sha256(data).hexdigest() if data else None,
        bytes=len(data) if data else None,
        pages=pages or None,
        downloaded_at=datetime.now(UTC) if data else None,
    )


def write_lock(path: Path, entries: list[LockEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.model_dump(mode="json") for e in entries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
