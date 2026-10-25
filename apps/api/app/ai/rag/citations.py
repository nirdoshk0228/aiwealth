"""Passages for the model and citations back from its answer.

The knowledge-base tool gives each passage a reference "<doc_id> p.<page>". The system prompt
asks the model to cite claims as [<doc_id> p.<page>]. After the answer, every bracketed
reference is matched against the passages actually retrieved in this turn:
- matched references become Citation objects (document title, page, source URL) for the UI;
- references to anything not retrieved are reported as invented citations (a guardrail flag).
"""

import re
from typing import Any

from app.ai.rag.retrieve import Hit
from app.schemas.chat import Citation

PASSAGE_CHARS = 700  # per passage shown to the model (token budget)
NOT_FOUND = "not found in the documents"

# Square brackets as instructed, or the fullwidth lenticular brackets some models (gpt-oss) use
# for citations: 【sid_x p.7】. Narrow no-break spaces count as spaces (\s covers them).
_REF = re.compile(r"[\[【]\s*([a-z0-9_]+)\s+p\.?\s*(\d+)\s*[\]】]", re.IGNORECASE)


def ref(doc_id: str, page: int) -> str:
    return f"{doc_id} p.{page}"


def passages(hits: list[Hit]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        text = h.chunk.text
        if len(text) > PASSAGE_CHARS:
            text = text[:PASSAGE_CHARS].rsplit(" ", 1)[0] + " ..."
        out.append(
            {
                "ref": ref(h.chunk.doc_id, h.chunk.page),
                "document": h.chunk.title,
                "page": h.chunk.page,
                "text": text,
                "url": h.chunk.url,
                "chunk_id": h.chunk.id,
            }
        )
    return out


def extract_citations(
    answer: str, retrieved: list[dict[str, Any]]
) -> tuple[list[Citation], list[str]]:
    """(citations in order of first mention, references not among the retrieved passages)."""
    known = {p["ref"].lower(): p for p in retrieved}
    cites: list[Citation] = []
    seen: set[str] = set()
    invented: list[str] = []
    for m in _REF.finditer(answer):
        key = ref(m.group(1).lower(), int(m.group(2))).lower()
        if key in seen:
            continue
        seen.add(key)
        p = known.get(key)
        if p is None:
            invented.append(f"[{m.group(1)} p.{m.group(2)}]")
            continue
        cites.append(
            Citation(
                document=p["ref"].split(" p.")[0],
                title=p["document"],
                page=p["page"],
                url=p.get("url"),
            )
        )
    return cites, invented
