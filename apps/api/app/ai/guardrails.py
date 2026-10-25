"""Guardrails on what goes into and comes out of the model.

1. Numeric grounding (output). Every number in an answer must appear in a source the model was
   given in this turn: a tool output, a retrieved passage, or the user's own messages. Numbers
   are compared as values after normalising their format, so "Rs 13,843", "13,843" and
   "₹13,843" match, and "1.38%" matches "1.38%" (a percentage only matches a percentage).
   Not checked: integers below 10 written without a unit (list markers, "six levels", "3
   years" style wording), and four-digit years 1990-2039 that appear in a source date.
2. Compliance (output). Phrases that tell the user to buy, sell, hold or switch, recommend a
   product, or promise returns. Detected sentences are reported; `redact_advice` replaces them.
3. Prompt injection (input). Retrieved passages are untrusted text from PDFs. Sentences that
   look like instructions to the model are removed before the model sees them.

The agent loop uses these: an answer that fails 1 or 2 gets one repair turn (the model is told
exactly what to fix); if it still fails, ungrounded numbers are flagged to the user and advice
sentences are redacted. The disclaimer is always attached.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# --- numeric grounding ---

_NUM = re.compile(
    r"(?<![\w.])(?:₹\s?|rs\.?\s?|inr\s?)?"  # optional currency prefix
    r"([-+−‑–]?\s?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|[-+−‑–]?\s?\d+(?:\.\d+)?)"  # 1,23,456.7 or 12.5
    r"(\s?%)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^(19[9]\d|20[0-3]\d)$")


@dataclass(frozen=True)
class Number:
    text: str  # as written in the answer
    value: float
    percent: bool


def _parse(m: re.Match[str]) -> Number | None:
    raw = m.group(1)
    cleaned = (
        raw.replace(",", "").replace(" ", "").replace("−", "-").replace("‑", "-").replace("–", "-")
    )
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return Number(text=m.group(0).strip(), value=value, percent=m.group(2) is not None)


def numbers(text: str) -> list[Number]:
    out = []
    for m in _NUM.finditer(text):
        n = _parse(m)
        if n is not None:
            out.append(n)
    return out


def _date_parts(text: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))


def _exempt(n: Number, source_years: set[str]) -> bool:
    if n.percent:
        return False
    digits = n.text.lstrip("+-−‑– ")
    if re.fullmatch(r"\d", digits):  # single-digit integers: list markers, small counts
        return True
    return bool(_YEAR.match(digits)) and digits in source_years


def _key(n: Number) -> tuple[float, bool]:
    return (round(abs(n.value), 6), n.percent)


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    ungrounded: list[str]
    checked: int


def source_text(
    outputs: Iterable[dict[str, Any]], passages: Iterable[dict[str, Any]], user: Iterable[str]
) -> str:
    parts = [json.dumps(o, ensure_ascii=False) for o in outputs]
    parts += [p.get("text", "") for p in passages]
    parts += list(user)
    return "\n".join(parts)


def check_grounding(answer: str, sources: str) -> GroundingResult:
    """Every number of `answer` (bar the exemptions in the module docstring) must occur, as a
    value with the same percent-or-not kind, somewhere in `sources`. Signs are ignored when
    matching, because answers legitimately say "a loss of 8.31%" for a P&L of "-8.31%"."""
    available = {_key(n) for n in numbers(sources)}
    years = _date_parts(sources)
    # Citations like [sid_x p.7] are references, not figures.
    body = re.sub(r"\[[a-z0-9_]+\s+p\.?\s*\d+\]", " ", answer, flags=re.IGNORECASE)
    bad: list[str] = []
    checked = 0
    for n in numbers(body):
        if _exempt(n, years):
            continue
        checked += 1
        if _key(n) not in available and n.text not in bad:
            bad.append(n.text)
    return GroundingResult(grounded=not bad, ungrounded=bad, checked=checked)


# --- compliance ---

_ADVICE = [
    r"\byou\s+(?:should|must|need\s+to|ought\s+to)\s+(?:buy|sell|invest|switch|redeem|exit|hold|"
    r"increase|reduce|add|move|put|allocate|stop)\b",
    r"\b(?:i|we)\s+(?:would\s+)?(?:recommend|suggest|advise)\s+(?:that\s+you\s+)?"
    r"(?:buy|sell|invest|switch|redeem|exit|hold|buying|selling|investing|switching|"
    r"redeeming|holding|increasing|reducing|allocating|moving)\b",
    r"\b(?:buy|sell)\s+(?:now|immediately|today)\b",
    r"\bguaranteed\s+(?:returns?|profits?|gains?)\b",
    r"\b(?:will|is\s+going\s+to)\s+(?:definitely|surely|certainly)\s+"
    r"(?:rise|grow|outperform|beat|double|go\s+up)\b",
    r"\bthe\s+best\s+(?:fund|stock|investment)\s+(?:to|for)\s+(?:buy|invest)\b",
    r"\b(?:it\s+is|it's)\s+a\s+good\s+time\s+to\s+(?:buy|sell|invest)\b",
]
_ADVICE_RE = re.compile("|".join(f"(?:{p})" for p in _ADVICE), re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]*[.!?]?", re.MULTILINE)
REDACTED = "[Removed: the assistant does not give buy, sell or hold recommendations.]"


def advice_sentences(answer: str) -> list[str]:
    out = []
    for s in _SENTENCE.findall(answer):
        if s.strip() and _ADVICE_RE.search(s):
            out.append(s.strip())
    return out


def redact_advice(answer: str) -> str:
    for s in advice_sentences(answer):
        answer = answer.replace(s, REDACTED)
    return answer


# --- prompt injection in retrieved text ---

_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)?\s*(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier|system)"
    r"|\bsystem\s+prompt\b|\byou\s+are\s+now\b|\bact\s+as\s+(?:a|an|the)\b"
    r"|\bnew\s+instructions?\s*:|\b(?:assistant|system|user)\s*:\s"
    r"|<\s*/?\s*(?:script|system|instructions?)\b"
    r"|\b(?:call|use|invoke)\s+the\s+\w+\s+tool\b"
    r"|\bdo\s+not\s+(?:tell|inform)\s+the\s+user\b",
    re.IGNORECASE,
)
INJECTION_REMOVED = "[removed: instruction-like text]"


def sanitize_passage(text: str) -> tuple[str, int]:
    """Remove sentences that look like instructions to an AI. Returns (text, sentences removed).
    Document text is data: legitimate SIDs do not address an assistant, so false positives are
    rare, and a removed sentence is visible in the trace."""
    removed = 0
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if _INJECTION.search(s):
            removed += 1
            out.append(INJECTION_REMOVED)
        else:
            out.append(s)
    return " ".join(x for x in out if x), removed
