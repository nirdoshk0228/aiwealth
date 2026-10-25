"""Scoring functions for the evaluation (eval/run_eval.py). Pure, deterministic, unit-tested.

Retrieval (no LLM):
    hit@k  = 1 if any of the top-k chunks is on a gold (document, page), else 0
    MRR@10 = 1 / rank of the first gold chunk in the top 10 (0 if none)
Answers (LLM output scored by rules, so re-runs are free and reproducible):
    key_facts   every number in the gold phrases appears in the answer (as a value; single-digit
                integers without a unit excepted, as in the grounding check), and every gold
                phrase without numbers appears (normalised text)
    cites_valid the answer cites at least one passage and only passages it was shown
    cites_gold  at least one citation is a gold page
    grounded    numeric grounding passes against the passages and the question
    abstained   (unanswerable questions) the answer says "not found in the documents"
Tool calls:
    tool_correct  the expected tool was called (or none, when none is expected)
    args_correct  the expected arguments match on that call (dotted paths; numbers to 1e-6)
"""

import re
from collections.abc import Sequence
from typing import Any

from app.ai.guardrails import numbers

NOT_FOUND = "not found in the documents"


def _norm(s: str) -> str:
    """Lower case; hyphens of every kind, other punctuation and all spaces (incl. U+202F)
    become single spaces, so "equity-related" with U+2011 matches "Equity Related"."""
    s = re.sub(r"[-‐‑‒–—−]", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w%.,/]+", " ", s)).strip()


# --- retrieval ---


def first_gold_rank(retrieved: Sequence[tuple[str, int]], gold: set[tuple[str, int]]) -> int | None:
    for rank, key in enumerate(retrieved, start=1):
        if key in gold:
            return rank
    return None


def retrieval_scores(
    retrieved: Sequence[tuple[str, int]], gold: set[tuple[str, int]]
) -> dict[str, float]:
    r = first_gold_rank(retrieved[:10], gold)
    return {
        "hit@1": float(r is not None and r <= 1),
        "hit@3": float(r is not None and r <= 3),
        "hit@5": float(r is not None and r <= 5),
        "mrr@10": 1.0 / r if r is not None else 0.0,
    }


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


# --- answers ---


def key_facts_present(answer: str, phrases: Sequence[str]) -> bool:
    """Numbers of the gold phrases must appear in the answer; phrases without numbers must
    appear as text (normalised: case, spacing, punctuation)."""
    ans_norm = _norm(answer)
    ans_nums = {(round(abs(n.value), 6), n.percent) for n in numbers(answer)}
    for phrase in phrases:
        # Same exemption as the grounding check: single-digit integers without a unit ("Re. 1",
        # list markers) are not key facts.
        nums = [
            n for n in numbers(phrase) if n.percent or not re.fullmatch(r"\d", n.text.lstrip("+- "))
        ]
        if nums:
            for n in nums:
                key = (round(abs(n.value), 6), n.percent)
                # Accept "2.00 %" written as "2%" (same value, both percent) - handled by value.
                if key not in ans_nums:
                    return False
        elif _norm(phrase) not in ans_norm:
            return False
    return True


def abstained(answer: str) -> bool:
    return NOT_FOUND in answer.lower()


# --- tool calls ---


def _get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for path, want in expected.items():
        got = _get(actual, path)
        if isinstance(want, int | float) and not isinstance(want, bool):
            if not isinstance(got, int | float) or abs(float(got) - float(want)) > 1e-6 * max(
                1.0, abs(float(want))
            ):
                return False
        elif isinstance(want, str):
            if not isinstance(got, str) or got.strip().lower() != want.strip().lower():
                return False
        elif got != want:
            return False
    return True


def score_tool_case(
    calls: Sequence[tuple[str, dict[str, Any]]],
    expect_tool: str | None,
    expect_args: dict[str, Any],
    also_accept: Sequence[dict[str, Any]] = (),
) -> dict[str, float]:
    """`calls` = (tool name, arguments) in order. list_stress_scenarios may precede the
    expected stress call (looking up an id is reasonable), other extra calls do not matter.

    `also_accept` lists further valid answers for an ambiguous prompt, each
    {"tool": name, "args": {...}}; the case scores as its best-matching alternative."""
    options = [(expect_tool, expect_args)] + [(a["tool"], a.get("args", {})) for a in also_accept]
    best = {"tool_correct": 0.0, "args_correct": 0.0}
    for tool, args in options:
        if tool is None:
            ok = float(not calls)
            score = {"tool_correct": ok, "args_correct": ok}
        else:
            matching = [a for name, a in calls if name == tool]
            score = {
                "tool_correct": float(bool(matching)),
                "args_correct": float(any(args_match(a, args) for a in matching)),
            }
        best = max(best, score, key=lambda x: (x["args_correct"], x["tool_correct"]))
    return best


_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "any",
    "are",
    "was",
    "were",
    "its",
    "into",
    "under",
    "per",
    "has",
    "have",
    "been",
    "will",
    "shall",
    "may",
    "not",
    "all",
}


def _content_words(s: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _norm(s))
    return {w.rstrip("s") for w in words if len(w) >= 3 and w not in _STOP}


def key_facts_lenient(answer: str, phrases: Sequence[str]) -> bool:
    """A looser companion to `key_facts_present` (added after inspecting the 2026-10-25 run, so
    reported separately, never instead of the strict one): numbers match by value whether or not
    either side has a % sign; a phrase without numbers matches when all its content words (3+
    letters, minus common stop words, plural s ignored) occur in the answer, in any order."""
    ans_values = {round(abs(n.value), 6) for n in numbers(answer)}
    ans_words = _content_words(answer)
    for phrase in phrases:
        nums = [
            n for n in numbers(phrase) if n.percent or not re.fullmatch(r"\d", n.text.lstrip("+- "))
        ]
        if nums:
            if any(round(abs(n.value), 6) not in ans_values for n in nums):
                return False
        elif not _content_words(phrase) <= ans_words:
            return False
    return True
