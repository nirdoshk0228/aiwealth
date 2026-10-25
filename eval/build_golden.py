"""Build eval/golden_qa.jsonl from eval/golden_qa.source.jsonl and the downloaded documents.

For each answerable question, the gold pages are computed, never typed: every page of the listed
documents whose text contains ALL `answer_contains` phrases (case- and whitespace-insensitive).
A question whose phrases are found nowhere fails the build, so a typo or a wrong document cannot
enter the set. Unanswerable questions (answerable: false) are checked the other way: their fund
must not have any document in the corpus.

Every row is marked "status": "DRAFT" until the developer reviews it.

Usage: python eval/build_golden.py   (needs data/docs/*.pdf from scripts/ingest_docs.py)
"""

import json
import re
import sys
from pathlib import Path

from app.ai.rag.corpus import load_manifest, pdf_pages
from app.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "golden_qa.source.jsonl"
OUT = ROOT / "eval" / "golden_qa.jsonl"
DOCS = ROOT / "data" / "docs"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> int:
    manifest = load_manifest(ROOT / "data" / "docs_manifest.json")
    doc_ids = {d.id for d in manifest.documents}
    covered_codes = {d.fund_code for d in manifest.documents if d.fund_code}
    names = {f.name.split(" - ")[0].lower(): f.code for f in load_universe().funds}
    cache: dict[str, list[str]] = {}

    def pages(doc_id: str) -> list[str]:
        if doc_id not in cache:
            cache[doc_id] = [norm(p) for p in pdf_pages((DOCS / f"{doc_id}.pdf").read_bytes())]
        return cache[doc_id]

    rows, errors = [], []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        q.setdefault("answerable", True)
        q["status"] = "DRAFT"
        if q["answerable"]:
            gold = []
            for d in q["doc_ids"]:
                if d not in doc_ids:
                    errors.append(f"{q['id']}: unknown document {d}")
                    continue
                phrases = [norm(a) for a in q["answer_contains"]]
                gold += [
                    {"doc_id": d, "page": i + 1}
                    for i, text in enumerate(pages(d))
                    if all(p in text for p in phrases)
                ]
            if not gold:
                errors.append(f"{q['id']}: answer phrases not found in {q['doc_ids']}")
            q["gold"] = gold
        else:
            fund = (q.get("fund") or "").lower()
            code = next((c for n, c in names.items() if fund and n.startswith(fund)), None)
            if code is not None and code in covered_codes:
                errors.append(f"{q['id']}: {q['fund']} has a document; not unanswerable")
        rows.append(q)
    if errors:
        print("\n".join(errors))
        return 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    n_ans = sum(r["answerable"] for r in rows)
    print(f"wrote {OUT}: {len(rows)} questions ({n_ans} answerable, {len(rows) - n_ans} not)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
