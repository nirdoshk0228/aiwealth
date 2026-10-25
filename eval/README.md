# eval

Evaluation is a first-class deliverable. Only numbers produced by `make eval` may appear in the
README or a resume.

| File | What it is |
|---|---|
| `golden_qa.source.jsonl` | Questions with the document(s) and short phrases that answer them (hand-written) |
| `build_golden.py` | Builds `golden_qa.jsonl`: gold pages are **computed** as the pages whose text contains all the phrases; a question whose phrases are found nowhere fails the build. Unanswerable questions must concern funds without documents |
| `golden_qa.jsonl` | 59 questions (49 answerable, 10 unanswerable), every row `"status": "DRAFT"` until reviewed |
| `tool_call_cases.jsonl` | 36 prompts with the expected tool and key arguments |
| `run_eval.py` | The suite (see its docstring); writes `results/summary.md` and `results/results.json` |
| `results/` | Output (gitignored) |

```
python scripts/seed_data.py && python scripts/ingest_docs.py   # data and document index
make eval                                                     # or: python eval/run_eval.py
python eval/run_eval.py --only rag --only var                 # no LLM calls
```

Metrics (details in `apps/api/app/ai/evaluation.py`):

- **Retrieval:** hit@1/3/5 and MRR@10 on gold (document, page) pairs, for dense, hybrid
  (BM25 + dense, RRF) and hybrid + cross-encoder rerank, without and with the fund filter.
- **Answers** (generated from the retrieved passages with the best configuration): key facts
  present, citations valid, cites a gold page, numbers grounded, correct abstention ("not found in
  the documents") on unanswerable questions, false abstention on answerable ones; optional LLM
  judge for faithfulness (`--judge`).
- **Two key-facts metrics.** *Strict*: every number of the gold phrase by value and kind, every
  number-free phrase as normalised text. *Lenient*: numbers by value whether or not a % sign is
  written, number-free phrases by content words in any order. The lenient one was **added after
  inspecting the first run** (correct paraphrases such as "65 % to 100 %" for "(65-100 %)" were
  scored as misses), so it is reported next to the strict one, never instead of it.
- **Ambiguous tool cases.** A case may list `also_accept` alternatives; it scores as its
  best-matching valid answer. Only t18 ("How exposed am I to an IT sector shock scenario?") has
  one: the committed `it_sector_shock` scenario and a custom IT-sector shock both answer it
  (decided 2026-10-25, wording kept). The report prints each case's grading note.
- **Fixes made after the first run, and why they are not tuning:** the citation parser now also
  accepts the 【doc p.N】 brackets the model uses (a product bug: citation chips were missing);
  text normalisation treats Unicode hyphens as spaces; the assistant is now told whether a
  portfolio is loaded (a product bug: it sometimes declined to call portfolio tools); fund-name
  matching ignores spacing ("Flexicap" = "Flexi Cap"). None of them changes which passages or
  answers count as correct.
- **Tool calls:** the expected tool was called, and with the expected arguments.
- **VaR backtest:** rolling 1-day VaR of ^NSEI and an equal-weight 20-stock basket, compared with
  the table in docs/methodology.md.

Free-tier budget: LLM responses are cached on disk (`.cache/llm`, gitignored), so re-runs cost
nothing; uncached calls are paced under ~7,000 tokens a minute and the report states the tokens
used. The golden set is a **test set**: retrieval settings were not tuned on it.
