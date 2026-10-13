# eval

Evaluation is a first-class deliverable. Planned contents:

- `golden_qa.jsonl`: about 50 questions, including unanswerable ones, for RAG eval
  (hit@3/5 and faithfulness; dense vs hybrid vs hybrid+rerank)
- `tool_call_cases.jsonl`: 30-40 prompts for tool-selection accuracy
- VaR backtest: breach rate and Kupiec test

`make eval` will write reports to `eval/results/` (gitignored). Only numbers from those reports may
appear in the README or resume.
