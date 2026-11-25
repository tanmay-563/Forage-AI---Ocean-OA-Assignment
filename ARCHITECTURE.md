# Architecture

ForgeAI uses one generic Python pipeline for CLI and Streamlit. A brief and document collection flow through ingestion, requirements, an agent definition, a retrieval-backed runner, evidence-derived evaluation, judging, bounded optimization, and reports.

```text
brief + documents
  -> intake / requirements / diagnostics
  -> extraction and provenance-preserving chunks
  -> generated agent and selected tools
  -> deterministic evaluation cases + exact evidence references
  -> deterministic judge (+ optional explicitly configured Ollama judge)
  -> <=3 optimization iterations on a fixed optimization set
  -> post-selection holdout check, scorecards, per-case records, reports
```

## Evaluation generation and provenance

`forgeai/evaluation/cases.py` generates cases from qualifying source text and CSV row fields. Prose and PDF text use sentence-quality filters; CSV rows use non-empty named fields and retain the row number. Case provenance can include multiple `EvidenceProvenance` references. Validation checks each exact quotation, source filename, chunk ID, page/row metadata, and the primary reference. Multi-section or cross-document comparisons require separate evidence references and are generated only when the sources have a shared, specific subject. Numeric CSV comparisons record both quoted operands and verify the derived difference and lower-priced item. Missing-information/refusal cases are explicit and counted separately. Every build enforces at least 20 factual cases.

`forgeai/ingestion.py` records page/row diagnostics and prose exclusion reasons. Evaluations do not force categories that the source set cannot substantiate. Distinct facts are not enough by themselves to guarantee useful questions: the deterministic topic templates remain a known quality limitation.

## Judge and evaluator configuration

`forgeai/evaluation/judge.py` computes deterministic lexical correctness, evidence presence, citations, refusal correctness, completeness, and a separate writing-quality proxy. Writing quality does not affect the factual pass threshold. Per-case records preserve retrieved evidence, checks, explanation, any LLM result, disagreements, and final decision.

`forgeai/evaluation/semantic_judge.py` optionally sends two anonymous answer texts to an Ollama evaluator in both orders. It records preference changes and independent/same-model metadata. Configure `FORGEAI_JUDGE_PROVIDER=ollama` and `FORGEAI_JUDGE_MODEL`; no evaluator is the default and produces an explicit unavailable status. Same-model judging is disclosed and is not called independent. Blind labels and order swaps are bias probes, not guarantees against bias.

## Optimizer and interfaces

`forgeai/evaluation/optimizer.py` preserves case IDs, caps the loop at three iterations, records detailed failed-case evidence and exact configuration snapshots, and selects changes using optimization-set results only. With at least 25 factual cases, a deterministic one-in-five holdout is scored after selection and is not used for candidate acceptance. Smaller sets disclose the lack of a useful holdout. Regression and score decreases are retained in logs.

`forgeai/pipeline.py` is shared by `forgeai/cli.py` and `app.py`; both interfaces therefore use the same case validation and judge. The runner loads definitions and their own collection dynamically. Tools are allowlisted and validated before execution.
