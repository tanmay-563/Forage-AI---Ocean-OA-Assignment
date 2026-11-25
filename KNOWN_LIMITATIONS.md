# Known limitations

- Default provider is offline and extractive. No live answer model was used in the latest regression runs.
- LLM judging is opt-in (`FORGEAI_JUDGE_PROVIDER=ollama`, `FORGEAI_JUDGE_MODEL=...`). No independent evaluator was available in the current environment, so semantic judgment was unavailable and deterministic checks were used. Same-model judging is disclosed when configured; it is not independent.
- Anonymous A/B order-swapped judge calls detect some position-sensitive outcomes, but they do not establish evaluator impartiality or eliminate bias. The Ollama rubric itself is not calibrated against human judgments.
- Deterministic semantic correctness is a lexical proxy and can penalize valid paraphrases or accept superficial word overlap. Writing quality currently uses a simple length proxy, not a trained readability metric.
- Case generation uses deterministic source sentence and CSV field templates. Multi-section and cross-document comparison cases require shared terms; some prompts may be broad or fail retrieval. Inspect each generated case and per-case judge result before using evaluation scores as product-quality claims.
- CSV factual cases target named fields and row-level field pairs; numeric comparison is implemented for matching currency and price columns. Ambiguity, conflicting evidence, and nuanced refusal cases are not synthesized unless a safe evidence-grounded rule exists; coverage is not guaranteed.
- The holdout is a fixed, deterministic one-in-five split of factual cases only when at least 25 factual cases exist. Smaller collections disclose that no meaningful holdout is available. Holdout results are not used to select changes.
- The optimizer changes limited prompt and retrieval settings. It does not tune chunking, query rewrites, embeddings, or rerankers. A score improvement on the optimization set is not a general quality guarantee.
- PDF extraction can still produce artifacts or lose layout/table relationships. Empty and scanned pages are diagnosed but OCR is not implemented.
- Topic coverage is inferred from headings and lexical matches; nuanced contradictions and partial coverage need human review.
- Weather calls Open-Meteo and requires network access; tests mock the HTTP call. It is attached only when selected.
- Demo HR, procurement, and lab safety documents are synthetic and not real legal, supplier, engineering, or safety advice.
- Streamlit retains uploaded files under `.forgeai_uploads/` so agent definitions remain runnable. Users can remove those local uploads when no longer needed.
