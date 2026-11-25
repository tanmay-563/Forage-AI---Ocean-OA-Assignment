# Implementation checklist

- [x] Inspect workspace and machine; record unavailable checks.
- [x] Create plan, architecture, decisions, and checklist.
- [x] Foundation, configuration, validated schemas, and provider abstraction.
- [x] PDF/TXT/Markdown/CSV ingestion, section-aware chunking, source metadata, lexical retrieval.
- [x] Intake, requirements, explicit-gap groundability, and declarative agent generation.
- [x] Safe registered calculator, current-date, CSV lookup, and keyless weather tools.
- [x] Provenance-validated source cases; prose/PDF quality filters, CSV row/column references, checked numerical comparisons, and separately counted refusal cases.
- [x] Optional anonymous order-swapped Ollama judge with independent scoring dimensions and same-model disclosure; no evaluator is marked unavailable.
- [x] Detailed optimizer evidence, exact before/after settings, fixed post-selection holdout when sample size permits, and three-iteration cap.
- [x] Generic CLI runner with citations, refusal, ambiguity handling, and selected tools.
- [x] HR and marine demo collections with saved run artifacts.
- [x] Separate university lab generalization run with saved artifacts.
- [x] Automated tests for provenance, CSV, comparison evidence, blind order swapping, scoring separation, and holdout split.
- [x] Streamlit interface backed by the same build/runner/report functions; AppTest smoke run passed.
- [x] README, demo script, traceability, limitations, and submission preparation.
- [ ] Live Ollama answer and semantic-judge run; Ollama was absent, and no model was downloaded.
- [ ] Semantic review of generated topic coverage and evaluation wording by a domain expert.

## Latest verification

Latest verification:

- `.venv\Scripts\python.exe -m pytest -q`: 46 passed.
- HR demo: 42 evidence-backed + 2 refusal cases; 42/44 pass. Optimization set 88.57% to 97.14%; 9-case holdout 77.78% to 88.89%; one accepted iteration raised top-k 1 to 3 and added the evidence/citation instruction.
- Marine procurement: 76 evidence-backed + 2 refusal cases; 74/78 pass. Optimization set 96.77% unchanged; 16-case holdout 87.5% unchanged. Candidate raised top-k 5 to 7 and added the evidence/citation instruction, but was not accepted because its optimization score did not improve.
- University lab safety: 25 evidence-backed + 2 refusal cases; 27/27 pass. Five-case holdout 100% before and after; no optimizer change was needed.
- Internship PDF: 6 evidence-backed candidates and 2 refusal cases; build correctly stopped because 20 factual cases are mandatory.
- CLI evaluation completed for all three finished demos. Streamlit AppTest started with zero exceptions.
- No LLM evaluator was configured; semantic judging is marked unavailable. No independent evaluator was used.

Outputs are in `artifacts_quality_release_final2/`. Earlier artifacts were preserved and not updated by this verification.
