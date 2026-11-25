# Implementation checklist

- [x] Inspect workspace and machine; record unavailable checks.
- [x] Create plan, architecture, decisions, and checklist.
- [x] Foundation, configuration, validated schemas, and provider abstraction.
- [x] PDF/TXT/Markdown/CSV ingestion, section-aware chunking, source metadata, lexical retrieval.
- [x] Intake, requirements, explicit-gap groundability, and declarative agent generation.
- [x] Safe registered calculator, current-date, CSV lookup, and keyless weather tools.
- [x] Dynamic source-backed evaluation cases; all three executed collections produced 20 cases.
- [x] Deterministic judge and optional Ollama semantic judge with separate scores/disagreement reporting.
- [x] Failure classification and maximum-three-iteration targeted optimizer.
- [x] Generic CLI runner with citations, refusal, ambiguity handling, and selected tools.
- [x] HR and marine demo collections with saved run artifacts.
- [x] Separate university lab generalization run with saved artifacts.
- [x] Automated test coverage and local results recorded.
- [x] Streamlit interface backed by the same build/runner/report functions; AppTest smoke run passed.
- [x] README, demo script, traceability, limitations, and submission preparation.
- [ ] Live Ollama answer and semantic-judge run; Ollama was absent, and no model was downloaded.
- [ ] Semantic review of generated topic coverage and evaluation wording by a domain expert.

## Latest verification

- `.venv\Scripts\python.exe -m pytest -q`: 24 passed.
- Python compileall: passed.
- Streamlit AppTest: zero exceptions.
- `pip check`: no broken requirements.
- HR policy: 20 cases; initial deterministic pass rate 80%, optimized final 100% after one targeted retrieval iteration (`--top-k 1`).
- Marine procurement: 20 cases, final deterministic pass rate 100%.
- University lab generalization: 20 cases, final deterministic pass rate 100%.
- Marine and university scores are lexical/extractive results with no needed optimizer changes; these are not live LLM quality scores.
