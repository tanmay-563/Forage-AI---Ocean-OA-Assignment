# Requirement traceability

| Requirement | Implementation | Test coverage | Demonstration / verification | Status |
|---|---|---|---|---|
| Shared CLI/UI generation and validation | `forgeai/pipeline.py`, `forgeai/cli.py`, `app.py` | `test_cli_and_streamlit_import_the_shared_build_pipeline` | Fresh CLI builds and Streamlit AppTest | Implemented |
| PDF, TXT, Markdown, CSV ingestion and diagnostics | `forgeai/ingestion.py` | PDF failure, empty page, CSV row, text tests | `source_diagnostics.json` per build | Implemented; PDF layout/OCR limits remain |
| Minimum 20 evidence-backed cases; separate refusals | `forgeai/evaluation/cases.py`, `forgeai/pipeline.py` | minimum and insufficient-content tests | Fresh demo case files; internship PDF stopped below minimum | Implemented; source-limited builds fail honestly |
| Exact evidence provenance across multiple chunks | `forgeai/models.py`, `forgeai/evaluation/cases.py` | quote, page, missing chunk, multi-document references | `evals/cases.json` | Implemented |
| Natural, diverse, source-grounded categories incl. CSV and numeric cases | `forgeai/evaluation/cases.py` | CSV row/column, numeric calculation, and cross-document provenance tests | Category counts in `evaluation_quality.json` | Partial; deterministic topic templates can be broad |
| Separate deterministic judge dimensions and transparent records | `forgeai/evaluation/judge.py`, `forgeai/models.py`, `forgeai/pipeline.py` | scoring separation and fallback tests | `evaluation_results.json`, scorecards | Implemented; correctness is lexical |
| Blind comparison and order swapping | `forgeai/evaluation/semantic_judge.py` | blind payload and position-sensitive decision test | Optional Ollama configuration only | Implemented as a probe; live judge not run |
| Independent evaluator config and disclosure | `forgeai/config.py`, `forgeai/evaluation/optimizer.py`, `semantic_judge.py` | same-model metadata test; judge fallback test | `FORGEAI_JUDGE_PROVIDER`, `FORGEAI_JUDGE_MODEL` | Implemented; independent evaluator not available during this run |
| Detailed optimizer, exact settings, holdout, <=3 iterations | `forgeai/evaluation/optimizer.py`, `forgeai/pipeline.py` | max-iteration and holdout split tests | `optimization_logs.json`, `optimization_report.json` | Implemented; holdout only with >=25 factual cases |
| HR, marine, and lab demos | Generic pipeline and `demos/` | Full pytest + CLI build/evaluate | `artifacts_quality_release_final2/` | HR 42+2 cases; marine 76+2; lab 25+2; pass counts 42/44, 74/78, 27/27 |
| Internship insufficient-evidence scenario | `forgeai/evaluation/cases.py`, `pipeline.py` | insufficient content test | Diagnostic output in `artifacts_quality_release_final2/data_science_internship_assistant/` | 6 evidence candidates; stopped below 20, no fabricated completion |
| Documentation reflects implementation and limits | `README.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_CHECKLIST.md`, `KNOWN_LIMITATIONS.md` | Documentation reviewed with source | Current repository docs | Updated |
