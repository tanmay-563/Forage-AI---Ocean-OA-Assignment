# Requirement traceability

| Requirement | Implementation | Test | Demo artifact / verification | Status |
|---|---|---|---|---|
| Natural-language brief and document folder | `forgeai/intake.py`, `forgeai/cli.py` | `test_intake_asks_at_most_five_and_records_unattended` | All generated `requirements.json` files | Implemented and run |
| Up to five intake questions and structured requirements | `forgeai/intake.py`, `forgeai/models.py` | intake test | `requirements.json`, `intake_questions.json` | Implemented |
| Groundability, unsupported topics, explicit gaps | `forgeai/groundability.py` | `test_groundability_topics_and_retrieval` | Each `groundability.json` | Partial; explicit gaps detected, nuanced conflicts need review |
| Declarative generic agent | `forgeai/generator.py`, `forgeai/models.py`, `forgeai/runner.py` | definition/runner tests | Each `agent_definition.json` | Implemented |
| PDF, TXT, Markdown, CSV ingestion and references | `forgeai/ingestion.py` | PDF/TXT/CSV tests | Supplier PDFs and catalogue CSV | Implemented and tested with PyMuPDF |
| Local retrieval, citations, refusal, ambiguity | `forgeai/retrieval.py`, `forgeai/runner.py` | retrieval, runner, refusal, ambiguity tests | CLI chat and saved scorecards | Implemented; lexical retrieval limits noted |
| Tool registry, safe calculator, CSV lookup, date, weather | `forgeai/tools/registry.py` | tool tests with mocked weather HTTP | Marine agent and tool CLI | Implemented; tools are executable |
| Dynamic 20+ evaluation cases | `forgeai/evaluation/cases.py` | dynamic-case test | Each `evals/cases.json` has 20 cases | Implemented; lexical question wording has limits |
| Deterministic and optional semantic judge | `forgeai/evaluation/judge.py`, `forgeai/evaluation/semantic_judge.py` | deterministic and mocked semantic judge tests | Scorecards and disagreement fields | Implemented; live Ollama judge not run |
| Four failure classes and <=3 optimizer iterations | `forgeai/evaluation/optimizer.py` | failure-class and optimizer-limit tests | HR optimization log; latest run made no changes | Implemented; limited targeted knobs |
| Honest final report | `forgeai/pipeline.py` | three end-to-end CLI runs | Three `final_report.json` and `.md` files | Implemented with actual deterministic scores |
| Generic runner loads new definitions | `forgeai/runner.py`, `forgeai/cli.py` | generic runner test | All three definitions load using relative collection paths | Implemented |
| HR demo | `demos/hr_policy/`, generic pipeline | end-to-end CLI run | `artifacts/hr_policy_assistant/` | 20 cases, 20 passed |
| Marine procurement demo | `demos/marine_procurement/`, generic pipeline | end-to-end CLI run | `artifacts/marine_ship_spares_procurement_assistant/` | 20 cases, 20 passed |
| Generalization test | `demos/generalization/`, generic pipeline | end-to-end CLI run | `artifacts/university_laboratory_safety_assistant/` | 20 cases, 20 passed |
| Streamlit interface | `app.py` | Streamlit `AppTest` smoke run | `streamlit run app.py` | Implemented; smoke test had zero exceptions |
| Security/configuration/docs | `.env.example`, `.gitignore`, `forgeai/config.py`, README | path, calculator, validation tests | README, demo script, limitations | Implemented; external provider/network not live-tested |
