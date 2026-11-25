# ForgeAI project plan

## Objective
Build a local-first, generic agent factory that converts a natural-language brief and document folder into a validated agent definition, runnable grounded agent, generated evaluation suite, optimization history, and honest report.

## Environment findings
- Windows 10 Home Single Language; Python 3.11.6; Node.js 22.15.0; npm 10.9.2; Git 2.44.0.
- Available RAM: 15.7 GiB (reported by psutil). Free disk on C: 263.4 GiB.
- Ollama is not installed/on PATH; no model downloads will be initiated.
- Python packages available at inspection: Pydantic, pytest, pandas, FAISS, sentence-transformers, Streamlit, FastAPI, httpx. PyMuPDF and pypdf were absent.
- Existing workspace was empty and not a Git repository.
- Windows CIM access was denied; OS memory value was cross-checked with psutil. Listener-port inspection returned no usable data in the sandbox.

## Delivery sequence
1. Foundation: package layout, config, validated models, provider abstraction, project docs.
2. Ingestion/retrieval: TXT, Markdown, CSV, optional PyMuPDF PDF extraction, chunk metadata, lightweight lexical retrieval with optional embeddings.
3. Intake, requirements, groundability, and generated declarative agent.
4. Safe registered tools and generic chat runner.
5. Evaluation prompts, dynamic cases, deterministic/hybrid judge, bounded optimizer, reports.
6. Synthetic HR and marine collections, plus a distinct lab-safety generalization case.
7. Automated tests and a lightweight Streamlit UI if the CLI workflow is sound.
8. Documentation, traceability, and final audit.

## Constraints and honest status
- No paid service or API key is required. Ollama is an optional provider; offline extractive answering is explicitly labeled and uses retrieved text only.
- PDF extraction depends on PyMuPDF. If it cannot be installed in the current environment, PDF support will be implemented and declared as a dependency, with the limitation stated until verified.
- Scores must come from executed cases. Optimization changes are bounded and only applied when linked to observed failure classifications.
- Public publishing/deployment is out of scope unless explicitly authorized.

## Latest status
- Core CLI, all three workflow builds, saved artifact sets, the UI smoke check, and the automated test suite are complete.
- The HR, marine, and generalization suites each contain 20 cases and currently pass 20/20 under the deterministic/extractive runner.
- HR starts at 16/20 with `--top-k 1` and improves to 20/20 after a traceable retrieval adjustment; the other two complete suites need no optimization changes.
- Live Ollama execution and expert semantic review remain unverified.
