# Architecture

ForgeAI is a local Python package with a generic pipeline. A brief plus a document directory flows through intake/requirements, ingestion and groundability, declarative definition generation, generic retrieval-backed execution, generated evaluation, scoring, bounded optimization, and report generation. Demo domains are data folders and briefs; they do not have separate application code.

```text
brief + documents
      -> intake -> requirements + groundability -> agent_definition.json
      -> ingestion -> chunks/index -> generic runner -> answers + source references
      -> dynamic eval cases -> deterministic judge (+ optional LLM judge)
      -> <=3 targeted iterations -> scorecards + report
```

## Key modules
- `forgeai/models.py`: validated data contracts.
- `forgeai/ingestion.py`, `retrieval.py`: extraction, metadata-preserving chunks, local retrieval.
- `forgeai/intake.py`, `groundability.py`, `generator.py`: generic factory stages.
- `forgeai/providers.py`: Ollama HTTP provider and explicit offline extractive provider.
- `forgeai/tools/`: allowlisted executable tools with input validation.
- `forgeai/evaluation/`: dynamic cases, deterministic/hybrid judge, critic, optimizer.
- `forgeai/runner.py`: loads any valid definition; no domain-specific branching.
- `forgeai/pipeline.py`, `cli.py`: complete workflow and commands.
- `app.py`: lightweight Streamlit client of the same pipeline.

Uploaded content is treated as untrusted evidence. It cannot replace system instructions. Only files under the declared collection root are ingested or exposed to CSV lookup. No arbitrary Python execution is used.

## Retrieval and model behavior
Retrieval is local and lightweight. Sentence-transformers/FAISS are optional accelerators; lexical retrieval is the no-download fallback. Ollama is optional. Offline mode is explicitly extractive and returns evidence excerpts or abstains; it does not pretend to be generative. The judge reports deterministic checks separately from optional semantic model review.
