# Decisions

1. **Python-first package and CLI.** The brief makes CLI mandatory and UI secondary; avoid infrastructure and keep components directly testable.
2. **No automatic model download.** Ollama was not found, and 15.7 GiB RAM does not justify silently selecting/downloading a model.
3. **Explicit offline extractive mode.** A no-key default must not fabricate conversational model behavior. It will quote retrieved evidence and abstain when evidence is insufficient.
4. **Lightweight lexical retrieval baseline.** Avoid large embedding downloads. Optional embedding dependencies may be activated when already installed/model cached.
5. **PDF extraction is an optional dependency.** PyMuPDF was absent at inspection; do not silently claim PDF runtime verification until available.
6. **Synthetic demos share the same pipeline.** Domain variation lives in document content, briefs, and generated artifacts.
7. **Transparent deterministic scoring first.** Semantic judging may be enabled through a configured provider; results and uncertainty are kept separate.
