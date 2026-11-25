# Known limitations

- The inspected environment has no Ollama installation or model. No model was downloaded. Default answers are explicit extractive excerpts and abstentions, not synthesized LLM answers.
- Ollama provider and an optional semantic judge are implemented for answer/evaluation runs, but no live Ollama model was available during verification. LLM-assisted intake/case generation is not available.
- Topic coverage is inferred from Markdown headings. It cannot reliably identify nuanced partial coverage or contradictions.
- Evaluation questions are primarily derived from source sentences, with multi-document, multi-section, ambiguity, and refusal cases added where the collection supports them. A single-file collection cannot demonstrate cross-document synthesis, and not every domain has an ambiguous topic such as annual versus sick leave.
- Deterministic judge uses lexical matching and cannot robustly assess semantic paraphrases. Optional semantic judging is a single-model rubric score, not a calibrated unbiased judge; no pairwise position-bias experiment is performed.
- Optimizer supports simple targeted changes to top-k and prompt clauses. It does not tune embedding models, chunking, or query rewrites. Candidate regressions are logged and rolled back.
- The weather tool calls Open-Meteo and therefore needs network access; tests mock the HTTP call. It is attached only when the brief requests weather.
- PDF extraction requires PyMuPDF. The source PDF helper uses only Python's standard library; PDF reading still requires the declared dependency.
- The synthetic HR, marine, and laboratory documents are not real legal, supplier, engineering, or safety advice.
- Streamlit stores uploaded collections in `.forgeai_uploads/` so generated agent definitions remain runnable. The user should remove that ignored directory manually when no longer needed.
