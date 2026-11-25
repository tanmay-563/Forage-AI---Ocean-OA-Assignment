# Evaluation case generation contract

The current generator is deterministic and does not call an LLM. It creates factual retrieval questions only from qualifying source sentences or named CSV row fields, and validates quotations against exact ingested chunks. Where at least two distinct, non-boilerplate terms connect evidence in separate document sections or files, it may add a comparison case with separate provenance for each passage. Numeric CSV comparison cases are created only when both operands and a common currency are present; arithmetic and the lower-valued record are rechecked against the cited rows.

PDF and prose candidates must pass minimum-length, sentence-ending, and recognizable-predicate checks. CSV candidates are validated as named, non-empty row fields and retain their source row. Headers, fragments, duplicates, near-duplicate questions, invalid references, and unsupported facts are rejected. Missing-information cases state a refusal expectation explicitly and are counted separately from factual cases.

Category coverage depends on the actual collection. The generator does not force ambiguity, conflicts, numerical calculations, multi-document coverage, or any other unsupported category. Every completed build requires at least 20 validated evidence-backed cases plus separate refusal cases; the minimum is never reduced or filled with invented material.
