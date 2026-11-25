# Evaluation case generation prompt

ForgeAI generates evaluation cases from the active document chunks, rather than reusing a domain-wide question list. This prompt is stored with every generated agent for reproducibility and as the intended contract for an optional LLM-assisted generator:

> Given the current agent requirements and only the supplied source passages, create distinct evaluation cases. Include direct facts, multi-section and multi-document synthesis, ambiguous wording, incomplete evidence, out-of-scope requests, and hallucination-resistance/refusal cases. Phrase questions using the topic or entity, without copying the expected answer or numeric facts into the question. Each answerable case must quote a verifiable supporting source and list required facts; each unanswerable case must expect refusal and must not assert an unsupported answer. Do not invent facts. Reject duplicates and cases without usable evidence. Return schema-valid JSON.

The built-in offline implementation is deterministic and creates source-linked factual, multi-document, multi-section, ambiguous-topic, explicit-gap/refusal, out-of-scope, and hallucination-resistance cases when the source collection supports those categories. It records unavailable category coverage honestly and fills the suite only with additional real evidence cases.
