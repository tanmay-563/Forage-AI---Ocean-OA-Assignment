from __future__ import annotations
import re
from ..models import EvalCase, EvalResult


def evaluate(case: EvalCase, output: dict, semantic_judge=None) -> EvalResult:
    answer = str(output.get("answer", ""))
    answer_low = answer.casefold()
    refs = output.get("sources", [])
    retrieved_evidence = output.get("evidence", [])
    evidence_text = "\n".join(str(item.get("text", "")) for item in retrieved_evidence).casefold()
    is_refusal = any(term in answer_low for term in ("cannot answer", "cannot verify", "not enough evidence", "insufficient evidence", "don't have", "does not specify", "does not define", "does not state", "does not guarantee", "does not promise", "not stated", "not specified", "not be interpreted"))
    expected = case.expected_facts
    fact_scores = []
    for fact in expected:
        norm = fact.casefold().strip()
        if norm in answer_low:
            fact_scores.append(1.0)
        else:
            words = {w for w in re.findall(r"[a-z0-9]+", norm) if len(w) > 3}
            answer_words = set(re.findall(r"[a-z0-9]+", answer_low))
            fact_scores.append(len(words & answer_words) / max(1, len(words)))
    correctness = sum(fact_scores) / len(fact_scores) if fact_scores else (1.0 if case.refusal_expected and is_refusal else (0.8 if answer and refs else 0.0))
    # A passage may mention an unrelated limitation alongside a supported fact.
    # That quoted sentence must not turn a correct positive answer into a refusal.
    if case.answerable and expected and correctness >= 0.60:
        is_refusal = False
    expected_sources = case.supporting_sources or ([case.supporting_source] if case.supporting_source else [])
    cited_sources = {r.get("source") for r in refs}
    missing_sources = case.answerable and bool(expected_sources) and not all(source in cited_sources for source in expected_sources)
    citation = 1.0 if (refs and (not expected_sources or all(source in cited_sources for source in expected_sources))) or (not case.answerable and not refs and not expected_sources) else 0.0
    if case.expected_evidence:
        evidence_hits = sum(1 for snippet in case.expected_evidence if snippet.casefold() in evidence_text)
        evidence_quality = evidence_hits / len(case.expected_evidence)
    else:
        evidence_quality = 1.0 if (refs or (case.refusal_expected and is_refusal)) else 0.0
    refusal = 1.0 if is_refusal == case.refusal_expected else 0.0
    grounded = 1.0 if (refs and citation) or (is_refusal and not refs and case.supporting_source is None) else 0.0
    completeness = correctness if expected else (1.0 if case.refusal_expected and is_refusal else (1.0 if answer and refs else 0.0))
    passed = bool(answer.strip()) and correctness >= 0.60 and evidence_quality >= 0.60 and citation == 1 and refusal == 1 and grounded == 1 and completeness >= 0.60
    cls = None
    notes = []
    if missing_sources:
        cls, notes = "Retrieval miss", ["One or more expected document sources were not retrieved."]
    elif is_refusal and case.answerable and refs:
        cls, notes = "Wrong refusal", ["Evidence was retrieved for an answerable case, but the response refused."]
    elif is_refusal and case.answerable and not refs:
        cls, notes = "Retrieval miss", ["No evidence source was returned for an answerable case."]
    elif not refs and case.answerable:
        cls, notes = "Hallucination", ["The response answered without any retrieved evidence."]
    elif not is_refusal and not case.answerable:
        cls, notes = "Hallucination", ["The response answered a case that should be refused."]
    elif correctness < .60:
        cls, notes = "Prompt defect", ["Retrieved evidence did not satisfy the expected fact criteria."]
    elif not citation:
        cls, notes = "Prompt defect", ["Citation did not identify the expected source."]
    elif evidence_quality < 0.60:
        cls, notes = "Retrieval miss", ["The retrieved passages did not include the expected supporting evidence."]
    elif grounded < 0.60:
        cls, notes = "Hallucination", ["The response was not adequately grounded in cited evidence."]
    elif not passed and cls is None:
        cls, notes = "Prompt defect", ["The response failed one or more output-quality requirements."]
    semantic_scores = None
    semantic_error = None
    disagreements = []
    if semantic_judge:
        try:
            semantic = semantic_judge(case, output)
            semantic_scores = semantic["scores"]
            notes.extend(str(note) for note in semantic.get("notes", []))
            deterministic = {"groundedness": grounded, "correctness": correctness, "evidence_quality": evidence_quality,
                "citation_correctness": citation, "refusal_appropriateness": refusal, "completeness": completeness}
            disagreements = [f"{key}: deterministic={value:.2f}, semantic={semantic_scores[key]:.2f}" for key, value in deterministic.items() if abs(value - semantic_scores[key]) >= 0.4]
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
    return EvalResult(test_id=case.test_id, question=case.question, answer=answer, sources=refs, groundedness=grounded,
        correctness=correctness, evidence_quality=evidence_quality, citation_correctness=citation, refusal_appropriateness=refusal,
        completeness=completeness, instruction_compliance=1.0 if answer else 0.0, passed=passed,
        failure_classification=cls, judge_notes=notes, semantic_scores=semantic_scores, semantic_judge_error=semantic_error, judge_disagreements=disagreements)


def score(results: list[EvalResult]) -> dict:
    dimensions = ["groundedness", "correctness", "evidence_quality", "citation_correctness", "refusal_appropriateness", "completeness", "instruction_compliance"]
    semantic = [r.semantic_scores for r in results if r.semantic_scores is not None]
    semantic_summary = {d: round(sum(item[d] for item in semantic) / len(semantic), 4) for d in dimensions} if semantic else None
    return {"case_count": len(results), "pass_count": sum(r.passed for r in results), "pass_rate": round(sum(r.passed for r in results) / max(1, len(results)), 4),
            "dimensions": {d: round(sum(getattr(r, d) for r in results) / max(1, len(results)), 4) for d in dimensions},
            "semantic_dimensions": semantic_summary, "semantic_judge_cases": len(semantic),
            "judge_disagreements": [{"test_id": r.test_id, "differences": r.judge_disagreements} for r in results if r.judge_disagreements],
            "semantic_judge_errors": [{"test_id": r.test_id, "error": r.semantic_judge_error} for r in results if r.semantic_judge_error],
            "failures": [r.model_dump() for r in results if not r.passed], "method": "deterministic primary scoring; optional Ollama semantic scores are reported separately and do not override deterministic safety checks"}
