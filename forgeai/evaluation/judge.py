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
    expected_locations = case.evidence_provenance
    if not expected_locations and case.chunk_id:
        expected_locations = [{"source_filename": case.source_filename or case.supporting_source, "chunk_id": case.chunk_id,
            "page_number": case.page_number, "row_number": case.row_number}]
    citation_valid = bool(refs) and (not expected_sources or all(source in cited_sources for source in expected_sources))
    for expected_ref in expected_locations:
        source = expected_ref.source_filename if hasattr(expected_ref, "source_filename") else expected_ref.get("source_filename")
        chunk_id = expected_ref.chunk_id if hasattr(expected_ref, "chunk_id") else expected_ref.get("chunk_id")
        page = expected_ref.page_number if hasattr(expected_ref, "page_number") else expected_ref.get("page_number")
        row = expected_ref.row_number if hasattr(expected_ref, "row_number") else expected_ref.get("row_number")
        citation_valid = citation_valid and any(ref.get("source") == source and ref.get("chunk_id") == chunk_id and
            ref.get("page") == page and ref.get("row") == row for ref in refs)
    citation = 1.0 if citation_valid or (not case.answerable and not refs and not expected_sources) else 0.0
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
    semantic = None
    if semantic_judge:
        try:
            semantic = semantic_judge(case, output)
            semantic_scores = semantic["scores"]
            notes.extend(str(note) for note in semantic.get("notes", []))
            deterministic = {"groundedness": grounded, "semantic_correctness": correctness,
                "citation_correctness": citation, "refusal_correctness": refusal, "completeness": completeness}
            disagreements = [f"{key}: deterministic={value:.2f}, semantic={semantic_scores[key]:.2f}" for key, value in deterministic.items() if key in semantic_scores and abs(value - semantic_scores[key]) >= 0.4]
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
    writing_quality = 1.0 if answer.strip() and len(answer.split()) <= 80 else (0.5 if answer.strip() else 0.0)
    judge_meta = {key: semantic[key] for key in ("order_swapped_result", "same_model_judging", "evaluator_model", "independent_evaluator") if semantic and key in semantic}
    decision = "pass" if passed else "fail"
    deterministic_checks = {"answer_present": bool(answer.strip()), "expected_facts": fact_scores,
        "groundedness": grounded, "evidence_quality": evidence_quality, "citation_correctness": citation,
        "refusal_correctness": refusal, "completeness": completeness, "writing_quality": writing_quality,
        "expected_sources_present": not missing_sources}
    explanation = "; ".join(notes) if notes else ("All deterministic evidence and answer checks passed." if passed else "One or more deterministic answer or provenance checks failed.")
    return EvalResult(test_id=case.test_id, question=case.question, answer=answer, sources=refs, groundedness=grounded,
        correctness=correctness, evidence_quality=evidence_quality, citation_correctness=citation, refusal_appropriateness=refusal,
        completeness=completeness, instruction_compliance=1.0 if answer else 0.0, passed=passed,
        failure_classification=cls, judge_notes=notes, semantic_scores=semantic_scores, semantic_judge_error=semantic_error, judge_disagreements=disagreements,
        writing_quality=writing_quality, deterministic_checks=deterministic_checks,
        evidence_used=retrieved_evidence, judge_explanation=explanation,
        llm_judge_result=semantic_scores, order_swapped_result=judge_meta.get("order_swapped_result"),
        evaluator_metadata={key: judge_meta[key] for key in ("same_model_judging", "evaluator_model", "independent_evaluator") if key in judge_meta},
        final_decision=decision)


def score(results: list[EvalResult]) -> dict:
    dimensions = ["groundedness", "correctness", "evidence_quality", "citation_correctness", "refusal_appropriateness", "completeness", "instruction_compliance", "writing_quality"]
    semantic = [r.semantic_scores for r in results if r.semantic_scores is not None]
    semantic_dimensions = sorted(set.intersection(*(set(item) for item in semantic))) if semantic else []
    semantic_summary = {d: round(sum(item[d] for item in semantic) / len(semantic), 4) for d in semantic_dimensions} if semantic else None
    return {"case_count": len(results), "pass_count": sum(r.passed for r in results), "pass_rate": round(sum(r.passed for r in results) / max(1, len(results)), 4),
            "dimensions": {d: round(sum(getattr(r, d) for r in results) / max(1, len(results)), 4) for d in dimensions},
            "semantic_dimensions": semantic_summary, "semantic_judge_cases": len(semantic),
            "semantic_judging_status": "available" if semantic else "unavailable; deterministic checks only",
            "independent_evaluator_cases": sum(bool(r.evaluator_metadata.get("independent_evaluator")) for r in results),
            "same_model_judge_cases": sum(bool(r.evaluator_metadata.get("same_model_judging")) for r in results),
            "order_swap_results": [{"test_id": r.test_id, **r.order_swapped_result} for r in results if r.order_swapped_result],
            "judge_disagreements": [{"test_id": r.test_id, "differences": r.judge_disagreements} for r in results if r.judge_disagreements],
            "semantic_judge_errors": [{"test_id": r.test_id, "error": r.semantic_judge_error} for r in results if r.semantic_judge_error],
            "failures": [r.model_dump() for r in results if not r.passed], "method": "deterministic primary scoring; optional Ollama semantic scores are reported separately and do not override deterministic safety checks"}
