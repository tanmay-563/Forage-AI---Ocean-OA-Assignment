from __future__ import annotations
from ..models import AgentDefinition, EvalCase
from ..runner import AgentRunner
from .judge import evaluate, score
from .semantic_judge import OllamaSemanticJudge
from ..config import Settings
import re


def _settings_snapshot(definition):
    return {"system_prompt": definition.system_prompt,
        "chunk_size": definition.chunking.get("chunk_chars", definition.chunking.get("chunk_size")),
        "chunk_overlap": definition.chunking.get("overlap", definition.chunking.get("chunk_overlap")),
        "top_k": definition.retrieval.get("top_k"),
        "retrieval_threshold": definition.retrieval.get("minimum_score"),
        "reranking": definition.retrieval.get("reranking"),
        "refusal_instructions": definition.refusal_policy,
        "output_contract": definition.output_contract}


def _holdout_split(cases):
    factual = [c for c in cases if c.answerability_classification == "evidence_backed"]
    if len(factual) < 25:
        return list(cases), [], "Not enough factual cases for a useful holdout while retaining 20 optimization cases."
    holdout_ids = {case.test_id for index, case in enumerate(sorted(factual, key=lambda c: c.test_id)) if index % 5 == 0}
    return [c for c in cases if c.test_id not in holdout_ids], [c for c in cases if c.test_id in holdout_ids], None


def _failure_record(result, case):
    expected = case.expected_evidence
    retrieved = [str(item.get("text", "")) for item in result.evidence_used]
    joined = " ".join(retrieved).casefold()
    return {"failed_test_id": result.test_id, "original_question": case.question,
        "agent_response": result.answer, "expected_behavior": case.expected_behavior,
        "retrieved_evidence": result.evidence_used, "expected_evidence": expected,
        "exact_evidence_mismatch": [snippet for snippet in expected if snippet.casefold() not in joined],
        "failure_classification": result.failure_classification,
        "root_cause": result.judge_notes or ["No evidence-backed answer met the deterministic checks."]}


def run_suite(runner: AgentRunner, cases: list[EvalCase]):
    settings = Settings.from_env()
    semantic = None
    if settings.judge_provider == "ollama" and settings.judge_model:
        generation_model = settings.ollama_model if runner.provider.name == "ollama" else None
        semantic = OllamaSemanticJudge(settings.ollama_url, settings.judge_model, settings.request_timeout, generation_model)
    return [evaluate(case, runner.answer(case.question), semantic) for case in cases]


def optimize_with_report(definition: AgentDefinition, chunks, cases: list[EvalCase], collection, max_iterations: int = 3):
    if not 0 <= max_iterations <= 3: raise ValueError("Optimizer iteration limit must be between zero and three")
    current = definition.model_copy(deep=True)
    optimization_cases, holdout_cases, holdout_limitation = _holdout_split(cases)
    initial_results = run_suite(AgentRunner(current, collection), optimization_cases)
    initial = score(initial_results)
    logs = []
    previous_rate = initial["pass_rate"]
    for iteration in range(1, max_iterations + 1):
        before_settings = _settings_snapshot(current)
        current_results = run_suite(AgentRunner(current, collection), optimization_cases)
        failed = [r for r in current_results if not r.passed]
        if not failed: break
        classes = sorted({r.failure_classification for r in failed if r.failure_classification})
        failed_ids = {r.test_id for r in failed}
        related = [case for case in optimization_cases if case.test_id in failed_ids]
        candidate = current.model_copy(deep=True)
        changes = []
        if "Retrieval miss" in classes:
            old = int(candidate.retrieval.get("top_k", 5)); new = min(old + 2, 12)
            if new != old:
                candidate.retrieval["top_k"] = new
                changes.append("Increased top_k to target missing expected document sources while preserving the refusal threshold.")
        if "Prompt defect" in classes or "Hallucination" in classes:
            instruction = "Preserve complete evidence statements, do not infer unstated facts, and cite the exact supporting source."
            if instruction not in candidate.system_prompt:
                candidate.system_prompt += " " + instruction
                changes.append("Strengthened evidence and citation instruction after answer-criteria failures.")
        if "Wrong refusal" in classes:
            instruction = "Answer when retrieved evidence directly supports the question; reserve refusal for missing or conflicting evidence."
            if instruction not in candidate.system_prompt:
                candidate.system_prompt += " " + instruction
                changes.append("Clarified the answer-versus-refusal boundary after observed wrong refusals.")
        if not changes: break
        candidate.metadata["optimization_iteration"] = iteration
        result = run_suite(AgentRunner(candidate, collection), optimization_cases)
        final = score(result)
        case_by_id = {case.test_id: case for case in optimization_cases}
        after_settings = _settings_snapshot(candidate)
        logs.append({"iteration": iteration, "initial_score": previous_rate, "before_score": previous_rate,
            "failed_test_ids": sorted(failed_ids), "failure_classifications": classes,
            "failure_details": [_failure_record(result_item, case_by_id[result_item.test_id]) for result_item in failed],
            "root_cause_analysis": [f"{cls}: {sum(1 for r in failed if r.failure_classification == cls)} case(s)" for cls in classes],
            "changes_made": changes, "reason_for_changes": "Changes target the observed failure categories; identical validated cases were rerun.",
            "original_settings": before_settings, "modified_settings": after_settings,
            "after_score": final["pass_rate"], "final_score": final["pass_rate"], "scorecard": final,
            "regressions": [r.test_id for r in result if not r.passed and r.test_id not in failed_ids],
            "remaining_failures": [r.test_id for r in result if not r.passed], "accepted": final["pass_rate"] > previous_rate,
            "rollback_reason": None if final["pass_rate"] > previous_rate else "Candidate did not improve optimization-set pass rate; previous best definition retained.",
            "remaining_limitations": ["Lexical retrieval and deterministic scoring limit semantic assessment."]})
        if final["pass_rate"] <= previous_rate: break
        current = candidate
        previous_rate = final["pass_rate"]
    # Holdout results are evaluated after change selection and never affect acceptance or rollback.
    if holdout_cases:
        baseline_holdout = score(run_suite(AgentRunner(definition, collection), holdout_cases))
        final_holdout = score(run_suite(AgentRunner(current, collection), holdout_cases))
    else:
        baseline_holdout = final_holdout = None
    report = {"maximum_iterations": max_iterations, "iterations_run": len(logs),
        "optimization_case_ids": [case.test_id for case in optimization_cases],
        "holdout_case_ids": [case.test_id for case in holdout_cases],
        "optimization_initial_score": initial, "optimization_final_score": score(run_suite(AgentRunner(current, collection), optimization_cases)),
        "holdout_baseline_score": baseline_holdout, "holdout_final_score": final_holdout,
        "holdout_used_for_selection": False, "holdout_limitation": holdout_limitation,
        "case_ids_preserved_across_iterations": True}
    return logs, current, report


def optimize(definition: AgentDefinition, chunks, cases: list[EvalCase], collection, max_iterations: int = 3) -> tuple[list[dict], list]:
    """Compatibility wrapper; detailed report is consumed by the build pipeline."""
    logs, best, _ = optimize_with_report(definition, chunks, cases, collection, max_iterations)
    return logs, best
