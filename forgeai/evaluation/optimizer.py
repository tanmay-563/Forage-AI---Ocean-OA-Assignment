from __future__ import annotations
from ..models import AgentDefinition, EvalCase
from ..runner import AgentRunner
from .judge import evaluate, score
from .semantic_judge import OllamaSemanticJudge
from ..config import Settings


def run_suite(runner: AgentRunner, cases: list[EvalCase]):
    settings = Settings.from_env()
    semantic = OllamaSemanticJudge(settings.ollama_url, settings.ollama_model, settings.request_timeout) if runner.provider.name == "ollama" else None
    return [evaluate(case, runner.answer(case.question), semantic) for case in cases]


def optimize(definition: AgentDefinition, chunks, cases: list[EvalCase], collection, max_iterations: int = 3) -> tuple[list[dict], list]:
    if not 0 <= max_iterations <= 3: raise ValueError("Optimizer iteration limit must be between zero and three")
    current = definition.model_copy(deep=True)
    initial_results = run_suite(AgentRunner(current, collection), cases)
    initial = score(initial_results)
    logs = []
    previous_rate = initial["pass_rate"]
    for iteration in range(1, max_iterations + 1):
        failed = [r for r in run_suite(AgentRunner(current, collection), cases) if not r.passed]
        if not failed: break
        classes = sorted({r.failure_classification for r in failed if r.failure_classification})
        failed_ids = {r.test_id for r in failed}
        related = [case for case in cases if case.test_id in failed_ids]
        candidate = current.model_copy(deep=True)
        changes = []
        if "Retrieval miss" in classes:
            old = int(candidate.retrieval.get("top_k", 5)); candidate.retrieval["top_k"] = min(old + 2, 12)
            changes.append("Increased top_k to target missing expected document sources while preserving the refusal threshold.")
        if "Prompt defect" in classes or "Hallucination" in classes:
            candidate.system_prompt += " Preserve complete evidence statements, do not infer unstated facts, and cite the exact supporting source."
            changes.append("Strengthened evidence and citation instruction after answer-criteria failures.")
        if "Wrong refusal" in classes:
            candidate.system_prompt += " Answer when retrieved evidence directly supports the question; reserve refusal for missing or conflicting evidence."
            changes.append("Clarified the answer-versus-refusal boundary after observed wrong refusals.")
        if not changes: break
        candidate.metadata["optimization_iteration"] = iteration
        result = run_suite(AgentRunner(candidate, collection), cases)
        final = score(result)
        logs.append({"iteration": iteration, "initial_score": previous_rate, "failed_test_ids": sorted(failed_ids), "failure_classifications": classes,
            "root_cause_analysis": [f"{cls}: {sum(1 for r in failed if r.failure_classification == cls)} case(s)" for cls in classes],
            "changes_made": changes, "reason_for_changes": "Changes target the observed failure categories; identical validated cases were rerun.",
            "final_score": final["pass_rate"], "scorecard": final, "regressions": [r.test_id for r in result if not r.passed and r.test_id not in failed_ids],
            "remaining_failures": [r.test_id for r in result if not r.passed], "accepted": final["pass_rate"] > previous_rate,
            "rollback_reason": None if final["pass_rate"] > previous_rate else "Candidate did not improve pass rate; previous best definition retained."})
        if final["pass_rate"] <= previous_rate: break
        current = candidate
        previous_rate = final["pass_rate"]
    return logs, current
