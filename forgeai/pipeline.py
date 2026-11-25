from __future__ import annotations
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from .evaluation.cases import generate_cases, save_cases
from .evaluation.judge import score
from .evaluation.optimizer import optimize, run_suite
from .generator import generate
from .groundability import analyze
from .ingestion import ingest
from .intake import intake
from .models import EvalCase
from .runner import AgentRunner

logger = logging.getLogger("forgeai")


def _write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _md_report(report: dict) -> str:
    lines = [f"# Final report: {report['agent_name']}", "", f"Generated: {report['generated_at']}", "",
        "## Evaluation", f"- Evaluation cases: {report['evaluation']['case_count']}", f"- Initial pass rate: {report['initial_score']['pass_rate']:.1%}",
        f"- Final pass rate: {report['final_score']['pass_rate']:.1%}", f"- Judge: {report['final_score']['method']}", "",
        "## Groundability", f"- Documents: {', '.join(report['document_summary']['files'])}",
        f"- Supported headings: {', '.join(report['groundability']['supported_topics']) or 'None detected'}", "",
        "## Optimization", f"- Iterations: {len(report['optimization_logs'])}"]
    for log in report["optimization_logs"]:
        lines += [f"- Iteration {log['iteration']}: {log['initial_score']:.1%} to {log['final_score']:.1%}; {', '.join(log['failure_classifications']) or 'no categories'}"]
    remaining = [f"- {item['test_id']}: {item['failure_classification']}" for item in report["final_score"]["failures"]] or ["- None"]
    lines += ["", "## Remaining failures", *remaining, "", "## Limitations", *[f"- {item}" for item in report["known_limitations"]], ""]
    return "\n".join(lines)


def build_agent(brief: str, documents: Path, artifacts_root: Path = Path("artifacts"), unattended: bool = True,
                clarifications: list[str] | None = None, minimum_cases: int = 20, iterations: int = 3, top_k: int = 5) -> dict:
    docs = documents.resolve()
    chunks = ingest(docs)
    req, questions = intake(brief, docs, clarifications, unattended)
    ground = analyze(chunks)
    req.supported_topics = ground["supported_topics"] or req.supported_topics
    req.unsupported_topics = [*ground["unsupported_topics"], "Topics not evidenced by the supplied collection"]
    req.known_limitations = ["Groundability topics are inferred from document structure.", "Deterministic lexical checks are primary; the optional semantic judge is reported separately."]
    if os.getenv("FORGEAI_PROVIDER", "offline").strip().lower() != "ollama":
        req.known_limitations.append("Semantic LLM judging was not enabled for this run.")
    out = (artifacts_root / req.agent_id).resolve()
    collection_reference = os.path.relpath(docs, out).replace("\\", "/")
    agent = generate(req, collection_reference, top_k=top_k)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Building agent %s from %d documents", req.agent_id, len(req.available_documents))
    _write(out / "requirements.json", req.model_dump())
    _write(out / "groundability.json", ground)
    _write(out / "agent_definition.json", agent.model_dump())
    _write(out / "intake_questions.json", {"questions": questions, "answers": clarifications or [], "unattended": unattended})
    project_root = Path(__file__).resolve().parent.parent
    (out / "evaluation_generation_prompt.md").write_text((project_root / "prompts" / "evaluation_generation.md").read_text(encoding="utf-8"), encoding="utf-8")
    cases = generate_cases(chunks, minimum_cases)
    save_cases(cases, out / "evals" / "cases.json")
    runner = AgentRunner(agent, docs)
    initial_results = run_suite(runner, cases)
    initial = score(initial_results)
    logger.info("Initial evaluation for %s: %d/%d cases passed", req.agent_id, initial["pass_count"], initial["case_count"])
    _write(out / "initial_evaluation.json", initial)
    logs, optimized = optimize(agent, chunks, cases, docs, iterations)
    _write(out / "optimization_logs.json", logs)
    _write(out / "agent_definition.json", optimized.model_dump())
    final_results = run_suite(AgentRunner(optimized, docs), cases)
    final = score(final_results)
    logger.info("Final evaluation for %s: %d/%d cases passed after %d optimizer iterations", req.agent_id, final["pass_count"], final["case_count"], len(logs))
    _write(out / "final_scorecard.json", final)
    report = {"agent_id": req.agent_id, "agent_name": req.agent_name, "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": req.model_dump(), "agent_definition": optimized.model_dump(), "document_summary": {"files": req.available_documents, "chunk_count": len(chunks)},
        "evaluation": {"case_count": len(cases), "case_file": "evals/cases.json"}, "initial_score": initial, "optimization_logs": logs,
        "final_score": final, "failure_classifications": {name: sum(1 for r in final_results if r.failure_classification == name) for name in ("Retrieval miss", "Prompt defect", "Hallucination", "Wrong refusal")},
        "groundability": ground, "optimization_decisions": [l["changes_made"] for l in logs], "known_limitations": req.known_limitations,
        "final_recommendation": "Suitable for review and local demonstration; human review is required before operational use."}
    _write(out / "final_report.json", report)
    (out / "final_report.md").write_text(_md_report(report), encoding="utf-8")
    return report


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list): raise ValueError("Evaluation cases must be a JSON list")
    cases = [EvalCase.model_validate(item) for item in data]
    if len({case.test_id for case in cases}) != len(cases): raise ValueError("Evaluation case IDs must be unique")
    if len({case.question.casefold().strip() for case in cases}) != len(cases): raise ValueError("Evaluation questions must be unique")
    return cases
