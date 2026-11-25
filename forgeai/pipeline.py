from __future__ import annotations
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from .evaluation.cases import (InsufficientEvidenceError, generate_cases_with_report,
                               save_cases, validate_case_diversity, validate_case_provenance)
from .evaluation.judge import score
from .evaluation.optimizer import optimize_with_report, run_suite
from .generator import generate
from .groundability import analyze
from .ingestion import diagnose_collection, ingest
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
        "## Evidence quality", f"- Evidence-backed cases: {report['evaluation']['valid_evidence_backed_cases']}",
        f"- Unsupported/refusal cases: {report['evaluation']['valid_unsupported_refusal_cases']}",
        f"- Rejected cases: {report['evaluation']['rejected_case_count']}",
        f"- Provenance references validated: {report['evaluation'].get('evidence_provenance_validated', 0)}",
        f"- Topic diversity: {report['evaluation']['topic_diversity']}",
        f"- Document coverage: {', '.join(report['evaluation']['document_coverage']) or 'None'}",
        f"- Case categories: {', '.join(f'{name} ({count})' for name, count in report['evaluation'].get('category_coverage', {}).items()) or 'None'}",
        "", "## Optimization", f"- Iterations: {len(report['optimization_logs'])}"]
    for log in report["optimization_logs"]:
        lines += [f"- Iteration {log['iteration']}: {log['initial_score']:.1%} to {log['final_score']:.1%}; {', '.join(log['failure_classifications']) or 'no categories'}"]
    optimization = report.get("optimization_report", {})
    opt_initial, opt_final = optimization.get("optimization_initial_score"), optimization.get("optimization_final_score")
    if opt_initial and opt_final:
        lines.append(f"- Optimization set: {opt_initial['pass_rate']:.1%} to {opt_final['pass_rate']:.1%} ({opt_final['case_count']} cases).")
    holdout_initial, holdout_final = optimization.get("holdout_baseline_score"), optimization.get("holdout_final_score")
    if holdout_initial and holdout_final:
        lines.append(f"- Holdout: {holdout_initial['pass_rate']:.1%} to {holdout_final['pass_rate']:.1%} ({holdout_final['case_count']} cases; not used for selection).")
    elif optimization.get("holdout_limitation"):
        lines.append(f"- Holdout: unavailable. {optimization['holdout_limitation']}")
    diagnostic_lines = []
    for document in report.get("source_diagnostics", {}).get("documents", []):
        if document.get("page_count") is not None:
            pages = document.get("pages", [])
            diagnostic_lines.append(f"- {document['source']}: {document['page_count']} pages; "
                f"{sum(page['extracted_characters'] for page in pages)} extracted characters; "
                f"{sum(page['qualifying_sentences'] for page in pages)} qualifying sentences; "
                f"{sum(not page['extracted_characters'] for page in pages)} empty-text pages")
        else:
            detail = document.get("text_diagnostics")
            records = document.get("records", [])
            units = detail["qualifying_sentences"] if detail else sum(row["qualifying_sentences"] for row in records)
            chars = detail["extracted_characters"] if detail else sum(row["extracted_characters"] for row in records)
            diagnostic_lines.append(f"- {document['source']}: {chars} extracted characters; {units} qualifying evidence units")
    lines += ["", "## Source diagnostics", *(diagnostic_lines or ["- No supported source files diagnosed."])]
    remaining = [f"- {item['test_id']}: {item['failure_classification']}" for item in report["final_score"]["failures"]] or ["- None"]
    lines += ["", "## Remaining failures", *remaining, "", "## Limitations", *[f"- {item}" for item in report["known_limitations"]], ""]
    return "\n".join(lines)


def _md_insufficient_evidence(report: dict) -> str:
    lines = ["# Evaluation build stopped: insufficient evidence", "",
        f"- Valid evidence-backed cases: {report['valid_evidence_backed_cases']} (20 required)",
        f"- Valid unsupported/refusal cases: {report['valid_unsupported_refusal_cases']}",
        f"- Rejected case candidates: {report['rejected_case_count']}",
        f"- Rejected source sentences: {report.get('rejected_source_sentence_count', 0)}", "",
        "## Reasons", *[f"- {reason}" for reason in report["insufficient_evidence_reasons"]], "",
        "## Source files"]
    for document in report.get("source_diagnostics", {}).get("documents", []):
        if document.get("page_count") is not None:
            for page in document.get("pages", []):
                lines.append(f"- {document['source']} page {page['page']}: {page['extracted_characters']} extracted characters, "
                    f"{page['qualifying_sentences']} qualifying sentences, {len(page['excluded_sentences'])} excluded sentence fragments.")
        else:
            lines.append(f"- {document['source']}: {document.get('extraction_error') or 'text diagnostics available'}")
    lines += ["", "See `source_diagnostics.json` for per-page/record details and exclusion reasons.", ""]
    return "\n".join(lines)


def build_agent(brief: str, documents: Path, artifacts_root: Path = Path("artifacts"), unattended: bool = True,
                clarifications: list[str] | None = None, minimum_cases: int = 20, iterations: int = 3, top_k: int = 5) -> dict:
    minimum_cases = max(20, minimum_cases)
    docs = documents.resolve()
    req, questions = intake(brief, docs, clarifications, unattended)
    out = (artifacts_root / req.agent_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_diagnostics = diagnose_collection(docs)
    _write(out / "source_diagnostics.json", source_diagnostics)
    chunks = ingest(docs)
    ground = analyze(chunks)
    req.supported_topics = ground["supported_topics"] or req.supported_topics
    req.unsupported_topics = [*ground["unsupported_topics"], "Topics not evidenced by the supplied collection"]
    req.known_limitations = ["Groundability topics are inferred from document structure.", "Deterministic lexical checks are primary; semantic scores are separate and cannot offset deterministic factual failures.",
        "Blind order-swapped pairwise judging is available only when an Ollama judge is explicitly configured; it is not a calibrated bias guarantee."]
    if os.getenv("FORGEAI_JUDGE_PROVIDER", "none").strip().lower() != "ollama" or not os.getenv("FORGEAI_JUDGE_MODEL"):
        req.known_limitations.append("No LLM evaluator was configured; semantic judging is unavailable and deterministic checks were used.")
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
    try:
        cases, evaluation_quality = generate_cases_with_report(chunks, minimum_cases)
    except InsufficientEvidenceError as exc:
        evaluation_quality = exc.report
        evaluation_quality["source_diagnostics"] = source_diagnostics
        _write(out / "evaluation_quality.json", evaluation_quality)
        (out / "evaluation_quality.md").write_text(_md_insufficient_evidence(evaluation_quality), encoding="utf-8")
        raise
    evaluation_quality["evidence_provenance_validated"] = sum(
        not validate_case_provenance(case, chunks) for case in cases
        if case.answerability_classification == "evidence_backed")
    evaluation_quality["source_diagnostics_file"] = "source_diagnostics.json"
    evaluation_quality["source_diagnostics"] = source_diagnostics
    _write(out / "evaluation_quality.json", evaluation_quality)
    save_cases(cases, out / "evals" / "cases.json")
    runner = AgentRunner(agent, docs)
    initial_results = run_suite(runner, cases)
    initial = score(initial_results)
    logger.info("Initial evaluation for %s: %d/%d cases passed", req.agent_id, initial["pass_count"], initial["case_count"])
    _write(out / "initial_evaluation.json", initial)
    _write(out / "initial_evaluation_results.json", [result.model_dump() for result in initial_results])
    logs, optimized, optimization_report = optimize_with_report(agent, chunks, cases, docs, iterations)
    _write(out / "optimization_logs.json", logs)
    _write(out / "optimization_report.json", optimization_report)
    _write(out / "agent_definition.json", optimized.model_dump())
    final_results = run_suite(AgentRunner(optimized, docs), cases)
    final = score(final_results)
    logger.info("Final evaluation for %s: %d/%d cases passed after %d optimizer iterations", req.agent_id, final["pass_count"], final["case_count"], len(logs))
    _write(out / "final_scorecard.json", final)
    _write(out / "evaluation_results.json", [result.model_dump() for result in final_results])
    report = {"agent_id": req.agent_id, "agent_name": req.agent_name, "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements": req.model_dump(), "agent_definition": optimized.model_dump(), "document_summary": {"files": req.available_documents, "chunk_count": len(chunks)},
        "evaluation": {"case_count": len(cases), "case_file": "evals/cases.json", **evaluation_quality},
        "source_diagnostics": source_diagnostics, "initial_score": initial, "optimization_logs": logs,
        "optimization_report": optimization_report,
        "final_score": final, "failure_classifications": {name: sum(1 for r in final_results if r.failure_classification == name) for name in ("Retrieval miss", "Prompt defect", "Hallucination", "Wrong refusal")},
        "groundability": ground, "optimization_decisions": [l["changes_made"] for l in logs], "known_limitations": req.known_limitations,
        "final_recommendation": "Suitable for review and local demonstration; human review is required before operational use."}
    _write(out / "final_report.json", report)
    (out / "final_report.md").write_text(_md_report(report), encoding="utf-8")
    return report


def load_cases(path: Path, chunks=None, minimum_evidence_cases: int = 20) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list): raise ValueError("Evaluation cases must be a JSON list")
    cases = [EvalCase.model_validate(item) for item in data]
    if len({case.test_id for case in cases}) != len(cases): raise ValueError("Evaluation case IDs must be unique")
    if len({case.question.casefold().strip() for case in cases}) != len(cases): raise ValueError("Evaluation questions must be unique")
    if chunks is not None:
        for case in cases:
            errors = validate_case_provenance(case, chunks)
            if errors:
                raise ValueError(f"Invalid provenance for {case.test_id}: {'; '.join(errors)}")
        diversity_errors = validate_case_diversity(cases)
        if diversity_errors:
            raise ValueError("Invalid evaluation case diversity: " + "; ".join(diversity_errors))
        evidence_cases = [case for case in cases if case.answerability_classification == "evidence_backed"]
        refusal_cases = [case for case in cases if case.answerability_classification == "unsupported_refusal"]
        if len(evidence_cases) < minimum_evidence_cases:
            raise ValueError(f"Evaluation suite has {len(evidence_cases)} valid evidence-backed cases; {minimum_evidence_cases} are required")
        if not refusal_cases:
            raise ValueError("Evaluation suite must include valid unsupported/refusal cases")
    return cases
