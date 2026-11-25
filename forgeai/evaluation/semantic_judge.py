from __future__ import annotations
import json
import httpx

DIMENSIONS = ("semantic_correctness", "groundedness", "citation_correctness", "completeness", "refusal_correctness", "writing_quality")


class OllamaSemanticJudge:
    """Optional semantic reviewer; deterministic checks remain independently visible."""
    def __init__(self, base_url: str, model: str, timeout: float = 60, generation_model: str | None = None):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.generation_model = generation_model
        self.same_model = bool(generation_model and generation_model.casefold() == model.casefold())

    def __call__(self, case, output: dict) -> dict:
        expected = " ".join(case.expected_facts) or case.expected_behavior
        payload = {"question": case.question, "expected_behavior": case.expected_behavior,
            "evidence": output.get("evidence", []), "sources": output.get("sources", []),
            "answer_A": str(output.get("answer", "")), "answer_B": expected}
        first = self._review(payload)
        swapped_payload = {**payload, "answer_A": payload["answer_B"], "answer_B": payload["answer_A"]}
        second = self._review(swapped_payload)
        first_pref = first.get("preferred")
        second_pref = second.get("preferred")
        candidate_winner = {"A": "candidate", "B": "reference", "tie": "tie"}.get(first_pref, "unavailable")
        swapped_candidate_winner = {"A": "reference", "B": "candidate", "tie": "tie"}.get(second_pref, "unavailable")
        return {"scores": {**first["scores"], "correctness": first["scores"]["semantic_correctness"],
                "refusal_appropriateness": first["scores"]["refusal_correctness"],
                "evidence_quality": first["scores"]["groundedness"], "instruction_compliance": first["scores"]["writing_quality"]},
            "notes": first.get("notes", []),
            "order_swapped_result": {"first": candidate_winner, "swapped": swapped_candidate_winner,
                "decision_changed": candidate_winner != "unavailable" and swapped_candidate_winner != "unavailable" and candidate_winner != swapped_candidate_winner,
                "same_model_judging": self.same_model},
            "same_model_judging": self.same_model, "evaluator_model": self.model,
            "independent_evaluator": not self.same_model}

    def _review(self, payload):
        response = httpx.post(f"{self.base_url}/api/chat", json={"model": self.model, "stream": False, "format": "json",
            "messages": [
                {"role": "system", "content": "Evaluate anonymized answers A and B. The expected facts and evidence are the reference criteria. Score the requested six dimensions separately; factual errors cannot be offset by writing quality. Judge correctness by required facts, groundedness by supplied evidence, citation correctness by source match, completeness by required information, refusal correctness by the expected behavior, and writing quality only for clarity. Do not reward length or verbosity. Return JSON with scores containing exactly the six dimension keys, preferred (A, B, or tie), and notes."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}, timeout=self.timeout)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str): raise ValueError("Semantic judge returned no message content")
        data = json.loads(content)
        scores = data.get("scores")
        # Accept the old response dimension names for stored test fixtures and upgrades.
        if isinstance(scores, dict) and set(scores) == {"groundedness", "correctness", "evidence_quality", "citation_correctness", "refusal_appropriateness", "completeness", "instruction_compliance"}:
            scores = {"semantic_correctness": scores["correctness"], "groundedness": scores["groundedness"],
                "citation_correctness": scores["citation_correctness"], "completeness": scores["completeness"],
                "refusal_correctness": scores["refusal_appropriateness"], "writing_quality": scores["instruction_compliance"]}
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS): raise ValueError("Semantic judge returned malformed score dimensions")
        normalized = {}
        for name in DIMENSIONS:
            value = scores[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Semantic judge score is invalid: {name}")
            normalized[name] = float(value)
        notes = data.get("notes", [])
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes): raise ValueError("Semantic judge notes must be strings")
        preferred = data.get("preferred")
        if preferred not in {None, "A", "B", "tie"}: raise ValueError("Semantic judge preference is invalid")
        return {"scores": normalized, "notes": notes, "preferred": preferred}
