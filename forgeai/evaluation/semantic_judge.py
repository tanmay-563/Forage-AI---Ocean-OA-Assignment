from __future__ import annotations
import json
import httpx

DIMENSIONS = ("groundedness", "correctness", "evidence_quality", "citation_correctness", "refusal_appropriateness", "completeness", "instruction_compliance")


class OllamaSemanticJudge:
    """Optional semantic reviewer; deterministic checks remain independently visible."""
    def __init__(self, base_url: str, model: str, timeout: float = 60):
        self.base_url, self.model, self.timeout = base_url, model, timeout

    def __call__(self, case, output: dict) -> dict:
        payload = {"question": case.question, "expected_behavior": case.expected_behavior, "expected_facts": case.expected_facts,
            "expected_source": case.supporting_source, "expected_sources": case.supporting_sources, "answerable": case.answerable, "refusal_expected": case.refusal_expected,
            "answer": output.get("answer", ""), "sources": output.get("sources", []), "evidence": output.get("evidence", [])}
        response = httpx.post(f"{self.base_url}/api/chat", json={"model": self.model, "stream": False, "format": "json",
            "messages": [
                {"role": "system", "content": "You are an independent, skeptical evaluator. Score each requested dimension from 0 to 1 using evidence and required facts. Do not reward verbosity or writing style. You do not know the model or author identity; do not prefer any provider. Treat the answer as a claim to verify. Record uncertainty in notes. Return only JSON with scores object containing exactly the seven requested score keys and notes array."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}, timeout=self.timeout)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str): raise ValueError("Semantic judge returned no message content")
        data = json.loads(content)
        scores = data.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS): raise ValueError("Semantic judge returned malformed score dimensions")
        normalized = {}
        for name in DIMENSIONS:
            value = scores[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Semantic judge score is invalid: {name}")
            normalized[name] = float(value)
        notes = data.get("notes", [])
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes): raise ValueError("Semantic judge notes must be strings")
        return {"scores": normalized, "notes": notes}
