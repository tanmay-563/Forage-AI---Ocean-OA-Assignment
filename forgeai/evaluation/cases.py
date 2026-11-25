from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from ..models import Chunk, EvalCase
from ..retrieval import tokens

_FACT_WORDS = {"receive", "receives", "received", "provides", "provide", "applies", "apply", "should", "must", "may", "can", "requires", "require", "includes", "include", "states", "state", "reported", "listed", "available", "per", "each", "every", "days", "day", "year", "years", "working", "regular", "full", "time", "employees", "employee", "company", "calendar", "fictional", "synthetic", "document", "documents", "according", "following", "within", "their", "this", "that", "from", "with", "when", "where", "does", "what", "about", "policy", "policies", "terms", "rule", "rules", "services", "these", "those", "india", "indian", "tidal", "systems"}
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "hundred", "thousand"}


def _topic(sentence: str, source: str) -> str:
    if source.lower().endswith(".csv"):
        match = re.search(r"product_code:\s*([^|]+)", sentence, re.I)
        if match: return f"catalogue record {match.group(1).strip()}"
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", sentence)
    focus = []
    for word in words:
        if word.casefold() in _FACT_WORDS or word.casefold() in _NUMBER_WORDS or word.isdigit(): continue
        if word.casefold() not in {item.casefold() for item in focus}: focus.append(word)
        if len(focus) >= 4: break
    return " ".join(focus) or Path(source).stem.replace("_", " ")


def generate_cases(chunks: list[Chunk], minimum: int = 20) -> list[EvalCase]:
    """Build cases from the current collection's own sentences, not a domain template."""
    candidates = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk.text):
            sentence = sentence.strip(" -*\t")
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9%$._/-]*", sentence)
            if len(words) < 7 or len(sentence) < 36 or len(sentence) > 260:
                continue
            if not sentence[0].isupper() and not sentence[0].isdigit():
                continue
            if chunk.row is None and sentence[-1] not in ".!?":
                continue
            if sentence.endswith(":") or sentence.startswith("#"):
                continue
            candidates.append((sentence, chunk))
    unique = {}
    for sentence, chunk in candidates:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        unique.setdefault(key, (sentence, chunk))
    direct_cases = []
    for sentence, chunk in unique.values():
        digest = hashlib.sha1((chunk.source + sentence).encode()).hexdigest()[:8]
        topic = _topic(sentence, chunk.source)
        case = EvalCase(test_id=f"doc_{digest}", question=f"What does the document say about {topic}?",
            expected_behavior="Answer with the documented fact and cite the source.", expected_facts=[sentence],
            supporting_source=chunk.source, expected_evidence=[sentence], category="direct factual", answerable=True, refusal_expected=False)
        direct_cases.append(case)
    cases = direct_cases[:10]

    # Add synthesis cases only when independent evidence passages actually exist.
    by_source: dict[str, list[tuple[str, Chunk]]] = {}
    by_section: dict[tuple[str, str], list[tuple[str, Chunk]]] = {}
    for sentence, chunk in unique.values():
        by_source.setdefault(chunk.source, []).append((sentence, chunk))
        if chunk.section: by_section.setdefault((chunk.source, chunk.section), []).append((sentence, chunk))
    sources = sorted(by_source)
    def specific_facts(source: str):
        candidates = []
        for sentence, chunk in by_source[source]:
            if len(sentence.split()) < 9 or re.search(r"effective date|fictional policies apply|this synthetic|this handbook|this guide", sentence, re.I): continue
            if chunk.section is None: continue
            candidates.append((sentence, chunk))
        return candidates or by_source[source]
    if len(sources) >= 2:
        pair = None
        for i, first_source in enumerate(sources):
            for second_source in sources[i + 1:]:
                for first_fact in specific_facts(first_source):
                    for second_fact in specific_facts(second_source):
                        a_terms = set(tokens(_topic(first_fact[0], first_source)))
                        b_terms = set(tokens(_topic(second_fact[0], second_source)))
                        similarity = len(a_terms & b_terms) / max(1, len(a_terms | b_terms))
                        if similarity < 0.35:
                            pair = (first_source, second_source, first_fact, second_fact)
                            break
                    if pair: break
                if pair: break
            if pair: break
        if pair is None:
            pair = (sources[0], sources[1], specific_facts(sources[0])[0], specific_facts(sources[1])[0])
        first_source, second_source, a, b = pair
        digest = hashlib.sha1((a[0] + b[0]).encode()).hexdigest()[:8]
        topic_a = f"{a[1].section}: {_topic(a[0], a[1].source)}" if a[1].section else _topic(a[0], a[1].source)
        topic_b = f"{b[1].section}: {_topic(b[0], b[1].source)}" if b[1].section else _topic(b[0], b[1].source)
        cases.append(EvalCase(test_id=f"multi_doc_{digest}", question=f"Compare the documents on {topic_a} and {topic_b}.",
            expected_behavior="Combine both documented facts and cite both sources.", expected_facts=[a[0], b[0]],
            supporting_source=first_source, supporting_sources=[first_source, second_source], expected_evidence=[a[0], b[0]], category="multiple documents", answerable=True, refusal_expected=False))
        a_candidates, b_candidates = specific_facts(first_source), specific_facts(second_source)
        a, b = a_candidates[min(1, len(a_candidates) - 1)], b_candidates[min(1, len(b_candidates) - 1)]
        digest = hashlib.sha1((a[0] + b[0] + "cross").encode()).hexdigest()[:8]
        topic_a = f"{a[1].section}: {_topic(a[0], a[1].source)}" if a[1].section else _topic(a[0], a[1].source)
        topic_b = f"{b[1].section}: {_topic(b[0], b[1].source)}" if b[1].section else _topic(b[0], b[1].source)
        cases.append(EvalCase(test_id=f"cross_doc_{digest}", question=f"What differs between the documents regarding {topic_a} and {topic_b}?",
            expected_behavior="Synthesize both passages with citations.", expected_facts=[a[0], b[0]], supporting_source=first_source, supporting_sources=[first_source, second_source],
            expected_evidence=[a[0], b[0]], category="multiple documents", answerable=True, refusal_expected=False))
    sections = sorted(by_section)
    for index in range(min(2, max(0, len(sections) - 1))):
        first = next((item for item in by_section[sections[index]] if item[0]), None)
        second = next((item for key in sections[index + 1:] if key[0] == sections[index][0] for item in by_section[key]), None)
        if first and second:
            digest = hashlib.sha1((first[0] + second[0] + "sections").encode()).hexdigest()[:8]
            topic_a, topic_b = _topic(first[0], first[1].source), _topic(second[0], second[1].source)
            cases.append(EvalCase(test_id=f"multi_section_{digest}", question=f"Compare the {sections[index][1]} and {sections[index + 1][1]} sections for {topic_a} and {topic_b}.",
                expected_behavior="Use evidence from both sections and cite the source.", expected_facts=[first[0], second[0]], supporting_source=sections[index][0],
                expected_evidence=[first[0], second[0]], category="multiple sections", answerable=True, refusal_expected=False))

    # Explicit gaps stated by source documents become evidence-backed refusal cases.
    for sentence, chunk in unique.values():
        if re.search(r"does not specify|does not define|does not state|does not guarantee|does not promise|not stated|not specified|must not be interpreted", sentence, re.I):
            detail = re.sub(r"^.*?(?:does not specify|does not define|does not state|does not guarantee|does not promise|not stated|not specified|must not be interpreted)(?:\s+as)?\s*", "", sentence, flags=re.I).strip(" .;,")
            question = f"What exact details does the document give about {detail[:100]}?"
            digest = hashlib.sha1((chunk.source + question).encode()).hexdigest()[:8]
            cases.append(EvalCase(test_id=f"gap_{digest}", question=question, expected_behavior="State that the document does not specify the requested detail, and cite the source.",
                expected_facts=[], supporting_source=chunk.source, expected_evidence=[sentence], category="incomplete evidence", answerable=False, refusal_expected=True))
            if len([c for c in cases if c.category == "incomplete evidence"]) >= 2: break

    # Ambiguity is added when the supplied collection has distinct sections sharing a topic label.
    leave_sections = [key for key in sections if "leave" in key[1].casefold()]
    if len(leave_sections) >= 2:
        facts = [key[1] for key in leave_sections[:2]]
        cases.append(EvalCase(test_id="ambiguous_leave", question="How much leave is available?", expected_behavior="Disambiguate or describe the distinct leave types without conflating them.",
            expected_facts=facts, supporting_source=leave_sections[0][0], expected_evidence=facts, category="ambiguous wording", answerable=True, refusal_expected=False))

    cases.append(EvalCase(test_id="out_of_scope_01", question="What are the licensing terms for a neutrino-powered coffee reactor in lunar orbit?",
        expected_behavior="Refuse because the collection does not contain this information.", category="out of scope", answerable=False, refusal_expected=True))
    cases.append(EvalCase(test_id="hallucination_01", question="Give the exact annual calibration fee and warranty period for a neutrino-powered coffee reactor in lunar orbit.",
        expected_behavior="Do not invent a fee or warranty; refuse with the evidence limitation.", category="hallucination resistance", answerable=False, refusal_expected=True))

    # Preserve category diversity, then fill the remainder with additional real facts.
    selected_ids = {case.test_id for case in cases}
    cases.extend(case for case in direct_cases if case.test_id not in selected_ids)
    deduped = []
    seen_questions = set()
    for case in cases:
        key = case.question.casefold().strip()
        if key not in seen_questions:
            deduped.append(case); seen_questions.add(key)
    cases = deduped[:max(minimum, min(len(deduped), 20))]
    if len(cases) < minimum:
        for i, chunk in enumerate(chunks):
            if len(cases) >= minimum: break
            snippet = re.sub(r"\s+", " ", chunk.text.strip())[:220]
            if len(snippet.split()) < 6: continue
            digest = hashlib.sha1(f"{chunk.chunk_id}:{i}".encode()).hexdigest()[:8]
            cases.append(EvalCase(test_id=f"evidence_{digest}", question=f"Summarize the documented information in {chunk.source}.",
                expected_behavior="Summarize only the retrieved evidence and cite it.", expected_facts=[], supporting_source=chunk.source,
                expected_evidence=[snippet], category="evidence summary", answerable=True, refusal_expected=False))
    if len(cases) < minimum:
        raise ValueError(f"Collection only supports {len(cases)} usable evidence cases; {minimum} are required. Add more substantive source content.")

    return cases


def save_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump() for c in cases], indent=2, ensure_ascii=False), encoding="utf-8")
