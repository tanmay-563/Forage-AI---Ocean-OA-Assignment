from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..ingestion import qualifying_sentence_records
from ..models import Chunk, EvalCase, EvidenceProvenance


_STOP = {"about", "according", "after", "also", "among", "and", "are", "candidate", "candidates", "each", "for", "from", "has", "have", "include", "includes", "into", "involves", "its", "may", "must", "not", "per", "provide", "provided", "provides", "require", "requires", "shall", "should", "states", "that", "the", "their", "there", "these", "this", "those", "through", "under", "was", "were", "what", "when", "where", "which", "with", "within", "will", "year", "years", "collection", "documents", "document", "does", "specify", "specifies", "state", "states", "details", "describe", "supported", "give", "each", "sections", "section", "according", "is", "say", "stated", "does", "employee", "employees", "company", "business", "policy", "policies", "supplier", "suppliers", "marine", "laboratory", "lab", "system", "data", "days", "time", "northstar", "section", "description", "quantity", "available", "product", "products", "catalogue", "catalog", "record", "records", "unit", "units", "item", "items", "information", "provided", "source", "two", "than", "company", "businesses", "organization", "organizations", "general", "various", "certain", "following"}


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _topic(sentence: str, source: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", sentence)
             if w.casefold() not in _STOP and w.casefold() not in {"marked", "flag", "flags"}]
    distinct = []
    for word in words:
        stem = word.casefold().removesuffix("s")
        if stem not in {item.casefold().removesuffix("s") for item in distinct}:
            distinct.append(word)
        if len(distinct) == 4:
            break
    return " ".join(distinct) or Path(source).stem.replace("_", " ")


def _question_topic(sentence: str, source: str) -> str:
    predicate = re.search(r"\b(?:is|are|was|were|provides?|requires?|includes?|offers?|supports?|contains?|allows?|receives?|received|begins?|ends?|uses?|reports?|lists?|delivers?|costs?|takes?|needs?)\b", sentence, flags=re.I)
    if predicate:
        subject = sentence[:predicate.start()].strip(" ,:;.-")
        words = subject.split()
        informative = [word for word in words if word.casefold().strip(".,:;") not in _STOP]
        if 1 <= len(words) <= 6 and informative and words[0].casefold() not in {"it", "this", "these", "those", "there", "they", "he", "she"}:
            return subject.casefold()
    return _topic(sentence, source).casefold()


def _similarity(left: str, right: str) -> float:
    a, b = set(_norm(left).split()), set(_norm(right).split())
    return len(a & b) / max(1, len(a | b))


def _question_similarity(left: str, right: str) -> float:
    a = {word for word in _norm(left).split() if word not in _STOP}
    b = {word for word in _norm(right).split() if word not in _STOP}
    return len(a & b) / max(1, len(a | b))


def validate_case_provenance(case: EvalCase, chunks: list[Chunk]) -> list[str]:
    """Validate generated case provenance against the exact ingested chunk records."""
    if case.answerability_classification == "unsupported_refusal":
        if case.answerable or not case.refusal_expected:
            return ["unsupported case must be unanswerable and expect refusal"]
        if case.evidence_validation_status != "not_applicable":
            return ["unsupported case evidence validation must be not_applicable"]
        if case.expected_facts or case.expected_evidence or case.supporting_quotation or case.evidence_provenance:
            return ["unsupported case must not claim source facts or quotations"]
        return []
    errors = []
    if not case.answerable or case.refusal_expected:
        errors.append("evidence-backed case must be answerable and not expect refusal")
    if case.evidence_validation_status != "validated":
        errors.append("evidence-backed case is not marked validated")
    if not case.question.strip() or not case.expected_behavior.strip():
        errors.append("missing question or expected behavior")
    if not case.expected_facts:
        errors.append("evidence-backed case must have explicit expected facts")
    refs = case.evidence_provenance or ([EvidenceProvenance(source_filename=case.source_filename or "",
        page_number=case.page_number, row_number=case.row_number, chunk_id=case.chunk_id or "",
        quotation=case.supporting_quotation or "")] if case.chunk_id else [])
    if not refs:
        errors.append("missing source filename, chunk ID, or supporting quotation")
        return errors
    for ref in refs:
        if not ref.source_filename or not ref.chunk_id or not ref.quotation:
            errors.append("missing source filename, chunk ID, or supporting quotation")
            continue
        source_chunks = [chunk for chunk in chunks if chunk.chunk_id == ref.chunk_id]
        if not source_chunks:
            errors.append(f"referenced source chunk does not exist: {ref.chunk_id}")
            continue
        chunk = source_chunks[0]
        if chunk.source != ref.source_filename:
            errors.append("source filename does not match referenced chunk")
        if chunk.page != ref.page_number:
            errors.append("page number does not match referenced chunk")
        if chunk.row != ref.row_number:
            errors.append("row number does not match referenced chunk")
        if ref.quotation not in chunk.text:
            errors.append("supporting quotation is not present in referenced chunk")
    if case.evidence_provenance and case.chunk_id:
        primary = case.evidence_provenance[0]
        if not any(chunk.chunk_id == case.chunk_id for chunk in chunks):
            errors.append(f"referenced source chunk does not exist: {case.chunk_id}")
        if (case.chunk_id, case.source_filename, case.page_number, case.row_number, case.supporting_quotation) != (primary.chunk_id, primary.source_filename, primary.page_number, primary.row_number, primary.quotation):
            errors.append("primary provenance fields do not match the first evidence reference")
        primaries = [chunk for chunk in chunks if chunk.chunk_id == case.chunk_id]
        if primaries:
            if primaries[0].page != case.page_number:
                errors.append("page number does not match referenced chunk")
            if primaries[0].row != case.row_number:
                errors.append("row number does not match referenced chunk")
    if case.evidence_provenance:
        if case.category == "numerical reasoning":
            metadata = case.reasoning_metadata
            if metadata.get("operation") != "absolute_difference" or len(refs) != 2:
                errors.append("numerical reasoning metadata or operand evidence is malformed")
            else:
                operands = metadata.get("operands", [])
                if len(operands) != 2:
                    errors.append("numerical reasoning case must declare two evidence-backed operands")
                else:
                    amounts = []
                    for operand, ref in zip(operands, refs):
                        if str(operand.get("item", "")) not in ref.quotation:
                            errors.append("numerical item identifier does not match its quoted CSV row")
                            break
                        match = re.search(rf"(?:^|\|\s*){re.escape(metadata.get('column', 'unit_price'))}:\s*([0-9]+(?:\.[0-9]+)?)", ref.quotation)
                        if not match or match.group(1) != str(operand.get("value")):
                            errors.append("numerical operand does not match its exact quoted CSV field")
                            break
                        try:
                            amounts.append(Decimal(match.group(1)))
                        except InvalidOperation:
                            errors.append("numerical operand is not a valid decimal")
                            break
                    if len(amounts) == 2:
                        difference = abs(amounts[0] - amounts[1])
                        if str(difference) != str(metadata.get("expected_difference")):
                            errors.append("numerical result does not match the declared arithmetic")
                        lower_index = 0 if amounts[0] < amounts[1] else 1 if amounts[1] < amounts[0] else None
                        if lower_index is None or metadata.get("lower_item") != operands[lower_index].get("item"):
                            errors.append("lower-priced item does not match the cited numeric operands")
                        rendered = " ".join(case.expected_facts)
                        if str(difference) not in rendered or str(metadata.get("lower_item")) not in rendered:
                            errors.append("expected numerical fact does not match the validated calculation")
                        if metadata.get("currency") not in " ".join(ref.quotation for ref in refs):
                            errors.append("numerical currency is not present in the cited rows")
        for fact in case.expected_facts:
            if case.category != "numerical reasoning" and not any(fact in ref.quotation for ref in refs):
                errors.append("expected fact is not supported by any quoted evidence reference")
        return errors
    if not case.source_filename or not case.chunk_id or not case.supporting_quotation:
        errors.append("missing source filename, chunk ID, or supporting quotation")
        return errors
    source_chunks = [chunk for chunk in chunks if chunk.chunk_id == case.chunk_id]
    if not source_chunks:
        errors.append("referenced source chunk does not exist")
        return errors
    chunk = source_chunks[0]
    if chunk.source != case.source_filename:
        errors.append("source filename does not match referenced chunk")
    if chunk.page != case.page_number:
        errors.append("page number does not match referenced chunk")
    if chunk.row != case.row_number:
        errors.append("row number does not match referenced chunk")
    if case.supporting_quotation not in chunk.text:
        errors.append("supporting quotation is not present in referenced chunk")
    if case.supporting_quotation not in case.expected_facts:
        errors.append("expected fact is not the verbatim supporting quotation")
    if case.expected_evidence != [case.supporting_quotation]:
        errors.append("expected evidence must equal the supporting quotation")
    return errors


def validate_case_diversity(cases: list[EvalCase]) -> list[str]:
    errors = []
    evidence = [case for case in cases if case.answerability_classification == "evidence_backed"]
    topics: list[str] = []
    facts: set[str] = set()
    for case in evidence:
        fact = _norm(case.expected_facts[0]) if case.expected_facts else ""
        topic = _topic(case.expected_facts[0], case.source_filename or "") if case.expected_facts else ""
        if len(case.expected_facts) == 1:
            if fact in facts:
                errors.append(f"duplicate factual evidence in {case.test_id}")
            facts.add(fact)
        if any(_norm(old.question) == _norm(case.question) or
               (old.category == case.category and not (old.category == case.category == "CSV lookup" and old.row_number != case.row_number) and
                _question_similarity(old.question, case.question) >= .35) for old in evidence[:len(topics)]):
            errors.append(f"duplicate or near-duplicate factual question in {case.test_id}")
        topics.append(topic)
    return errors


def generate_cases_with_report(chunks: list[Chunk], minimum: int = 20) -> tuple[list[EvalCase], dict]:
    candidates: list[tuple[str, Chunk]] = []
    rejected: list[dict] = []
    for chunk in chunks:
        accepted, excluded = qualifying_sentence_records(chunk.text, chunk.row)
        candidates.extend((sentence, chunk) for sentence in accepted)
        rejected.extend({"source": chunk.source, "page": chunk.page, "chunk_id": chunk.chunk_id, **item}
                        for item in excluded)

    # Remove repeated source sentences while retaining their first verifiable location.
    unique: dict[str, tuple[str, Chunk]] = {}
    duplicate_facts = 0
    for sentence, chunk in candidates:
        key = _norm(sentence)
        if key in unique:
            duplicate_facts += 1
        else:
            unique[key] = (sentence, chunk)

    factual: list[EvalCase] = []
    factual_topics: list[str] = []
    rejected_case_reasons: list[str] = ["duplicate source fact"] * duplicate_facts
    for sentence, chunk in unique.values():
        csv_fields = {}
        if chunk.row:
            for cell in sentence.split(" | "):
                key, separator, value = cell.partition(":")
                if separator:
                    csv_fields[key.strip()] = value.strip()
        if chunk.row and csv_fields:
            field = next((key.strip() for key, value in csv_fields.items() if key.strip().casefold() not in {"id", "code", "sku", "product_code"}), next(iter(csv_fields)))
            value = csv_fields[field].strip()
            identity = next((v.strip() for k, v in csv_fields.items() if k.casefold() in {"id", "code", "sku", "product_code", "item", "product"}), None)
            question = f"For CSV row {chunk.row}{f' ({identity})' if identity else ''}, what value is listed in the {field.strip()} column?"
            fact = f"{field.strip()}: {value}"
            category = "CSV lookup"
            quote = sentence
            topic = _topic(fact, chunk.source)
        else:
            topic = _question_topic(sentence, chunk.source)
            question = f"What does the source say about {topic}?"
            fact = sentence
            category = "direct factual retrieval"
            quote = sentence
        digest = hashlib.sha1(f"{chunk.chunk_id}:{sentence}".encode()).hexdigest()[:10]
        case = EvalCase(
            test_id=f"evidence_{digest}",
            question=question,
            expected_behavior="Answer with the requested documented fact and cite its source; do not add unsupported details.",
            expected_facts=[fact], supporting_source=chunk.source,
            expected_evidence=[quote], source_filename=chunk.source,
            page_number=chunk.page, row_number=chunk.row, chunk_id=chunk.chunk_id,
            supporting_quotation=quote, evidence_provenance=[EvidenceProvenance(source_filename=chunk.source,
                page_number=chunk.page, row_number=chunk.row, chunk_id=chunk.chunk_id, quotation=quote)],
            answerability_classification="evidence_backed",
            evidence_validation_status="validated", category=category,
            answerable=True, refusal_expected=False)
        errors = validate_case_provenance(case, chunks)
        if errors:
            rejected_case_reasons.extend(errors)
            continue
        if any(_norm(old.question) == _norm(case.question) or
               (not (old.category == case.category == "CSV lookup" and old.row_number != case.row_number) and
                _question_similarity(old.question, case.question) >= .35) for old in factual):
            rejected_case_reasons.append("duplicate or near-duplicate question")
            continue
        if any(_norm(old.expected_facts[0]) == _norm(sentence) for old in factual):
            rejected_case_reasons.append("duplicate fact")
            continue
        factual.append(case)

    csv_rows = []
    for chunk in chunks:
        if chunk.row is None:
            continue
        fields = {}
        for cell in chunk.text.split(" | "):
            key, separator, value = cell.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        code = next((value for key, value in fields.items() if key.casefold() in {"id", "code", "sku", "product_code", "item", "product"}), None)
        currency = fields.get("currency")
        price = fields.get("unit_price") or fields.get("price")
        if code and currency and price:
            try:
                csv_rows.append((chunk, code, currency, Decimal(price)))
            except InvalidOperation:
                pass
    numeric_pair = next(((left, right) for i, left in enumerate(csv_rows) for right in csv_rows[i + 1:]
                         if left[2] == right[2] and left[0].source == right[0].source and left[0].chunk_id != right[0].chunk_id), None)
    if numeric_pair:
        (left_chunk, left_code, currency, left_price), (right_chunk, right_code, _, right_price) = numeric_pair
        cheaper_code = left_code if left_price < right_price else right_code
        difference = abs(left_price - right_price)
        column = "unit_price" if "unit_price" in left_chunk.text else "price"
        question = f"Which item has the lower {column}, {left_code} or {right_code}, and what is the price difference?"
        refs = [EvidenceProvenance(source_filename=c.source, page_number=c.page, row_number=c.row,
            chunk_id=c.chunk_id, quotation=c.text) for c in (left_chunk, right_chunk)]
        fact = f"{cheaper_code} has the lower {column}; the absolute difference is {currency} {difference}."
        digest = hashlib.sha1(f"{left_chunk.chunk_id}:{right_chunk.chunk_id}:{column}:difference".encode()).hexdigest()[:10]
        case = EvalCase(test_id=f"evidence_numeric_{digest}", question=question,
            expected_behavior="Compare the two quoted numeric values, state the lower-priced item, and calculate their absolute difference.",
            expected_facts=[fact], supporting_source=left_chunk.source, supporting_sources=[left_chunk.source],
            expected_evidence=[left_chunk.text, right_chunk.text], source_filename=left_chunk.source,
            page_number=left_chunk.page, row_number=left_chunk.row, chunk_id=left_chunk.chunk_id,
            supporting_quotation=left_chunk.text, evidence_provenance=refs,
            reasoning_metadata={"operation": "absolute_difference", "column": column,
                "operands": [{"item": left_code, "value": str(left_price)}, {"item": right_code, "value": str(right_price)}],
                "expected_difference": str(difference), "currency": currency, "lower_item": cheaper_code},
            answerability_classification="evidence_backed", evidence_validation_status="validated",
            category="numerical reasoning", answerable=True, refusal_expected=False)
        if not validate_case_provenance(case, chunks) and not any(_norm(old.question) == _norm(question) for old in factual):
            factual.append(case)
        factual_topics.append(topic)

    # A row-level multi-column question checks relationships among actual fields on the same record.
    for chunk in chunks:
        if chunk.row is None:
            continue
        fields = {}
        for cell in chunk.text.split(" | "):
            key, separator, value = cell.partition(":")
            if separator and key.strip() and value.strip():
                fields[key.strip()] = value.strip()
        identity_key = next((key for key in fields if key.casefold() in {"id", "code", "sku", "product_code", "item", "product"}), None)
        columns = [key for key in fields if key != identity_key]
        if len(columns) < 2:
            continue
        selected = columns[:2]
        identity = fields.get(identity_key, "") if identity_key else ""
        facts = [f"{key}: {fields[key]}" for key in selected]
        question = f"For CSV row {chunk.row}{f' ({identity})' if identity else ''}, what are the listed {selected[0]} and {selected[1]} values?"
        if any(_norm(old.question) == _norm(question) for old in factual):
            continue
        digest = hashlib.sha1(f"{chunk.chunk_id}:{'|'.join(selected)}".encode()).hexdigest()[:10]
        case = EvalCase(test_id=f"evidence_rowfacts_{digest}", question=question,
            expected_behavior="Return both requested values from this row and cite the row; do not substitute values from another record.",
            expected_facts=facts, supporting_source=chunk.source, expected_evidence=[chunk.text],
            source_filename=chunk.source, page_number=chunk.page, row_number=chunk.row, chunk_id=chunk.chunk_id,
            supporting_quotation=chunk.text, evidence_provenance=[EvidenceProvenance(source_filename=chunk.source,
                page_number=chunk.page, row_number=chunk.row, chunk_id=chunk.chunk_id, quotation=chunk.text)],
            answerability_classification="evidence_backed", evidence_validation_status="validated",
            category="multi-fact reasoning", answerable=True, refusal_expected=False)
        if validate_case_provenance(case, chunks):
            continue
        if any(_question_similarity(old.question, question) >= .35 for old in factual):
            continue
        factual.append(case)

    # Add grounded comparisons only when the collection has an explicit shared subject.
    # Each side retains its own chunk and quotation; unrelated facts are never paired.
    pair_counts: Counter[str] = Counter()
    for index, (left_fact, left_chunk) in enumerate(unique.values()):
        left_terms = {word for word in re.findall(r"[a-z]{4,}", left_fact.casefold()) if word not in _STOP}
        for right_fact, right_chunk in list(unique.values())[index + 1:]:
            if left_chunk is right_chunk or left_chunk.source == right_chunk.source and left_chunk.section == right_chunk.section:
                continue
            right_terms = {word for word in re.findall(r"[a-z]{4,}", right_fact.casefold()) if word not in _STOP}
            shared = left_terms & right_terms
            if len(shared) < 2:
                continue
            cross_document = left_chunk.source != right_chunk.source
            category = "cross-document reasoning" if cross_document else "multi-section reasoning"
            scope = "each document" if cross_document else "each section"
            subject = " ".join(sorted(shared, key=lambda word: (-len(word), word))[:2])
            question = f"What does {scope} say about {subject}?"
            if any(_similarity(case.question, question) >= .78 for case in factual):
                continue
            pair_key = "cross" if cross_document else "sections"
            if pair_counts[pair_key] >= 2:
                continue
            refs = [EvidenceProvenance(source_filename=c.source, page_number=c.page, row_number=c.row,
                    chunk_id=c.chunk_id, quotation=f) for f, c in ((left_fact, left_chunk), (right_fact, right_chunk))]
            digest = hashlib.sha1(f"{refs[0].chunk_id}:{refs[1].chunk_id}:{subject}".encode()).hexdigest()[:10]
            case = EvalCase(test_id=f"evidence_compare_{digest}", question=question,
                expected_behavior="Compare only the details supported by both cited passages.",
                expected_facts=[left_fact, right_fact], supporting_source=left_chunk.source,
                supporting_sources=sorted({left_chunk.source, right_chunk.source}),
                expected_evidence=[left_fact, right_fact], source_filename=left_chunk.source,
                page_number=left_chunk.page, row_number=left_chunk.row, chunk_id=left_chunk.chunk_id,
                supporting_quotation=left_fact, evidence_provenance=refs,
                answerability_classification="evidence_backed", evidence_validation_status="validated",
                category=category, answerable=True, refusal_expected=False)
            if validate_case_provenance(case, chunks):
                continue
            if any(_norm(old.question) == _norm(question) or (old.category == category and _question_similarity(old.question, question) >= .35) for old in factual):
                continue
            factual.append(case)
            pair_counts[pair_key] += 1

    corpus = " ".join(chunk.text.casefold() for chunk in chunks)
    if any(word in corpus for word in ("internship", "employee", "employment", "candidate")):
        gap_candidates = ("stipend amount", "relocation allowance", "visa sponsorship", "equity grant", "joining bonus")
    elif any(word in corpus for word in ("procurement", "supplier", "catalogue", "product_code")):
        gap_candidates = ("warranty period", "certification expiry", "replacement guarantee", "patent status", "export license")
    elif any(word in corpus for word in ("laboratory", "laboratories", "safety", "chemical")):
        gap_candidates = ("calibration interval", "safety certification expiry", "manufacturer warranty", "emergency hotline", "disposal surcharge")
    else:
        gap_candidates = ("stipend amount", "warranty period", "calibration interval", "export license", "patent status")
    def all_words_absent(phrase):
        return all(not re.search(rf"\b{re.escape(word)}\b", corpus) for word in phrase.split())
    missing_topics = [candidate for candidate in gap_candidates if all_words_absent(candidate)][:2]
    while len(missing_topics) < 2:
        missing_topics.append(f"specified attribute {len(missing_topics) + 1}")
    unsupported = [
        EvalCase(test_id="unsupported_collection_gap_01",
                 question=f"What is the {missing_topics[0]}?",
                 expected_behavior="Refuse because no supplied evidence establishes this detail; do not guess.",
                 category="out of scope", answerable=False, refusal_expected=True,
                 answerability_classification="unsupported_refusal"),
        EvalCase(test_id="unsupported_missing_attribute_01",
                 question=f"Give the {missing_topics[1]} value.",
                 expected_behavior="Refuse to provide an unsupported value or fabricated citation.",
                 category="unsupported query", answerable=False, refusal_expected=True,
                 answerability_classification="unsupported_refusal")]
    if len(factual) < minimum:
        reasons = [f"Only {len(factual)} distinct evidence-backed cases passed validation; {minimum} are required."]
        if len(unique) < minimum:
            reasons.append(f"Only {len(unique)} distinct qualifying source sentences were extracted.")
        if rejected_case_reasons:
            reasons.append(f"{len(rejected_case_reasons)} candidate cases were rejected for duplication or invalid provenance.")
        summary = {"valid_evidence_backed_cases": len(factual), "valid_unsupported_refusal_cases": len(unsupported),
                       "rejected_case_count": len(rejected_case_reasons),
                       "rejected_source_sentence_count": len(rejected),
                   "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
                   "case_rejection_reasons": dict(Counter(rejected_case_reasons)),
                   "topic_diversity": len({_norm(_topic(case.expected_facts[0], case.source_filename or "")) for case in factual}),
                   "document_coverage": sorted({case.source_filename for case in factual if case.source_filename}),
                   "category_coverage": dict(Counter(case.category for case in factual + unsupported)),
                   "insufficient_evidence_reasons": reasons}
        raise InsufficientEvidenceError(summary)
    cases = factual + unsupported
    summary = {"valid_evidence_backed_cases": len(factual), "valid_unsupported_refusal_cases": len(unsupported),
               "rejected_case_count": len(rejected_case_reasons),
               "rejected_source_sentence_count": len(rejected),
               "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
               "case_rejection_reasons": dict(Counter(rejected_case_reasons)),
               "topic_diversity": len({_norm(_topic(case.expected_facts[0], case.source_filename or "")) for case in factual}),
               "document_coverage": sorted({case.source_filename for case in factual if case.source_filename}),
               "category_coverage": dict(Counter(case.category for case in factual + unsupported)),
               "insufficient_evidence_reasons": []}
    return cases, summary


class InsufficientEvidenceError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["insufficient_evidence_reasons"]))


def generate_cases(chunks: list[Chunk], minimum: int = 20) -> list[EvalCase]:
    """Compatibility API returning only a fully validated suite or raising."""
    return generate_cases_with_report(chunks, minimum)[0]


def save_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([case.model_dump() for case in cases], indent=2, ensure_ascii=False), encoding="utf-8")
