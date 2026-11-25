from __future__ import annotations
import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation
from .config import Settings
from .ingestion import ingest
from .models import AgentDefinition, SourceRef
from .providers import ExtractiveProvider, OllamaProvider
from .retrieval import Retriever
from .tools import build_registry
from .retrieval import tokens


def _explicit_cross_document_query(question: str) -> bool:
    return bool(re.search(r"\b(each document|each source|both documents|both sources|across documents|across sources|compare .* documents)\b", question, re.I))


def _numeric_conflict(question: str, records: list[dict]) -> list[dict] | None:
    """Return evidence for a narrow, source-level numeric conflict, if present."""
    query_terms = set(tokens(question)) - {"document", "documents", "source", "sources", "say", "says", "stated", "state", "according", "available", "evidence", "conflict", "contradict", "agree", "check", "please", "what", "which", "does", "each", "both", "the", "and", "about"}
    if len(query_terms) < 2:
        return None
    measure = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(bar|psi|mpa|mm|cm|m|kg|g|l/min|days?|hours?|°c|celsius|usd|eur|gbp)\b", re.I)
    claims = []
    for record in records:
        source = record.get("source")
        text = record.get("text", "")
        body = text.split("] ", 1)[-1]
        terms = set(tokens(body)) & query_terms
        values = {(unit.casefold(), Decimal(value)) for value, unit in measure.findall(body)}
        if terms and values:
            claims.append((source, terms, values, record))
    for index, (source_a, terms_a, values_a, record_a) in enumerate(claims):
        for source_b, terms_b, values_b, record_b in claims[index + 1:]:
            if source_a == source_b or len(terms_a & terms_b) < 2:
                continue
            for unit_a, value_a in values_a:
                for unit_b, value_b in values_b:
                    if unit_a == unit_b and value_a != value_b:
                        return [record_a, record_b]
    return None


def load_definition(path: str | Path) -> tuple[AgentDefinition, Path]:
    file = Path(path).resolve()
    data = json.loads(file.read_text(encoding="utf-8"))
    definition = AgentDefinition.model_validate(data)
    collection = Path(definition.document_collection)
    if not collection.is_absolute(): collection = (file.parent / collection).resolve()
    if not collection.is_dir(): raise FileNotFoundError(f"Agent document collection is missing: {collection}")
    return definition, collection


class AgentRunner:
    def __init__(self, definition: AgentDefinition, collection: Path, provider=None):
        self.definition, self.collection = definition, collection.resolve()
        config = Settings.from_env()
        self.chunks = ingest(self.collection, config.chunk_chars, config.chunk_overlap)
        self.retriever = Retriever(self.chunks, int(definition.retrieval.get("top_k", config.top_k)))
        self.provider = provider or (OllamaProvider(config.ollama_url, config.ollama_model, config.request_timeout) if config.provider == "ollama" else ExtractiveProvider())
        self.enabled_tools = {tool["name"] for tool in definition.tools}
        self.tool_registry = build_registry(self.collection)
        self.tool_registry._tools = {name: tool for name, tool in self.tool_registry._tools.items() if name in self.enabled_tools}

    def answer(self, question: str) -> dict:
        if not isinstance(question, str) or not question.strip(): raise ValueError("Question must not be empty")
        found = self.retriever.search(question, across_sources=_explicit_cross_document_query(question))
        threshold = float(self.definition.retrieval.get("minimum_score", 0.10))
        evidence = [(chunk, score) for chunk, score in found if score >= threshold]
        if isinstance(self.provider, ExtractiveProvider):
            selected = ExtractiveProvider.select_evidence(question, evidence)
        else:
            selected = [(chunk, chunk.text, score) for chunk, score in evidence]
        refs = []
        excerpts = []
        evidence_records = []
        for chunk, text, _ in selected:
            ref = SourceRef(source=chunk.source, page=chunk.page, row=chunk.row, chunk_id=chunk.chunk_id)
            if not any(old.source == ref.source and old.chunk_id == ref.chunk_id for old in refs): refs.append(ref)
            excerpt = f"[{ref.source}" + (f", page {ref.page}" if ref.page else "") + (f", row {ref.row}" if ref.row else "") + f"] {text}"
            excerpts.append(excerpt)
            evidence_records.append({**ref.model_dump(), "text": excerpt})
            # PDF extraction can split one sentence across adjacent chunks. If the
            # same exact supporting quotation also occurs in an overlapping source
            # chunk, preserve that corroborating provenance reference as well.
            normalized_quote = text.strip().casefold()
            if len(normalized_quote) >= 36:
                for overlapping in self.chunks:
                    if overlapping.source != chunk.source or overlapping.chunk_id == chunk.chunk_id:
                        continue
                    if normalized_quote not in overlapping.text.casefold():
                        continue
                    duplicate_ref = SourceRef(source=overlapping.source, page=overlapping.page, row=overlapping.row, chunk_id=overlapping.chunk_id)
                    if any(old.source == duplicate_ref.source and old.chunk_id == duplicate_ref.chunk_id for old in refs):
                        continue
                    refs.append(duplicate_ref)
                    evidence_records.append({**duplicate_ref.model_dump(), "text": f"[{duplicate_ref.source}" + (f", page {duplicate_ref.page}" if duplicate_ref.page else "") + (f", row {duplicate_ref.row}" if duplicate_ref.row else "") + f"] {text}"})
        tool_calls = []
        csv_price_rows = {}
        if "calculator" in self.enabled_tools and re.search(r"\b(calculate|what is|compute|total)\b", question, re.I):
            match = re.search(r"(?<![A-Za-z])([0-9][0-9., ]*(?:[+*/()%^-][0-9., ]*)+)", question)
            if match:
                expression = match.group(1).replace(",", "").strip()
                value = self.tool_registry.execute("calculator", {"expression": expression})
                excerpts.append(f"[calculator] {expression} = {value}")
                tool_calls.append({"tool": "calculator", "expression": expression})
        if "current_date" in self.enabled_tools and re.search(r"\b(today|current date|what date)\b", question, re.I):
            value = self.tool_registry.execute("current_date", {})
            excerpts.append(f"[current_date] {value}")
            tool_calls.append({"tool": "current_date"})
        if "csv_lookup" in self.enabled_tools and re.search(r"\b(catalog|catalogue|product|item|stock|available|availability|price|compare)\b|\b[A-Z]{2,6}-[0-9A-Z]{2,8}\b", question, re.I):
            for path in sorted(self.collection.rglob("*.csv")):
                codes = re.findall(r"\b[A-Z]{2,6}-[0-9A-Z]{2,8}\b", question, re.I)
                if codes:
                    lookup_terms = list(dict.fromkeys(codes))
                else:
                    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
                    skip = {"catalog", "catalogue", "product", "item", "stock", "available", "availability", "price", "compare", "what", "which", "the", "for", "show", "find", "list"}
                    lookup_terms = [word for word in words if word.casefold() not in skip][:3]
                rows_by_number = {}
                for term in lookup_terms:
                    result = self.tool_registry.execute("csv_lookup", {"filename": path.relative_to(self.collection).as_posix(), "query": term, "limit": 20})
                    for row in result:
                        row_number = int(row.pop("row"))
                        rows_by_number[row_number] = row
                if rows_by_number:
                    tool_calls.append({"tool": "csv_lookup", "filename": path.relative_to(self.collection).as_posix(), "query_terms": lookup_terms, "rows_returned": len(rows_by_number)})
                    for row_number, row in rows_by_number.items():
                        source_name = path.relative_to(self.collection).as_posix()
                        row_chunk = next((chunk for chunk in self.chunks if chunk.source == source_name and chunk.row == row_number), None)
                        if row_chunk is None:
                            continue
                        if not any(ref.source == source_name and ref.row == row_number for ref in refs):
                            ref = SourceRef(source=source_name, row=row_number, chunk_id=row_chunk.chunk_id)
                            refs.append(ref)
                            excerpt = f"[{source_name}, row {row_number}] {row_chunk.text}"
                            excerpts.append(excerpt)
                            evidence_records.append({**ref.model_dump(), "text": excerpt})
                        code_field = next((key for key in row if key.casefold() in {"product_code", "item_code", "sku", "code"}), None)
                        if code_field and row_chunk is not None:
                            csv_price_rows[str(row[code_field]).strip().casefold()] = (row, row_chunk)
        if "weather_forecast" in self.enabled_tools and re.search(r"\b(weather|forecast|temperature)\b", question, re.I):
            coordinates = re.search(r"(-?\d{1,2}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)", question)
            if coordinates:
                observation = self.tool_registry.execute("weather_forecast", {"latitude": float(coordinates.group(1)), "longitude": float(coordinates.group(2))})
                excerpts.append(f"[weather_forecast] {observation}")
                tool_calls.append({"tool": "weather_forecast", "coordinates": [coordinates.group(1), coordinates.group(2)]})
        # Price comparisons are calculated only from exact CSV rows returned by the
        # selected csv_lookup tool and mapped back to their ingested source chunks.
        if "csv_lookup" in self.enabled_tools and re.search(r"\b(price difference|difference between prices|difference between .* prices|compare .* prices|lower[- ]priced|lower price|cheaper|less expensive)\b", question, re.I):
            item_codes = list(dict.fromkeys(code.casefold() for code in re.findall(r"\b[A-Z]{2,6}-[0-9A-Z]{2,8}\b", question, re.I)))
            if len(item_codes) != 2:
                answer = "I cannot calculate a price difference because the question must identify exactly two items with unambiguous catalogue prices."
                return {"answer": answer, "sources": [], "evidence": [], "provider": self.provider.name, "retrieved": 0,
                        "tool_calls": tool_calls, "calculation": {"status": "refused", "reason": "exactly two item codes are required"}}
            resolved = [(code, csv_price_rows.get(code)) for code in item_codes]
            resolved = [(code, item) for code, item in resolved if item is not None]
            refs_out, evidence_out = [], []
            for _, (row, chunk) in resolved:
                ref = SourceRef(source=chunk.source, page=chunk.page, row=chunk.row, chunk_id=chunk.chunk_id)
                refs_out.append(ref)
                evidence_out.append({**ref.model_dump(), "text": f"[{ref.source}, row {ref.row}] {chunk.text}"})
            if len(resolved) != 2:
                answer = "I cannot calculate a price difference because a requested item is missing from the retrieved catalogue rows."
                return {"answer": answer, "sources": [ref.model_dump() for ref in refs_out], "evidence": evidence_out,
                        "provider": self.provider.name, "retrieved": len(refs_out), "tool_calls": tool_calls,
                        "calculation": {"status": "refused", "reason": "one or both catalogue price operands are missing"}}
            operands, error = [], None
            for code, (row, chunk) in resolved:
                fields = {key.casefold(): value for key, value in row.items()}
                price_key = next((key for key in fields if key in {"unit_price", "price"}), None)
                currency_key = next((key for key in fields if key in {"currency", "currency_code"}), None)
                unit_key = next((key for key in fields if key in {"unit", "unit_of_measure", "uom"}), None)
                if not price_key:
                    error = f"catalogue row for {code.upper()} has no price field"
                    break
                try:
                    amount = Decimal(str(fields[price_key]).replace(",", "").strip())
                    if not amount.is_finite(): raise InvalidOperation
                except (InvalidOperation, ValueError):
                    error = f"catalogue row for {code.upper()} has an ambiguous or invalid price"
                    break
                quote = chunk.text
                # The structured tool record must be visibly grounded by the exact source row.
                if str(fields[price_key]) not in quote or (currency_key and str(fields[currency_key]) not in quote) or (unit_key and str(fields[unit_key]) not in quote):
                    error = f"source quotation for {code.upper()} does not verify its price, currency, or unit"
                    break
                operands.append({"item": code.upper(), "amount": amount, "currency": str(fields[currency_key]).strip().upper() if currency_key else None,
                                 "unit": str(fields[unit_key]).strip().casefold() if unit_key else None,
                                 "source": chunk.source, "row": chunk.row, "page": chunk.page, "chunk_id": chunk.chunk_id, "quotation": quote})
            if not error and (not operands[0]["currency"] or not operands[1]["currency"] or operands[0]["currency"] != operands[1]["currency"]):
                error = "the two prices have missing or incompatible currencies"
            if not error and (not operands[0]["unit"] or not operands[1]["unit"] or operands[0]["unit"] != operands[1]["unit"]):
                error = "the two prices use missing or incompatible units of measure"
            if error:
                answer = f"I cannot calculate a comparable price difference: {error}."
                return {"answer": answer, "sources": [ref.model_dump() for ref in refs_out], "evidence": evidence_out,
                        "provider": self.provider.name, "retrieved": len(refs_out), "tool_calls": tool_calls,
                        "calculation": {"status": "refused", "reason": error}}
            first, second = operands
            difference = abs(first["amount"] - second["amount"])
            lower = first if first["amount"] < second["amount"] else second
            currency = first["currency"]
            formula = f"|{first['amount']} {currency}/{first['unit']} - {second['amount']} {currency}/{second['unit']}| = {difference} {currency}/{first['unit']}"
            citations = f"[{first['source']}, row {first['row']}] and [{second['source']}, row {second['row']}]"
            answer = f"{lower['item']} has the lower unit price. The absolute difference is {difference} {currency} per {first['unit']}. Formula: {formula}. Evidence: {citations}."
            return {"answer": answer, "sources": [ref.model_dump() for ref in refs_out], "evidence": evidence_out,
                    "provider": self.provider.name, "retrieved": len(refs_out), "tool_calls": [*tool_calls,
                    {"tool": "grounded_price_comparison", "formula": formula,
                     "operands": [{key: value for key, value in operand.items() if key != "amount"} | {"amount": str(operand["amount"])} for operand in operands],
                     "result": str(difference), "currency": currency, "unit": first["unit"]}],
                    "calculation": {"status": "calculated", "formula": formula, "result": str(difference), "currency": currency, "unit": first["unit"]}}
        conflict_evidence = _numeric_conflict(question, evidence_records)
        if conflict_evidence:
            citations = "; ".join(f"[{item['source']}" + (f", page {item['page']}" if item.get("page") else "") + (f", row {item['row']}" if item.get("row") else "") + "]" for item in conflict_evidence)
            answer = f"The retrieved documents provide conflicting numeric evidence for the requested fact. I cannot resolve it without an authoritative clarification. Sources: {citations}."
            return {"answer": answer, "sources": [{key: item[key] for key in ("source", "page", "row", "chunk_id")} for item in conflict_evidence],
                    "evidence": conflict_evidence, "provider": self.provider.name, "retrieved": len(conflict_evidence), "tool_calls": tool_calls,
                    "evidence_status": "conflicting"}
        if isinstance(self.provider, ExtractiveProvider) and len(question.split()) <= 5:
            query_terms = set(re.findall(r"[a-z0-9]+", question.casefold()))
            matched_sections = {}
            for chunk in self.chunks:
                if not chunk.section: continue
                overlap = len(query_terms & set(re.findall(r"[a-z0-9]+", chunk.section.casefold())))
                if overlap: matched_sections[chunk.section] = max(overlap, matched_sections.get(chunk.section, 0))
            if len(matched_sections) >= 2:
                top = max(matched_sections.values())
                tied = sorted(name for name, value in matched_sections.items() if value == top)
                if len(tied) >= 2:
                    options, option_refs, option_evidence = [], [], []
                    for section in tied[:4]:
                        found_section = self.retriever.search(section, top_k=1)
                        if not found_section: continue
                        chunk = found_section[0][0]
                        ref = SourceRef(source=chunk.source, page=chunk.page, row=chunk.row, chunk_id=chunk.chunk_id)
                        options.append(section)
                        option_refs.append(ref)
                        option_evidence.append({**ref.model_dump(), "text": f"## {section}\n{chunk.text}"})
                    if len(options) >= 2:
                        answer = "The documents cover more than one related topic: " + "; ".join(options) + ". Which one do you mean?"
                        return {"answer": answer, "sources": [ref.model_dump() for ref in option_refs], "evidence": option_evidence,
                                "provider": self.provider.name, "retrieved": len(option_refs), "tool_calls": tool_calls}
        answer = self.provider.answer(self.definition.system_prompt, question, excerpts)
        return {"answer": answer, "sources": [ref.model_dump() for ref in refs], "evidence": evidence_records,
                "provider": self.provider.name, "retrieved": len(refs), "tool_calls": tool_calls}


def from_definition(path: str | Path, provider=None) -> AgentRunner:
    definition, collection = load_definition(path)
    return AgentRunner(definition, collection, provider)
