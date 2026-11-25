from __future__ import annotations
import json
from pathlib import Path
import re
from .config import Settings
from .ingestion import ingest
from .models import AgentDefinition, SourceRef
from .providers import ExtractiveProvider, OllamaProvider
from .retrieval import Retriever
from .tools import build_registry


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
        found = self.retriever.search(question)
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
        tool_calls = []
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
                        if not any(ref.source == path.relative_to(self.collection).as_posix() and ref.row == row_number for ref in refs):
                            ref = SourceRef(source=path.relative_to(self.collection).as_posix(), row=row_number, chunk_id=f"tool-{path.name}-{row_number}")
                            refs.append(ref)
                            excerpt = f"[{path.relative_to(self.collection).as_posix()}, row {row_number}] " + " | ".join(f"{key}: {value}" for key, value in row.items())
                            excerpts.append(excerpt)
                            evidence_records.append({**ref.model_dump(), "text": excerpt})
        if "weather_forecast" in self.enabled_tools and re.search(r"\b(weather|forecast|temperature)\b", question, re.I):
            coordinates = re.search(r"(-?\d{1,2}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)", question)
            if coordinates:
                observation = self.tool_registry.execute("weather_forecast", {"latitude": float(coordinates.group(1)), "longitude": float(coordinates.group(2))})
                excerpts.append(f"[weather_forecast] {observation}")
                tool_calls.append({"tool": "weather_forecast", "coordinates": [coordinates.group(1), coordinates.group(2)]})
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
