from .models import Chunk
from .retrieval import Retriever
import re


def analyze(chunks: list[Chunk], threshold: float = 0.10) -> dict:
    headings = sorted({chunk.section for chunk in chunks if chunk.section})
    sources = sorted({chunk.source for chunk in chunks})
    gaps = []
    for chunk in chunks:
        for match in re.finditer(r"\b(?:does not specify|does not define|does not state|does not provide|not stated|not specified|not covered)\b\s*([^.;\n]{1,140})", chunk.text, re.I):
            detail = re.sub(r"\s+", " ", match.group(1)).strip(" ,:-")
            if detail:
                gaps.append({"detail": detail, "source": chunk.source, "page": chunk.page, "row": chunk.row, "classification": "explicitly unsupported/unknown"})
    unique_gaps = list({(gap["source"], gap["detail"].casefold()): gap for gap in gaps}.values())
    unsupported = sorted({gap["detail"] for gap in unique_gaps})
    return {"supported_topics": headings, "partially_supported_topics": [], "unsupported_topics": unsupported,
            "conflicting_information": [], "unknown_information": unique_gaps, "sources": sources,
            "coverage_note": "Supported topics are inferred from headings. Explicitly missing facts are captured from document statements; other partial coverage and conflicts need human review.",
            "evidence_checks": []}


def check_question(question: str, retriever: Retriever) -> tuple[bool, list]:
    found = retriever.search(question)
    return bool(found and found[0][1] >= 0.10), found
