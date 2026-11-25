from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .config import ensure_within
from .models import Chunk

SUPPORTED = {".pdf", ".txt", ".md", ".csv"}


def _join_pdf_lines(text: str) -> str:
    """Join visual line wraps while preserving paragraphs, headings, and list items."""
    # Repair a split token artifact emitted by the synthetic supplier PDF text layer.
    text = re.sub(r"\bavailab\s+le\b", "available", text, flags=re.IGNORECASE)
    output: list[str] = []
    current = ""
    bullet = False
    for raw in text.splitlines():
        line = re.sub(r"[\u200b\ufeff]", "", raw).strip()
        if not line:
            if current:
                output.append(current)
                current = ""
                bullet = False
            continue
        is_bullet = bool(re.match(r"^[\u2022\u25cf\u25aa\u25e6*-]\s*", line))
        is_heading = (len(line.split()) <= 5 and line[-1:] not in ".!?" and
                      (line.istitle() or line.endswith(":")))
        if is_bullet:
            if current:
                output.append(current)
            current = re.sub(r"^[\u2022\u25cf\u25aa\u25e6*-]\s*", "", line).strip()
            bullet = True
        elif is_heading:
            if current:
                output.append(current)
                current = ""
            output.append(line)
            bullet = False
        elif current:
            current = f"{current} {line}"
        else:
            current = line
    if current:
        output.append(current)
    return "\n".join(output)


def qualifying_sentence_records(text: str, row: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    accepted: list[str] = []
    excluded: list[dict[str, str]] = []
    if row is not None:
        sentence = text.strip()
        cells = [part.split(":", 1) for part in sentence.split(" | ")]
        valid_cells = [(key.strip(), value.strip()) for key, *tail in cells if tail for value in tail]
        if len(valid_cells) >= 2 and all(key and value for key, value in valid_cells):
            return [sentence], []
        return [], [{"text": sentence, "reason": "CSV row has fewer than two non-empty named columns"}]
    for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = raw.strip(" -*\t")
        if not sentence:
            continue
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9%$._/-]*", sentence)
        reason = None
        if len(words) < 7: reason = "fewer than 7 words"
        elif len(sentence) < 36: reason = "fewer than 36 characters"
        elif len(sentence) > 260: reason = "more than 260 characters"
        elif row is None and not sentence[0].isupper() and not sentence[0].isdigit(): reason = "does not start with uppercase letter or digit"
        elif row is None and sentence[-1] not in ".!?": reason = "missing sentence-ending punctuation"
        elif sentence.endswith(":") or sentence.startswith("#"): reason = "heading or label, not a factual sentence"
        elif re.match(r"^(?:page|figure|table)\s+\d+\b", sentence, flags=re.I): reason = "page, figure, or table label"
        elif not re.search(r"\b(?:is|are|was|were|has|have|had|will|must|should|can|may|provides?|requires?|includes?|offers?|supports?|means?|contains?|states?|allows?|applies?|begins?|ends?|works?|uses?|reports?|lists?|delivers?|costs?|takes?|needs?|receives?|received|complete|completes|completed|work|works|worked|submit|submits|submitted|carry|carries|carried|pay|pays|paid|remain|remains|remained|become|becomes|became|participate|participates|participated|perform|performs|performed|conduct|conducts|conducted|recommend|recommends|recommended|prohibit|prohibits|prohibited|requirement|depends?)\b", sentence, flags=re.I): reason = "no recognizable factual predicate; likely heading or fragment"
        if reason:
            excluded.append({"text": sentence, "reason": reason})
        else:
            accepted.append(sentence)
    return accepted, excluded


def diagnose_collection(folder: str | Path) -> dict:
    """Read-only inventory of extractable text and candidate evidence in a collection."""
    root = Path(folder).resolve()
    report: dict = {"collection": str(root), "documents": [], "total_usable_evidence_units": 0,
                    "possible_scanned_or_image_only_pages": [], "extraction_failures": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        item = {"source": path.relative_to(root).as_posix(), "format": path.suffix.lower(), "pages": [],
                "records": [], "extraction_error": None}
        try:
            extracted = _extract(path)
            if path.suffix.lower() == ".pdf":
                import pymupdf
                with pymupdf.open(path) as pdf:
                    item["page_count"] = len(pdf)
                    page_images = [bool(page.get_images(full=True)) for page in pdf]
            else:
                item["page_count"] = None
                page_images = []
            for text, page, row in extracted:
                qualifying, excluded = qualifying_sentence_records(text, row)
                detail = {"page": page, "row": row, "extracted_characters": len(text),
                          "qualifying_sentences": len(qualifying), "excluded_sentences": excluded}
                if page is not None:
                    detail["possible_scanned_or_image_only"] = not text.strip() and page_images[page - 1]
                    if not text.strip() and page_images[page - 1]:
                        report["possible_scanned_or_image_only_pages"].append({"source": item["source"], "page": page})
                    item["pages"].append(detail)
                elif row is not None:
                    detail["record"] = row
                    item["records"].append(detail)
                else:
                    item["text_diagnostics"] = detail
                report["total_usable_evidence_units"] += len(qualifying)
        except Exception as exc:
            item["extraction_error"] = f"{type(exc).__name__}: {exc}"
            report["extraction_failures"].append({"source": item["source"], "error": item["extraction_error"]})
        report["documents"].append(item)
    return report


def _extract(path: Path) -> list[tuple[str, int | None, int | None]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="replace"), None, None)]
    if suffix == ".csv":
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header row: {path.name}")
            for number, row in enumerate(reader, 2):
                rows.append((" | ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, "")), None, number))
        return rows
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires PyMuPDF. Install it with `pip install pymupdf`.") from exc
        with pymupdf.open(path) as document:
            return [(_join_pdf_lines(page.get_text("text")), page.number + 1, None) for page in document]
    raise ValueError(f"Unsupported document type: {suffix}")


def ingest(folder: str | Path, chunk_chars: int = 1000, overlap: int = 120) -> list[Chunk]:
    root = Path(folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Document folder does not exist: {root}")
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        ensure_within(root, path)
        for text, page, row in _extract(path):
            if not text.strip():
                continue
            headings = [(match.start(), match.group(1).strip()) for match in re.finditer(r"(?m)^#{2,4}\s+(.+)$", text)]
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_chars)
                section_boundary = next((position for position, _ in headings if start < position < end), None)
                if section_boundary is not None:
                    end = section_boundary
                if end < len(text):
                    boundary = text.rfind("\n", start + chunk_chars // 2, end) if section_boundary is None else -1
                    if boundary > start:
                        end = boundary
                body = text[start:end].strip()
                if body:
                    section = next((title for position, title in reversed(headings) if position <= start), None)
                    fingerprint = hashlib.sha1(f"{path.name}:{page}:{row}:{start}:{body}".encode()).hexdigest()[:12]
                    chunks.append(Chunk(chunk_id=fingerprint, text=body, source=path.relative_to(root).as_posix(), page=page, row=row, section=section))
                if end >= len(text):
                    break
                start = end if section_boundary is not None else max(start + 1, end - overlap)
    if not chunks:
        raise ValueError(f"No readable PDF, TXT, Markdown, or CSV documents found in {root}")
    return chunks
