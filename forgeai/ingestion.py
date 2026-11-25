from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

from .config import ensure_within
from .models import Chunk

SUPPORTED = {".pdf", ".txt", ".md", ".csv"}


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
            return [(page.get_text("text"), page.number + 1, None) for page in document]
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
