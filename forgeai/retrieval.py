from __future__ import annotations

import re
from collections import Counter

from .models import Chunk

_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "for", "on", "with", "what", "how", "does", "do", "can", "i", "we", "you", "our", "my", "it", "from", "by", "be", "this", "that"}


def tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text) if w.lower() not in _STOP]


class Retriever:
    def __init__(self, chunks: list[Chunk], top_k: int = 5):
        self.chunks, self.top_k = chunks, top_k

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        query_terms = Counter(tokens(query))
        if not query_terms:
            return []
        scored = []
        for chunk in self.chunks:
            body = Counter(tokens(chunk.text))
            matched = sum(min(count, body[term]) for term, count in query_terms.items())
            coverage = matched / max(1, sum(query_terms.values()))
            density = matched / max(1, sum(body.values()))
            score = coverage * 0.85 + min(0.15, density * 3)
            if score > 0.025:
                scored.append((chunk, score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0].source, pair[0].chunk_id))[:top_k or self.top_k]
