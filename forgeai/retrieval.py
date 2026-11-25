from __future__ import annotations

import re
from collections import Counter

from .models import Chunk

_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "for", "on", "with", "what", "how", "does", "do", "can", "i", "we", "you", "our", "my", "it", "from", "by", "be", "this", "that", "source", "sources", "document", "documents", "say", "says", "said", "tell", "describe", "explain", "according"}


def tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text) if w.lower() not in _STOP]


class Retriever:
    def __init__(self, chunks: list[Chunk], top_k: int = 5):
        self.chunks, self.top_k = chunks, top_k

    def search(self, query: str, top_k: int | None = None, *, across_sources: bool = False) -> list[tuple[Chunk, float]]:
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
        # Stable source order follows CSV row order on score ties, making generic
        # column questions select the first actual row rather than a hash-ordered one.
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0].source,
                        pair[0].row if pair[0].row is not None else 0, pair[0].page if pair[0].page is not None else 0,
                        pair[0].chunk_id))
        limit = top_k or self.top_k
        if not across_sources or not ranked:
            return ranked[:limit]
        # For explicit cross-document questions, spread only across candidates
        # reasonably close to the best match; do not inflate top-k or admit weak hits.
        floor = max(0.025, ranked[0][1] * 0.45)
        relevant = [item for item in ranked if item[1] >= floor]
        selected, selected_ids, per_source = [], set(), {}
        for chunk, score in relevant:
            if per_source.get(chunk.source, 0) == 0 and len(selected) < limit:
                selected.append((chunk, score)); selected_ids.add(chunk.chunk_id); per_source[chunk.source] = 1
        for chunk, score in relevant:
            if len(selected) >= limit: break
            if chunk.chunk_id in selected_ids or per_source.get(chunk.source, 0) >= 2: continue
            selected.append((chunk, score)); selected_ids.add(chunk.chunk_id); per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
        return sorted(selected, key=lambda pair: (-pair[1], pair[0].source, pair[0].chunk_id))
