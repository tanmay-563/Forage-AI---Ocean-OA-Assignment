import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    provider: str = "offline"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    request_timeout: float = 60.0
    chunk_chars: int = 1000
    chunk_overlap: int = 120
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("FORGEAI_PROVIDER", "offline").strip().lower(),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            request_timeout=float(os.getenv("FORGEAI_TIMEOUT", "60")),
            chunk_chars=max(200, int(os.getenv("FORGEAI_CHUNK_CHARS", "1000"))),
            chunk_overlap=max(0, int(os.getenv("FORGEAI_CHUNK_OVERLAP", "120"))),
            top_k=max(1, int(os.getenv("FORGEAI_TOP_K", "5"))),
        )


def ensure_within(root: Path, candidate: Path) -> Path:
    base = root.resolve()
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"Path escapes approved directory: {candidate}")
    return resolved
