from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(StrictModel):
    source: str
    page: int | None = None
    row: int | None = None
    chunk_id: str


class Chunk(StrictModel):
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    row: int | None = None
    section: str | None = None


class RequirementDocument(StrictModel):
    agent_id: str
    agent_name: str
    purpose: str
    audience: str
    scope: str
    supported_topics: list[str]
    unsupported_topics: list[str]
    tone: str
    must_answer: list[str]
    must_refuse: list[str]
    output_format: str
    available_documents: list[str]
    required_tools: list[str]
    assumptions: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class AgentDefinition(StrictModel):
    agent_id: str
    name: str
    version: str = "1.0"
    description: str
    system_prompt: str
    audience: str
    supported_topics: list[str]
    unsupported_topics: list[str]
    refusal_policy: dict[str, Any]
    retrieval: dict[str, Any]
    chunking: dict[str, Any]
    embedding: dict[str, Any]
    tools: list[dict[str, str]]
    output_contract: dict[str, Any]
    citation_requirements: dict[str, Any]
    document_collection: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
            raise ValueError("agent_id may contain only letters, digits, underscore, and hyphen")
        return value


class EvalCase(StrictModel):
    test_id: str
    question: str
    expected_behavior: str
    expected_facts: list[str] = Field(default_factory=list)
    supporting_source: str | None = None
    supporting_sources: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    category: str
    answerable: bool
    refusal_expected: bool


class EvalResult(StrictModel):
    test_id: str
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    groundedness: float = 0
    correctness: float = 0
    evidence_quality: float = 0
    citation_correctness: float = 0
    refusal_appropriateness: float = 0
    completeness: float = 0
    instruction_compliance: float = 0
    passed: bool = False
    failure_classification: Literal["Retrieval miss", "Prompt defect", "Hallucination", "Wrong refusal"] | None = None
    judge_notes: list[str] = Field(default_factory=list)
    semantic_scores: dict[str, float] | None = None
    semantic_judge_error: str | None = None
    judge_disagreements: list[str] = Field(default_factory=list)
