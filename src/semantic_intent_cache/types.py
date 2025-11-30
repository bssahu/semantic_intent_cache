"""Type definitions for semantic intent cache."""

from typing import TypedDict


class MatchResult(TypedDict):
    """Result from a match operation."""

    intent_id: str
    question: str
    similarity: float
    embedding: list[float] | None


class IngestResult(TypedDict):
    """Result from an ingest operation."""

    intent_id: str
    stored_variants: int
    total_generated: int
    tenant: str | None


class MatchResponse(TypedDict):
    """Response from match query."""

    match: MatchResult | None
    alternates: list[MatchResult]


class VariantMetadata(TypedDict):
    """Metadata for a variant."""

    intent_id: str
    question: str


class RedisDoc(TypedDict):
    """Redis document structure."""

    intent: str
    text: str
    embedding: bytes

