"""FastAPI application for semantic intent cache."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from semantic_intent_cache.config import settings
from semantic_intent_cache.sdk import SemanticIntentCache

logger = logging.getLogger(__name__)

# Global SDK instance
_sdk: SemanticIntentCache | None = None


def get_sdk() -> SemanticIntentCache:
    """Get or create SDK instance."""
    global _sdk
    if _sdk is None:
        _sdk = SemanticIntentCache(
            redis_url=settings.redis_url,
            index_name=settings.index_name,
            key_prefix=settings.key_prefix,
            vector_dim=settings.vector_dim,
            ef_construction=settings.ef_construction,
            m=settings.m,
        )
        # Ensure index exists
        _sdk.ensure_index()
    return _sdk


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting semantic intent cache API")
    get_sdk()  # Initialize SDK
    logger.info("SDK initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down semantic intent cache API")
    global _sdk
    if _sdk:
        _sdk.close()
        _sdk = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Semantic Intent Cache API",
    description="Production-ready semantic cache for intent matching",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic models
class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""

    intent_id: str = Field(..., description="Intent identifier")
    question: str = Field(..., description="Original question")
    auto_variant_count: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Number of variants to auto-generate",
    )
    variants: list[str] = Field(
        default_factory=list,
        description="Optional pre-generated variants",
    )


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    status: str = Field(..., description="Status")
    intent_id: str = Field(..., description="Intent identifier")
    stored_variants: int = Field(..., description="Number of variants stored")


class MatchRequest(BaseModel):
    """Request model for match endpoint."""

    query: str = Field(..., description="Query text")
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of results to return"
    )
    min_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )
    tenant: str | None = Field(
        default=None,
        description="Optional tenant filter",
    )


class MatchResultModel(BaseModel):
    """Match result model."""

    intent_id: str = Field(..., description="Intent identifier")
    question: str = Field(..., description="Matched question")
    similarity: float = Field(..., description="Similarity score")


class MatchResponseModel(BaseModel):
    """Response model for match endpoint."""

    match: MatchResultModel | None = Field(..., description="Best match")
    alternates: list[MatchResultModel] = Field(..., description="Alternative matches")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Status")
    healthy: bool = Field(..., description="Health check result")


class VariantResponse(BaseModel):
    """Response model for variants endpoint."""

    intent_id: str = Field(..., description="Intent identifier")
    variants: list[str] = Field(..., description="List of variant texts")
    count: int = Field(..., description="Number of variants")


class DeleteResponse(BaseModel):
    """Response model for delete intent endpoint."""

    intent_id: str = Field(..., description="Intent identifier")
    deleted_count: int = Field(..., description="Number of variants deleted")
    message: str = Field(..., description="Status message")


# API endpoints
@app.post("/cache/ingest", response_model=IngestResponse, tags=["Cache"])
async def ingest_intent(request: IngestRequest) -> IngestResponse:
    """
    Ingest an intent with variants.

    Stores the intent and its semantic variants in the cache.
    """
    try:
        sdk = get_sdk()
        result = sdk.ingest(
            intent_id=request.intent_id,
            question=request.question,
            auto_variant_count=request.auto_variant_count,
            variants=request.variants,
        )

        return IngestResponse(
            status="ok",
            intent_id=result["intent_id"],
            stored_variants=result["stored_variants"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/match", response_model=MatchResponseModel, tags=["Cache"])
async def match_query(request: MatchRequest) -> MatchResponseModel:
    """
    Match a query against stored intents.

    Returns the best match and alternatives based on semantic similarity.
    """
    try:
        sdk = get_sdk()
        result = sdk.match(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            tenant=request.tenant,
        )

        # Convert to response model
        match_model = None
        if result["match"]:
            match_model = MatchResultModel(
                intent_id=result["match"]["intent_id"],
                question=result["match"]["question"],
                similarity=result["match"]["similarity"],
            )

        alternates_model = [
            MatchResultModel(
                intent_id=alt["intent_id"],
                question=alt["question"],
                similarity=alt["similarity"],
            )
            for alt in result["alternates"]
        ]

        return MatchResponseModel(match=match_model, alternates=alternates_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error matching query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/variants/{intent_id}", response_model=VariantResponse, tags=["Cache"])
async def get_variants(intent_id: str) -> VariantResponse:
    """
    Retrieve all variant texts for an intent.

    Returns the list of stored variants for the specified intent.
    """
    try:
        sdk = get_sdk()
        variants = sdk.get_variants(intent_id)

        # Extract just the text values
        variant_texts = [v.get("text", "") for v in variants]

        return VariantResponse(
            intent_id=intent_id,
            variants=variant_texts,
            count=len(variant_texts),
        )
    except Exception as e:
        logger.error(f"Error retrieving variants for intent {intent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/cache/intent/{intent_id}", response_model=DeleteResponse, tags=["Cache"])
async def delete_intent(intent_id: str) -> DeleteResponse:
    """
    Delete an intent and all its variants.

    Permanently removes all variants associated with the specified intent.
    """
    try:
        sdk = get_sdk()
        
        # Check if intent exists
        variants = sdk.get_variants(intent_id)
        if not variants:
            raise HTTPException(
                status_code=404, detail=f"Intent '{intent_id}' not found"
            )
        
        # Delete the intent
        deleted_count = sdk.delete_intent(intent_id)
        
        return DeleteResponse(
            intent_id=intent_id,
            deleted_count=deleted_count,
            message=f"Successfully deleted intent '{intent_id}' with {deleted_count} variant(s)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting intent {intent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        sdk = get_sdk()
        healthy = sdk.health_check()
        return HealthResponse(status="ok", healthy=healthy)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="error", healthy=False)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Semantic Intent Cache API",
        "version": "0.1.0",
        "status": "ok",
    }

