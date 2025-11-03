"""Main SDK for semantic intent cache."""

import logging
from typing import Any

from semantic_intent_cache.embeddings.base import Embedder
from semantic_intent_cache.embeddings.st_local import SentenceTransformerEmbedder
from semantic_intent_cache.store.redis_store import RedisStore
from semantic_intent_cache.types import IngestResult, MatchResponse, MatchResult
from semantic_intent_cache.variants.base import VariantProvider
from semantic_intent_cache.variants.builtin import BuiltinVariantProvider

logger = logging.getLogger(__name__)


class SemanticIntentCache:
    """Main SDK for semantic intent cache."""

    def __init__(
        self,
        redis_url: str | None = None,
        embedder: Embedder | None = None,
        variant_provider: VariantProvider | None = None,
        index_name: str = "sc:idx",
        key_prefix: str = "sc:doc:",
        vector_dim: int = 384,
        ef_construction: int = 200,
        m: int = 16,
    ):
        """
        Initialize the semantic intent cache.

        Args:
            redis_url: Redis connection URL. Defaults to config or localhost.
            embedder: Embedding provider. Defaults to SentenceTransformers.
            variant_provider: Variant generator. Defaults to builtin.
            index_name: Name of the vector index.
            key_prefix: Prefix for document keys.
            vector_dim: Dimension of vectors.
            ef_construction: HNSW ef_construction parameter.
            m: HNSW M parameter.
        """
        from semantic_intent_cache.config import settings

        # Set up Redis
        self.redis_url = redis_url or settings.redis_url
        self.index_name = index_name or settings.index_name
        self.key_prefix = key_prefix or settings.key_prefix
        self.vector_dim = vector_dim or settings.vector_dim
        self.ef_construction = ef_construction or settings.ef_construction
        self.m = m or settings.m

        self.store = RedisStore(
            redis_url=self.redis_url,
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            vector_dim=self.vector_dim,
            ef_construction=self.ef_construction,
            m=self.m,
        )

        # Set up embedder
        if embedder is None:
            self.embedder = SentenceTransformerEmbedder(model_name=settings.embed_model_name)
        else:
            self.embedder = embedder

        # Set up variant provider
        if variant_provider is None:
            self.variant_provider = BuiltinVariantProvider()
        else:
            self.variant_provider = variant_provider

        logger.info("SemanticIntentCache initialized")

    def ensure_index(self) -> None:
        """Create the vector index if it doesn't exist."""
        self.store.ensure_index()

    def ingest(
        self,
        intent_id: str,
        question: str,
        auto_variant_count: int = 10,
        variants: list[str] | None = None,
    ) -> IngestResult:
        """
        Ingest an intent with variants.

        Args:
            intent_id: Intent identifier.
            question: Original question.
            auto_variant_count: Number of variants to auto-generate.
            variants: Optional pre-generated variants.

        Returns:
            IngestResult with intent_id and stored_variants count.
        """
        if not intent_id or not question:
            raise ValueError("intent_id and question are required")

        # Collect all variants
        all_variants = set()

        # Add original question
        all_variants.add(question)

        # Add pre-generated variants
        if variants:
            all_variants.update(variants)

        # Generate additional variants if needed
        if len(all_variants) < auto_variant_count:
            generated = self.variant_provider.generate(question, auto_variant_count)
            all_variants.update(generated)

        # Limit to requested count
        variant_list = list(all_variants)[:auto_variant_count]
        total_generated = len(variant_list)

        logger.info(
            f"Ingesting {total_generated} variants for intent: {intent_id}"
        )

        # Generate embeddings
        embeddings = self.embedder.encode(variant_list)

        # Store in Redis
        stored = self.store.upsert_variants(intent_id, variant_list, embeddings)

        return {
            "intent_id": intent_id,
            "stored_variants": stored,
            "total_generated": total_generated,
        }

    def match(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.75,
        tenant: str | None = None,
    ) -> MatchResponse:
        """
        Match a query against stored intents.

        Args:
            query: Query text.
            top_k: Number of results to return.
            min_similarity: Minimum similarity threshold.
            tenant: Optional tenant filter.

        Returns:
            MatchResponse with best match and alternates.
        """
        if not query:
            raise ValueError("query is required")

        # Generate embedding
        query_embedding = self.embedder.encode([query])[0]

        # Search
        results = self.store.knn_search(
            query_embedding,
            top_k=top_k,
            tenant=tenant,
        )

        # Filter by similarity
        filtered_results = [
            r
            for r in results
            if r.get("similarity", 0) >= min_similarity
        ]

        # Format response
        best_match: MatchResult | None = None
        alternates: list[MatchResult] = []

        if filtered_results:
            best = filtered_results[0]
            best_match = {
                "intent_id": best["intent_id"],
                "question": best["question"],
                "similarity": best["similarity"],
                "embedding": None,  # Don't return embeddings in response
            }
            alternates = [
                {
                    "intent_id": r["intent_id"],
                    "question": r["question"],
                    "similarity": r["similarity"],
                    "embedding": None,
                }
                for r in filtered_results[1:]
            ]

        return {
            "match": best_match,
            "alternates": alternates,
        }

    def get_variants(self, intent_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all variants for an intent.

        Args:
            intent_id: Intent identifier.

        Returns:
            List of variants with their text and metadata.
        """
        return self.store.get_variants_for_intent(intent_id)

    def health_check(self) -> bool:
        """
        Check cache health.

        Returns:
            True if healthy, False otherwise.
        """
        return self.store.health_check()

    def close(self) -> None:
        """Close the cache and clean up resources."""
        self.store.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"SemanticIntentCache(index={self.index_name}, embedder={self.embedder})"

    def __enter__(self) -> "SemanticIntentCache":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

