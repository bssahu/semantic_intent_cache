"""Redis store with vector index operations."""

import logging
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

logger = logging.getLogger(__name__)


class RedisStore:
    """Redis store for semantic intent cache."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        index_name: str = "sc:idx",
        key_prefix: str = "sc:doc:",
        vector_dim: int = 384,
        ef_construction: int = 200,
        m: int = 16,
    ):
        """
        Initialize the Redis store.

        Args:
            redis_url: Redis connection URL.
            index_name: Name of the vector index.
            key_prefix: Prefix for document keys.
            vector_dim: Dimension of vectors.
            ef_construction: HNSW ef_construction parameter.
            m: HNSW M parameter.
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.vector_dim = vector_dim
        self.ef_construction = ef_construction
        self.m = m

        # Initialize Redis client
        logger.info(f"Connecting to Redis: {redis_url}")
        self.client = redis.from_url(redis_url, decode_responses=False)
        self._ensure_connected()

    def _ensure_connected(self) -> None:
        """Ensure Redis connection is working."""
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def ensure_index(self) -> None:
        """
        Create the vector index if it doesn't exist.

        Uses FT.CREATE with DIALECT 2 and HNSW vector index.
        """
        # Check if index already exists
        try:
            self.client.ft(self.index_name).info()
            logger.info(f"Index {self.index_name} already exists")
            return
        except redis.ResponseError as e:
            # Index doesn't exist - proceed to create it
            if "no such index" in str(e).lower() or "unknown index" in str(e).lower():
                logger.debug(f"Index {self.index_name} does not exist, will create")
            else:
                # Unexpected error - re-raise
                raise

        # Create index with RediSearch
        logger.info(f"Creating vector index: {self.index_name}")
        schema = (
            TextField("intent", sortable=True, as_name="intent"),
            TextField("text", sortable=False, as_name="text"),
            VectorField(
                "embedding",
                algorithm="HNSW",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "EF_CONSTRUCTION": self.ef_construction,
                    "M": self.m,
                },
                as_name="embedding",
            ),
        )

        try:
            # Use DIALECT 2 for better vector support
            self.client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(index_type=IndexType.HASH, prefix=[self.key_prefix]),
            )
            logger.info(f"Index {self.index_name} created successfully")
        except redis.ResponseError as e:
            if b"Index already exists" in str(e).encode():
                logger.info(f"Index {self.index_name} already exists")
            else:
                logger.error(f"Failed to create index: {e}")
                raise

    def upsert_variants(
        self,
        intent_id: str,
        variants: list[str],
        embeddings: np.ndarray,
    ) -> int:
        """
        Upsert variants into Redis.

        Args:
            intent_id: Intent identifier.
            variants: List of variant texts.
            embeddings: Numpy array of embeddings.

        Returns:
            Number of variants stored.
        """
        if len(variants) != len(embeddings):
            raise ValueError("Mismatch between variants and embeddings length")

        stored = 0
        pipe = self.client.pipeline()

        for i, (variant, embedding) in enumerate(zip(variants, embeddings, strict=False)):
            doc_id = f"{intent_id}:{i}"
            key = f"{self.key_prefix}{doc_id}"

            # Convert embedding to bytes (FLOAT32)
            embedding_bytes = embedding.astype(np.float32).tobytes()

            # Store as hash
            doc = {
                "intent": intent_id,
                "text": variant,
                "embedding": embedding_bytes,
            }

            pipe.hset(key, mapping=doc)
            stored += 1

        # Execute pipeline
        pipe.execute()
        logger.info(f"Stored {stored} variants for intent: {intent_id}")

        return stored

    def knn_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_expr: str | None = None,
        tenant: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform KNN search on the vector index.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter_expr: Optional filter expression.
            tenant: Optional tenant identifier.

        Returns:
            List of matches with fields: intent_id, question, distance, similarity.
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")

        # Build filter
        query_filter = filter_expr
        if tenant:
            if query_filter:
                query_filter = f"(@tenant:{{{tenant}}}) ({query_filter})"
            else:
                query_filter = f"@tenant:{{{tenant}}}"

        # Convert embedding to bytes
        query_bytes = query_embedding.astype(np.float32).tobytes()

        # Build KNN query using query parameters
        try:
            ft = self.client.ft(self.index_name)

            # Set dialect to 2 for vector support
            try:
                ft.config_set("DEFAULT_DIALECT", "2")
            except Exception:
                pass  # Already set or not needed

            # Build KNN query string with DIALECT 2 syntax
            if query_filter:
                query_str = f"({query_filter})=>[KNN {top_k} @embedding $vec AS dist]"
            else:
                query_str = f"*=>[KNN {top_k} @embedding $vec AS dist]"
            
            # Pass raw bytes - redis-py will handle encoding
            results = ft.search(
                query_str,
                query_params={"vec": query_bytes},
            )

            # Parse results
            matches = []
            for doc in results.docs:
                # Extract fields (Document has attributes, not dict)
                intent_id = getattr(doc, "intent", "")
                question = getattr(doc, "text", "")
                
                # Get distance from KNN AS dist clause
                distance = getattr(doc, "dist", None)
                if distance is None:
                    distance = 1.0
                
                # Convert to float
                try:
                    distance = float(distance)
                except (ValueError, TypeError):
                    distance = 1.0

                # COSINE distance is 1 - similarity, so similarity = 1 - distance
                similarity = max(0.0, 1.0 - distance)

                matches.append({
                    "intent_id": intent_id,
                    "question": question,
                    "distance": distance,
                    "similarity": similarity,
                })

            return matches

        except Exception as e:
            logger.error(f"Error performing KNN search: {e}")
            return []

    def get_variants_for_intent(self, intent_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all variants for a given intent.

        Args:
            intent_id: Intent identifier.

        Returns:
            List of variants with their text and metadata.
        """
        try:
            ft = self.client.ft(self.index_name)
            
            # Search for all documents with this intent (intent is TEXT field)
            results = ft.search(f"@intent:{intent_id}")

            variants = []
            for doc in results.docs:
                variants.append({
                    "text": getattr(doc, "text", ""),
                    "intent_id": getattr(doc, "intent", ""),
                    "id": getattr(doc, "id", ""),
                })

            return sorted(variants, key=lambda x: x.get("id", ""))

        except Exception as e:
            logger.error(f"Error retrieving variants for intent {intent_id}: {e}")
            return []

    def health_check(self) -> bool:
        """
        Check Redis health.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        self.client.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"RedisStore(index={self.index_name}, dim={self.vector_dim})"

