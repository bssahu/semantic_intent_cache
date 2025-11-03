"""Unit tests for SDK (with stubs)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semantic_intent_cache.sdk import SemanticIntentCache


class TestSemanticIntentCache:
    """Test semantic intent cache SDK."""

    def test_init_default(self):
        """Test initialization with defaults."""
        with patch("semantic_intent_cache.sdk.RedisStore"):
            cache = SemanticIntentCache()
            assert cache is not None

    def test_ingest_happy_path(self):
        """Test successful ingest operation."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            # Setup mocks
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.upsert_variants.return_value = 12

            # Setup embedder
            mock_embedder = MagicMock()
            mock_embedder.dim = 384
            mock_embedder.encode.return_value = np.random.randn(12, 384).astype(np.float32)

            # Setup variant provider
            mock_provider = MagicMock()
            mock_provider.generate.return_value = [
                "How do I upgrade my plan?",
                "Can you tell me how do I upgrade my plan?",
                "I need to know how do I upgrade my plan?",
                "Could you explain how do I upgrade my plan?",
                "Please help me with how do I upgrade my plan?",
                "I'd like to understand how do I upgrade my plan?",
                "What should I do to how do I upgrade my plan?",
                "How can I how do I upgrade my plan?",
                "What's the process for how do I upgrade my plan?",
                "Steps to how do I upgrade my plan?",
                "Tell me about how do I upgrade my plan?",
                "I want to how do I upgrade my plan?",
            ]

            cache = SemanticIntentCache(
                embedder=mock_embedder,
                variant_provider=mock_provider,
            )

            # Ingest
            result = cache.ingest(
                intent_id="UPGRADE_PLAN",
                question="How do I upgrade my plan?",
                auto_variant_count=12,
            )

            # Verify
            assert result["intent_id"] == "UPGRADE_PLAN"
            assert result["stored_variants"] == 12
            mock_store.upsert_variants.assert_called_once()

    def test_ingest_empty_question(self):
        """Test ingest with empty question raises error."""
        with patch("semantic_intent_cache.sdk.RedisStore"):
            cache = SemanticIntentCache()

            with pytest.raises(ValueError, match="required"):
                cache.ingest(intent_id="TEST", question="", auto_variant_count=5)

    def test_match_happy_path(self):
        """Test successful match operation."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            # Setup mocks
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.knn_search.return_value = [
                {
                    "intent_id": "UPGRADE_PLAN",
                    "question": "How do I upgrade my plan?",
                    "distance": 0.1,
                    "similarity": 0.9,
                }
            ]

            # Setup embedder
            mock_embedder = MagicMock()
            mock_embedder.dim = 384
            mock_embedder.encode.return_value = np.random.randn(384).astype(np.float32)

            cache = SemanticIntentCache(embedder=mock_embedder)

            # Match
            result = cache.match(
                query="I want to upgrade my subscription",
                top_k=5,
                min_similarity=0.75,
            )

            # Verify
            assert result["match"] is not None
            assert result["match"]["intent_id"] == "UPGRADE_PLAN"
            assert result["match"]["similarity"] == 0.9
            mock_store.knn_search.assert_called_once()

    def test_match_no_results(self):
        """Test match with no results."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.knn_search.return_value = []

            mock_embedder = MagicMock()
            mock_embedder.dim = 384
            mock_embedder.encode.return_value = np.random.randn(384).astype(np.float32)

            cache = SemanticIntentCache(embedder=mock_embedder)

            result = cache.match(
                query="some query",
                top_k=5,
                min_similarity=0.75,
            )

            assert result["match"] is None
            assert len(result["alternates"]) == 0

    def test_match_below_threshold(self):
        """Test match with similarity below threshold."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.knn_search.return_value = [
                {
                    "intent_id": "UPGRADE_PLAN",
                    "question": "How do I upgrade my plan?",
                    "distance": 0.5,
                    "similarity": 0.5,  # Below threshold
                }
            ]

            mock_embedder = MagicMock()
            mock_embedder.dim = 384
            mock_embedder.encode.return_value = np.random.randn(384).astype(np.float32)

            cache = SemanticIntentCache(embedder=mock_embedder)

            result = cache.match(
                query="some query",
                top_k=5,
                min_similarity=0.75,
            )

            assert result["match"] is None

    def test_health_check(self):
        """Test health check."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            mock_store.health_check.return_value = True

            cache = SemanticIntentCache()

            assert cache.health_check() is True

    def test_context_manager(self):
        """Test context manager usage."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            with SemanticIntentCache() as cache:
                assert cache is not None

            mock_store.close.assert_called_once()

    def test_close(self):
        """Test close method."""
        with patch("semantic_intent_cache.sdk.RedisStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            cache = SemanticIntentCache()
            cache.close()

            mock_store.close.assert_called_once()

