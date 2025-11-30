"""Unit tests for Redis store (with mocks)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestRedisStore:
    """Test Redis store."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = MagicMock()
        client.ping.return_value = True
        client.from_url.return_value = client
        return client

    def test_upsert_variants_shape(self):
        """Test upsert_variants with correct shapes."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_pipe = MagicMock()
            mock_client.pipeline.return_value = mock_pipe
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            variants = ["variant1", "variant2", "variant3"]
            embeddings = np.random.randn(3, 384).astype(np.float32)

            store.upsert_variants("TEST_INTENT", variants, embeddings, tenant=None)

            # Verify pipeline was created and executed
            mock_client.pipeline.assert_called_once()
            assert mock_pipe.execute.called

    def test_upsert_variants_length_mismatch(self):
        """Test upsert_variants with mismatched lengths."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            variants = ["variant1", "variant2"]
            embeddings = np.random.randn(3, 384).astype(np.float32)

            with pytest.raises(ValueError, match="Mismatch"):
                store.upsert_variants("TEST_INTENT", variants, embeddings, tenant=None)

    def test_knn_search_shape(self):
        """Test knn_search with correct embedding shape."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_ft = MagicMock()
            mock_client.ft.return_value = mock_ft
            mock_from_url.return_value = mock_client

            # Mock search results - Document uses attributes, not dict
            mock_doc = MagicMock()
            mock_doc.intent = "TEST_INTENT"
            mock_doc.text = "test question"
            mock_doc.dist = 0.2  # Distance from KNN AS dist clause

            mock_results = MagicMock()
            mock_results.docs = [mock_doc]
            mock_ft.search.return_value = mock_results

            store = redis_store_module.RedisStore()
            store.client = mock_client

            query_embedding = np.random.randn(384).astype(np.float32)

            results = store.knn_search(query_embedding, top_k=5)

            # Verify results
            assert len(results) == 1
            assert results[0]["intent_id"] == "TEST_INTENT"
            assert results[0]["similarity"] == 0.8  # 1 - 0.2

    def test_knn_search_wrong_shape(self):
        """Test knn_search with wrong embedding shape."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            # 2D instead of 1D
            query_embedding = np.random.randn(1, 384).astype(np.float32)

            with pytest.raises(ValueError, match="1-dimensional"):
                store.knn_search(query_embedding, top_k=5)

    def test_health_check_ok(self):
        """Test health check when Redis is healthy."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            assert store.health_check() is True

    def test_health_check_fail(self):
        """Test health check when Redis is unhealthy."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_from_url.return_value = mock_client

            # First ping succeeds (for __init__), second fails (for health_check)
            mock_client.ping.side_effect = [True, Exception("Connection failed")]

            store = redis_store_module.RedisStore()
            store.client = mock_client

            assert store.health_check() is False

    def test_close(self):
        """Test close method."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            store.close()

            mock_client.close.assert_called_once()

    def test_upsert_variants_with_tenant(self):
        """Ensure tenant metadata is stored when provided."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_pipe = MagicMock()
            mock_client.pipeline.return_value = mock_pipe
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            variants = ["variant1"]
            embeddings = np.random.randn(1, 384).astype(np.float32)

            store.upsert_variants("TEST_INTENT", variants, embeddings, tenant="TENANT_A")

            mock_pipe.hset.assert_called_once()
            _, kwargs = mock_pipe.hset.call_args
            mapping = kwargs["mapping"]
            assert mapping["tenant"] == "TENANT_A"
    def test_list_intents(self):
        """Test listing unique intents from stored keys."""
        import semantic_intent_cache.store.redis_store as redis_store_module

        with patch.object(redis_store_module.redis, "from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.scan_iter.return_value = [
                b"sc:doc:INTENT_A:0",
                b"sc:doc:INTENT_B:1",
                b"sc:doc:INTENT_A:1",
                "sc:doc:INTENT_C:0",
            ]
            mock_from_url.return_value = mock_client

            store = redis_store_module.RedisStore()
            store.client = mock_client

            intents = store.list_intents()

            assert intents == ["INTENT_A", "INTENT_B", "INTENT_C"]

