"""Integration tests for API with Testcontainers."""

import pytest

try:
    from testcontainers.compose import DockerCompose
    from testcontainers.redis import RedisContainer
except ImportError:
    RedisContainer = None


@pytest.mark.integration
@pytest.mark.skipif(RedisContainer is None, reason="testcontainers not available")
class TestAPIIntegration:
    """Integration tests for API with real Redis."""

    @pytest.fixture(scope="class")
    def redis_container(self):
        """Start Redis Stack container."""
        with RedisContainer(image="redis/redis-stack:latest") as container:
            yield container

    @pytest.fixture
    def api_client(self, redis_container):
        """Create API client."""
        import httpx

        # Update Redis URL in settings
        import semantic_intent_cache.config
        from semantic_intent_cache.api.app import app
        semantic_intent_cache.config.settings.redis_url = redis_container.get_connection_url()

        # Clear any existing SDK instance
        import semantic_intent_cache.api.app as api_module
        api_module._sdk = None

        with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = await api_client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_ingest_and_match(self, api_client):
        """Test full ingest and match workflow."""
        # Ingest an intent
        ingest_request = {
            "intent_id": "UPGRADE_PLAN",
            "question": "How do I upgrade my plan?",
            "auto_variant_count": 6,
        }

        response = await api_client.post("/cache/ingest", json=ingest_request)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["intent_id"] == "UPGRADE_PLAN"
        assert data["stored_variants"] == 6

        # Match a similar query
        match_request = {
            "query": "I want to move to a higher tier",
            "top_k": 5,
            "min_similarity": 0.6,
        }

        response = await api_client.post("/cache/match", json=match_request)
        assert response.status_code == 200
        data = response.json()
        assert data["match"] is not None
        assert data["match"]["intent_id"] == "UPGRADE_PLAN"
        assert data["match"]["similarity"] >= 0.6

    @pytest.mark.asyncio
    async def test_match_no_results(self, api_client):
        """Test match with no results."""
        match_request = {
            "query": "This is a completely unrelated query about dinosaurs",
            "top_k": 5,
            "min_similarity": 0.8,
        }

        response = await api_client.post("/cache/match", json=match_request)
        assert response.status_code == 200
        data = response.json()
        assert data["match"] is None


@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires docker-compose")
class TestAPIDockerCompose:
    """Integration tests with docker-compose."""

    @pytest.fixture(scope="class")
    def docker_compose(self):
        """Start docker-compose services."""
        with DockerCompose(".") as compose:
            yield compose

    def test_docker_compose_up(self, docker_compose):
        """Test that docker-compose services are up."""
        # Just verify compose is running
        assert docker_compose is not None

