"""Benchmark test comparing direct LLM intent detection vs semantic cache."""

import json
import os
import time
from typing import Any

import pytest

# Skip entire module if opt-in flag not set
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SDK_BENCHMARK_TEST") != "1",
    reason="Set RUN_SDK_BENCHMARK_TEST=1 to run this benchmark",
)

try:
    from testcontainers.redis import RedisContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    RedisContainer = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def redis_url():
    """
    Get Redis URL - use testcontainers if Docker available, else fallback to localhost.
    
    Set REDIS_URL env var to override (e.g., redis://localhost:6379).
    """
    # Allow override via environment
    env_url = os.environ.get("REDIS_URL")
    if env_url:
        yield env_url
        return

    # Try testcontainers if available and Docker is running
    if TESTCONTAINERS_AVAILABLE:
        try:
            with RedisContainer(image="redis/redis-stack:latest") as container:
                yield container.get_connection_url()
                return
        except Exception as e:
            print(f"Docker not available ({e}), falling back to localhost Redis")

    # Fallback to localhost
    yield "redis://localhost:6379"


@pytest.fixture(scope="module")
def cache(redis_url):
    """
    Create SemanticIntentCache using BGE-large embeddings (1024 dims).

    Setup:
      - Ingest "upgrade plan?" -> INTENT_UPGRADE_PLAN
    Teardown:
      - Delete the intent and close the cache
    """
    from semantic_intent_cache.sdk import SemanticIntentCache
    from semantic_intent_cache.embeddings.st_local import SentenceTransformerEmbedder
    from semantic_intent_cache.variants.builtin import BuiltinVariantProvider

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-large-en-v1.5")
    sdk = SemanticIntentCache(
        redis_url=redis_url,
        embedder=embedder,
        variant_provider=BuiltinVariantProvider(),
        vector_dim=1024,
    )
    sdk.ensure_index()

    # Ingest the test intent
    sdk.ingest(
        intent_id="INTENT_UPGRADE_PLAN",
        question="upgrade plan?",
        auto_variant_count=10,
    )

    yield sdk

    # Teardown: delete intent and close
    sdk.delete_intent("INTENT_UPGRADE_PLAN")
    sdk.close()


@pytest.fixture(scope="module")
def bedrock_client():
    """Create Bedrock client for direct LLM calls."""
    from semantic_intent_cache.embeddings.bedrock_client import BedrockClient
    from semantic_intent_cache.config import settings

    return BedrockClient(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        aws_region=settings.aws_region,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KNOWN_INTENTS = [
    "INTENT_UPGRADE_PLAN",
    "INTENT_CANCEL_SUBSCRIPTION",
    "INTENT_BILLING_INQUIRY",
    "INTENT_TECHNICAL_SUPPORT",
    "INTENT_UNKNOWN",
]


def detect_intent_via_llm(
    bedrock: Any,
    model_id: str,
    question: str,
) -> tuple[str, float]:
    """
    Detect intent by prompting the LLM with a system prompt.

    Returns:
        (detected_intent, latency_seconds)
    """
    system_prompt = f"""You are an intent classification assistant.
Given a user question, classify it into exactly ONE of the following intents:
{json.dumps(KNOWN_INTENTS)}

Respond with ONLY the intent name, nothing else. If uncertain, respond with INTENT_UNKNOWN.
"""

    prompt = f"""{system_prompt}

User question: {question}

Intent:"""

    request_body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    start = time.perf_counter()
    response = bedrock.client.invoke_model(
        modelId=model_id,
        body=request_body.encode("utf-8"),
        contentType="application/json",
    )
    elapsed = time.perf_counter() - start

    response_body = json.loads(response["body"].read())
    content = response_body.get("content", [{}])[0].get("text", "").strip()

    # Normalize response
    detected = content.upper().replace(" ", "_")
    if detected not in KNOWN_INTENTS:
        detected = "INTENT_UNKNOWN"

    return detected, elapsed


def detect_intent_via_cache(
    cache: Any,
    question: str,
    min_similarity: float = 0.6,
) -> tuple[str | None, float]:
    """
    Detect intent using semantic cache.

    Returns:
        (detected_intent or None, latency_seconds)
    """
    start = time.perf_counter()
    result = cache.match(query=question, top_k=1, min_similarity=min_similarity)
    elapsed = time.perf_counter() - start

    if result["match"]:
        return result["match"]["intent_id"], elapsed
    return None, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSDKvsLLMBenchmark:
    """Compare direct LLM intent detection vs semantic cache."""

    @pytest.fixture(autouse=True)
    def _store_results(self):
        """Store results for comparison."""
        self.results: dict[str, Any] = {}
        yield

    def test_1_llm_intent_detection(self, bedrock_client):
        """Test 1: Detect intent via direct LLM call."""
        from semantic_intent_cache.config import settings

        model_id = settings.anthropic_model
        question = "upgrade plan?"

        detected, latency = detect_intent_via_llm(bedrock_client, model_id, question)

        print("\n" + "=" * 60)
        print("TEST 1: Direct LLM Intent Detection")
        print("=" * 60)
        print(f"Question:        {question}")
        print(f"Detected Intent: {detected}")
        print(f"Expected Intent: INTENT_UPGRADE_PLAN")
        print(f"Latency:         {latency * 1000:.2f} ms")
        print(f"Correct:         {detected == 'INTENT_UPGRADE_PLAN'}")
        print("=" * 60)

        self.results["llm"] = {
            "detected": detected,
            "latency_ms": latency * 1000,
            "correct": detected == "INTENT_UPGRADE_PLAN",
        }

        # Store in class attribute for cross-test access
        TestSDKvsLLMBenchmark.llm_result = self.results["llm"]

        assert detected == "INTENT_UPGRADE_PLAN", f"LLM detected {detected}"

    def test_2_cache_intent_detection(self, cache):
        """Test 2: Detect intent via semantic cache."""
        question = "upgrade plan?"

        detected, latency = detect_intent_via_cache(cache, question)

        print("\n" + "=" * 60)
        print("TEST 2: Semantic Cache Intent Detection")
        print("=" * 60)
        print(f"Question:        {question}")
        print(f"Detected Intent: {detected}")
        print(f"Expected Intent: INTENT_UPGRADE_PLAN")
        print(f"Latency:         {latency * 1000:.2f} ms")
        print(f"Correct:         {detected == 'INTENT_UPGRADE_PLAN'}")
        print("=" * 60)

        self.results["cache"] = {
            "detected": detected,
            "latency_ms": latency * 1000,
            "correct": detected == "INTENT_UPGRADE_PLAN",
        }

        # Store in class attribute for cross-test access
        TestSDKvsLLMBenchmark.cache_result = self.results["cache"]

        assert detected == "INTENT_UPGRADE_PLAN", f"Cache detected {detected}"

    def test_3_comparison_report(self):
        """Test 3: Generate comparison report."""
        llm = getattr(TestSDKvsLLMBenchmark, "llm_result", None)
        cache = getattr(TestSDKvsLLMBenchmark, "cache_result", None)

        if not llm or not cache:
            pytest.skip("Previous tests did not complete")

        speedup = llm["latency_ms"] / cache["latency_ms"] if cache["latency_ms"] > 0 else 0

        print("\n" + "=" * 60)
        print("COMPARISON REPORT: LLM vs Semantic Cache")
        print("=" * 60)
        print(f"{'Metric':<25} {'LLM':<20} {'Cache':<20}")
        print("-" * 60)
        print(f"{'Detected Intent':<25} {llm['detected']:<20} {cache['detected']:<20}")
        print(f"{'Correct':<25} {str(llm['correct']):<20} {str(cache['correct']):<20}")
        print(f"{'Latency (ms)':<25} {llm['latency_ms']:<20.2f} {cache['latency_ms']:<20.2f}")
        print(f"{'Speedup':<25} {'1.0x':<20} {f'{speedup:.1f}x':<20}")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  - Cache is {speedup:.1f}x faster than direct LLM call")
        print(f"  - Both methods correctly identified the intent: {llm['correct'] and cache['correct']}")
        print("=" * 60)

        # Assertions
        assert llm["correct"], "LLM should detect correct intent"
        assert cache["correct"], "Cache should detect correct intent"
        assert speedup > 1, f"Cache should be faster than LLM (speedup={speedup:.2f}x)"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Set opt-in flag
    os.environ["RUN_SDK_BENCHMARK_TEST"] = "1"

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))

