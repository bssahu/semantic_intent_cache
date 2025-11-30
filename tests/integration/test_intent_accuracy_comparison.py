"""Compare direct Anthropic intent detection vs semantic cache."""

from __future__ import annotations

import json
import os
import time
from statistics import mean
from typing import Iterable

import pytest

try:
    from testcontainers.redis import RedisContainer
except ImportError:
    RedisContainer = None

from semantic_intent_cache.config import settings
from semantic_intent_cache.embeddings.bedrock_client import BedrockClient
from semantic_intent_cache.sdk import SemanticIntentCache
from semantic_intent_cache.variants.builtin import BuiltinVariantProvider


RUN_ANTHROPIC_INTENT_TEST = os.getenv("RUN_ANTHROPIC_INTENT_TEST") == "1"

INTENT_FIXTURES = [
    {
        "intent_id": "UPGRADE_PLAN",
        "question": "How do I upgrade my plan?",
        "queries": [
            "Can you walk me through upgrading my subscription?",
            "I want to move to a higher plan tier.",
            "Show me how to switch to a premium plan.",
        ],
    },
    {
        "intent_id": "CANCEL_ACCOUNT",
        "question": "How can I cancel my account?",
        "queries": [
            "I'd like to terminate my subscription.",
            "Please help me close my account.",
            "What is the process for cancelling service?",
        ],
    },
    {
        "intent_id": "RESET_PASSWORD",
        "question": "How do I reset my password?",
        "queries": [
            "I forgot my password—how can I get back in?",
            "Walk me through setting a new password.",
            "What steps do I follow to recover my login?",
        ],
    },
]


def _build_intent_prompt(intents: Iterable[dict[str, str]], query: str) -> str:
    lines = ["You are an intent classification system.", "", "Available intents:"]
    for item in intents:
        lines.append(f"- {item['intent_id']}: {item['question']}")

    lines.extend(
        [
            "",
            "Respond **only** with valid JSON in the form:",
            '{"intent_id": "<INTENT_ID>"}',
            "Do not include any additional text or explanation.",
            "",
            f"User query: {query}",
        ]
    )

    return "\n".join(lines)


def _extract_json_intent(text: str) -> str | None:
    try:
        payload = json.loads(text.strip())
        value = payload.get("intent_id")
        if isinstance(value, str):
            return value.strip()
        return None
    except json.JSONDecodeError:
        # Best-effort fallback: attempt to locate JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                payload = json.loads(text[start : end + 1])
                value = payload.get("intent_id")
                if isinstance(value, str):
                    return value.strip()
            except json.JSONDecodeError:
            # If parsing still fails, give up gracefully
                return None
        return None


def _invoke_bedrock_intent(
    bedrock: BedrockClient,
    model_id: str,
    query: str,
    intents: list[dict[str, str]],
) -> str | None:
    prompt = _build_intent_prompt(intents, query)
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 200,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )

    response = bedrock.invoke_model(
        model_id=model_id,
        body=body,
        max_tokens=200,
        agent_name="IntentComparisonTest",
    )

    body_stream = response.get("body")
    if body_stream is None:
        return None

    payload = body_stream.read()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    data = json.loads(payload)
    content = data.get("content", [])
    if not content:
        return None

    text = content[0].get("text", "") if isinstance(content, list) else ""
    return _extract_json_intent(text)


def _evaluate_llm(
    bedrock: BedrockClient,
    model_id: str,
    fixtures: list[dict[str, object]],
) -> dict[str, float | int]:
    latencies: list[float] = []
    total = 0
    correct = 0

    for fixture in fixtures:
        intent_id = fixture["intent_id"]
        for query in fixture["queries"]:
            total += 1
            start = time.perf_counter()
            prediction = _invoke_bedrock_intent(bedrock, model_id, query, fixtures)
            latency = time.perf_counter() - start
            latencies.append(latency)

            if prediction == intent_id:
                correct += 1

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "avg_latency": mean(latencies) if latencies else 0.0,
    }


def _evaluate_cache(
    cache: SemanticIntentCache,
    fixtures: list[dict[str, object]],
) -> dict[str, float | int]:
    latencies: list[float] = []
    total = 0
    correct = 0

    for fixture in fixtures:
        intent_id = fixture["intent_id"]
        for query in fixture["queries"]:
            total += 1
            start = time.perf_counter()
            match = cache.match(query, top_k=1, min_similarity=0.0)
            latency = time.perf_counter() - start
            latencies.append(latency)

            predicted = match["match"]["intent_id"] if match["match"] else None
            if predicted == intent_id:
                correct += 1

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "avg_latency": mean(latencies) if latencies else 0.0,
    }


@pytest.mark.integration
@pytest.mark.skipif(RedisContainer is None, reason="testcontainers not available")
@pytest.mark.skipif(
    not RUN_ANTHROPIC_INTENT_TEST,
    reason="Set RUN_ANTHROPIC_INTENT_TEST=1 to enable Anthropic comparison test",
)
def test_llm_vs_cache_intent_detection():
    """Compare direct Anthropic intent detection vs semantic cache."""
    with RedisContainer(image="redis/redis-stack:latest") as redis:
        redis_url = redis.get_connection_url()
        settings.redis_url = redis_url

        bedrock = BedrockClient(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_region=settings.aws_region,
        )
        model_id = settings.anthropic_model

        cache = SemanticIntentCache(
            redis_url=redis_url,
            index_name="sc:test:intent-comparison",
            key_prefix="sc:test:intent:",
            variant_provider=BuiltinVariantProvider(),
        )

        # Ensure clean slate
        cache.store.client.flushdb()
        cache.ensure_index()

        for fixture in INTENT_FIXTURES:
            cache.ingest(
                intent_id=fixture["intent_id"],
                question=fixture["question"],
                auto_variant_count=len(fixture["queries"]) + 1,
                variants=list(fixture["queries"]),
            )

        llm_metrics = _evaluate_llm(bedrock, model_id, INTENT_FIXTURES)
        cache_metrics = _evaluate_cache(cache, INTENT_FIXTURES)

        cache.close()

        report = {
            "anthropic_model_id": model_id,
            "samples": llm_metrics["total_samples"],
            "llm": llm_metrics,
            "semantic_cache": cache_metrics,
        }

        print("\nIntent Detection Comparison Report")
        print(json.dumps(report, indent=2, sort_keys=True))

        assert llm_metrics["total_samples"] == cache_metrics["total_samples"]
        assert 0.0 <= llm_metrics["accuracy"] <= 1.0
        assert 0.0 <= cache_metrics["accuracy"] <= 1.0
        assert llm_metrics["avg_latency"] > 0.0
        assert cache_metrics["avg_latency"] > 0.0

