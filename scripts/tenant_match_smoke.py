"""Simple REST client to verify tenant-scoped ingest and match."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx


def wait_for_api(base_url: str, timeout: float = 10.0) -> None:
    """Wait until `/healthz` responds or raise after timeout."""
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/healthz"
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("healthy") is True:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"API at {url} did not become healthy within {timeout}s")


def pretty(data: Any) -> str:
    """Return deterministic pretty JSON-like string."""
    import json

    return json.dumps(data, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for tenant-aware ingest/match workflow."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL for the Semantic Intent Cache API.",
    )
    parser.add_argument(
        "--intent-id",
        default="UPGRADE_PLAN",
        help="Intent identifier to use for the smoke test.",
    )
    parser.add_argument(
        "--tenant",
        default="acme-support",
        help="Tenant identifier for the smoke test payloads.",
    )
    parser.add_argument(
        "--question",
        default="How do I upgrade my plan?",
        help="Canonical question text for the smoke test.",
    )
    parser.add_argument(
        "--query",
        default="how to get a higher plan?",
        help="Match query text to validate retrieval.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.6,
        help="Minimum similarity threshold for the match call.",
    )

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    ingest_url = f"{base_url}/cache/ingest"
    match_url = f"{base_url}/cache/match"
    variants_url = f"{base_url}/cache/variants/{args.intent_id}"

    print(f"→ Waiting for API health at {base_url} ...", flush=True)
    wait_for_api(base_url)
    print("✓ API healthy")

    ingest_payload = {
        "intent_id": args.intent_id,
        "question": args.question,
        "auto_variant_count": 8,
        "tenant": args.tenant,
    }

    print(f"\n→ Ingesting intent via {ingest_url}")
    ingest_resp = httpx.post(ingest_url, json=ingest_payload, timeout=15.0)
    ingest_resp.raise_for_status()
    ingest_data = ingest_resp.json()
    print(pretty(ingest_data))

    print(f"\n→ Fetching stored variants via {variants_url}")
    variants_resp = httpx.get(variants_url, timeout=10.0)
    variants_resp.raise_for_status()
    variants_data = variants_resp.json()
    print(pretty(variants_data))

    match_payload = {
        "query": args.query,
        "top_k": 5,
        "min_similarity": args.min_similarity,
        "tenant": args.tenant,
    }

    print(f"\n→ Matching query via {match_url}")
    match_resp = httpx.post(match_url, json=match_payload, timeout=15.0)
    match_resp.raise_for_status()
    match_data = match_resp.json()
    print(pretty(match_data))

    match = match_data.get("match")
    if not match:
        print("\n✗ No match returned", file=sys.stderr)
        return 1

    if match.get("intent_id") != args.intent_id:
        print(
            f"\n✗ Unexpected match intent: {match.get('intent_id')} (expected {args.intent_id})",
            file=sys.stderr,
        )
        return 1

    similarity = match.get("similarity", 0.0)
    if similarity < args.min_similarity:
        print(
            f"\n✗ Similarity {similarity:.3f} below threshold {args.min_similarity:.3f}",
            file=sys.stderr,
        )
        return 1

    print(f"\n✓ Match succeeded with similarity {similarity:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

