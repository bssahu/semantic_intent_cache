# Semantic Intent Cache

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/license/bssahu/semantic_intent_cache)](https://github.com/bssahu/semantic_intent_cache)

**Production-ready Python SDK for semantic intent matching** built on Redis Stack (RediSearch + HNSW vectors). Store questions with intent IDs, auto-generate semantic variants, and retrieve best matches using cosine similarity search.

> **Built with**: FastAPI | Redis Stack | Sentence Transformers | Anthropic Claude 3.7

## 🎯 What/Why

Semantic Intent Cache enables **intent classification** for conversational systems without complex ML pipelines:

- **Fast retrieval**: Sub-millisecond vector search with Redis Stack HNSW
- **High recall**: Auto-generate semantic variants to capture paraphrases
- **Zero training**: Works out-of-the-box with pre-trained embeddings
- **Production-ready**: Comprehensive tests, Docker support, FastAPI service
- **Pluggable**: Swap embedders (local SentenceTransformers) and variant providers (builtin or Anthropic/Bedrock)

### Use Cases

- **FAQ matching**: Map user questions to canonical answers
- **Intent routing**: Classify user intents in chatbots
- **Content recommendation**: Find similar queries or articles
- **A/B testing**: Track intent distribution across experiments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                                │
│                      "upgrade plan?"                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SemanticIntentCache                           │
│  ┌────────────────────┐         ┌──────────────────────────┐   │
│  │  VariantProvider   │────────▶│  Generate 10 variants     │   │
│  │ (builtin/Anthropic)│         │  - "How do I upgrade?"    │   │
│  └────────────────────┘         │  - "I want to upgrade?"   │   │
│                                  │  - "Steps to upgrade?"    │   │
│  ┌────────────────────┐         └──────────────────────────┘   │
│  │    Embedder        │         ┌──────────────────────────┐   │
│  │ (SentenceTransform)│────────▶│  Encode to 384-D vectors │   │
│  └────────────────────┘         └──────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Redis Stack                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RediSearch Vector Index (HNSW)                          │   │
│  │  - Index: sc:idx                                          │   │
│  │  - Keys:  sc:doc:INTENT:0, sc:doc:INTENT:1, ...          │   │
│  │  - Schema: intent (TAG), text (TEXT), embedding (VECTOR) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quickstart

### 1. Start Redis Stack

```bash
docker compose up -d redis
```

This starts Redis Stack on `localhost:6379` with Redis Insight UI at http://localhost:8001.

### 2. Install SDK

**Using pip (from source):**
```bash
git clone https://github.com/bssahu/semantic_intent_cache.git
cd semantic_intent_cache
pip install -r requirements.txt
# Or with dev dependencies:
pip install -r requirements-dev.txt
# Or with Anthropic/Bedrock support:
pip install -r requirements-anthropic.txt
```

**Using editable install:**
```bash
git clone https://github.com/bssahu/semantic_intent_cache.git
cd semantic_intent_cache
pip install -e ".[dev]"  # with dev tools
# or
pip install -e ".[anthropic]"  # with Anthropic support
```

### 3. Start API Server

```bash
semantic-intent-cache serve
```

Or with uvicorn directly:

```bash
uvicorn semantic_intent_cache.api.app:app --reload --port 8080
```

### 4. Ingest Intents

```bash
curl -X POST http://localhost:8080/cache/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "intent_id": "UPGRADE_PLAN",
    "question": "How do I upgrade my plan?",
    "auto_variant_count": 12
  }'
```

Response:

```json
{
  "status": "ok",
  "intent_id": "UPGRADE_PLAN",
  "stored_variants": 12
}
```

### 5. Match Queries

```bash
curl -X POST http://localhost:8080/cache/match \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to change to a higher tier",
    "top_k": 5,
    "min_similarity": 0.75
  }'
```

Response:

```json
{
  "match": {
    "intent_id": "UPGRADE_PLAN",
    "question": "How do I upgrade my plan?",
    "similarity": 0.87
  },
  "alternates": [
    {
      "intent_id": "UPGRADE_PLAN",
      "question": "Steps to upgrade my plan?",
      "similarity": 0.82
    }
  ]
}
```

### 6. View Variants for an Intent

```bash
curl http://localhost:8080/cache/variants/UPGRADE_PLAN
```

Response:

```json
{
  "intent_id": "UPGRADE_PLAN",
  "variants": [
    "How do I upgrade my plan?",
    "Can you tell me how do i upgrade my plan?",
    "I want to how do i upgrade my plan?",
    "Please help me with how do i upgrade my plan?"
  ],
  "count": 12
}
```

## 📚 SDK Usage

### Python SDK

```python
from semantic_intent_cache import SemanticIntentCache

# Initialize SDK
with SemanticIntentCache() as cache:
    # Ensure index exists
    cache.ensure_index()
    
    # Ingest an intent with 8 auto-generated variants
    result = cache.ingest(
        intent_id="UPGRADE_PLAN",
        question="How do I upgrade my plan?",
        auto_variant_count=8
    )
    print(f"Stored {result['stored_variants']} variants")
    
    # Match a query
    match = cache.match(
        query="Move to higher tier",
        min_similarity=0.8
    )
    
    if match["match"]:
        print(f"Matched intent: {match['match']['intent_id']}")
        print(f"Similarity: {match['match']['similarity']:.3f}")
    
    # View all variants for an intent
    variants = cache.get_variants("UPGRADE_PLAN")
    print(f"Stored variants: {len(variants)}")
    for v in variants[:3]:
        print(f"  - {v['text']}")
    
    # Delete an intent and all its variants
    deleted_count = cache.delete_intent("UPGRADE_PLAN")
    print(f"Deleted {deleted_count} variants")
```

### CLI

```bash
# Start server
semantic-intent-cache serve --host 0.0.0.0 --port 8080

# Ingest intent
semantic-intent-cache ingest \
  --intent UPGRADE_PLAN \
  --question "How do I upgrade my plan?" \
  --auto-variants 12

# Match query
semantic-intent-cache match \
  --query "Change to higher tier" \
  --top-k 5 \
  --min-sim 0.8

# List variants for an intent
semantic-intent-cache variants --intent UPGRADE_PLAN

# Delete an intent and all its variants
semantic-intent-cache delete --intent UPGRADE_PLAN
# Or skip confirmation prompt:
semantic-intent-cache delete --intent UPGRADE_PLAN --confirm

# Show config
semantic-intent-cache info
```

## ⚙️ Configuration

Environment variables (see `.env.example`):

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# Embeddings
# Option 1: AWS Bedrock Titan (requires AWS credentials)
EMBED_PROVIDER=titan
EMBED_MODEL_NAME=amazon.titan-embed-text-v1
VECTOR_DIM=1536  # Titan v1 embeddings are 1536 dimensions

# Option 2: Sentence Transformers (local, no AWS required)
# EMBED_PROVIDER=st_local
# EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
# VECTOR_DIM=384  # all-MiniLM-L6-v2 embeddings are 384 dimensions

# Variants
VARIANT_PROVIDER=anthropic  # or builtin

# AWS/Bedrock Configuration (required for titan embeddings and anthropic variants)
AWS_REGION=us-east-1
# Optional: AWS credentials (if not provided, uses default credential chain)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
ANTHROPIC_MODEL=anthropic.claude-3-haiku-20240307-v1:0
TITAN_EMBED_MODEL=amazon.titan-embed-text-v1

# Index
INDEX_NAME=sc:idx
KEY_PREFIX=sc:doc:
EF_CONSTRUCTION=200
M=16
```

### Switching Embedding Providers

You can easily switch between AWS Bedrock Titan and local Sentence Transformers:

**For AWS Bedrock Titan (current default):**
```bash
EMBED_PROVIDER=titan
EMBED_MODEL_NAME=amazon.titan-embed-text-v1
VECTOR_DIM=1536
```
- ✅ Cloud-based, always up-to-date
- ✅ No local model downloads
- ⚠️ Requires AWS credentials and Bedrock access

**For Sentence Transformers (local):**
```bash
EMBED_PROVIDER=st_local
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIM=384
```
- ✅ Works offline, no AWS required
- ✅ Faster for local development
- ⚠️ Requires downloading model on first use (~90MB)
- ⚠️ Model is cached locally

**Important:** When switching providers, you must:
1. Update `EMBED_PROVIDER`, `EMBED_MODEL_NAME`, and `VECTOR_DIM`
2. Drop and recreate the Redis index (dimensions must match)
3. Re-ingest all documents

### Custom Providers

You can customize embedding and variant providers programmatically:

**Custom Embedding Providers:**

```python
from semantic_intent_cache import SemanticIntentCache
from semantic_intent_cache.embeddings.titan_embedder import TitanEmbedder
from semantic_intent_cache.embeddings.st_local import SentenceTransformerEmbedder

# Use Titan embeddings with custom configuration
cache = SemanticIntentCache(
    embedder=TitanEmbedder(
        model_id="amazon.titan-embed-text-v1",
        aws_region="us-east-1",
        vector_dim=1536
    ),
    vector_dim=1536
)

# Use Sentence Transformers with a different model
cache = SemanticIntentCache(
    embedder=SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
    ),
    vector_dim=768
)
```

**Custom Variant Providers:**

```python
from semantic_intent_cache import SemanticIntentCache
from semantic_intent_cache.variants.anthropic_variants import AnthropicVariantProvider

# Use Anthropic/Bedrock for variants (requires anthropic extra)
cache = SemanticIntentCache(
    variant_provider=AnthropicVariantProvider(
        aws_region="us-east-1",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        aws_access_key_id="your_key",  # Optional
        aws_secret_access_key="your_secret"  # Optional
    )
)
```

## 🎛️ Performance Tips

### HNSW Tuning

Adjust `EF_CONSTRUCTION` and `M` for speed vs. accuracy:

- **M=16, EF_CONSTRUCTION=200**: Default (balanced)
- **M=32, EF_CONSTRUCTION=500**: Higher accuracy, slower
- **M=8, EF_CONSTRUCTION=100**: Faster, lower accuracy

```python
cache = SemanticIntentCache(
    m=32,
    ef_construction=500
)
```

### Batching

For bulk ingest, call `ingest()` multiple times (Redis pipelining handles it automatically). Future: `/cache/bulk-ingest` endpoint.

### Threshold Tuning

Start with `min_similarity=0.75-0.85`:

- **0.85-0.95**: Very precise, may miss paraphrases
- **0.75-0.85**: Balanced
- **0.60-0.75**: Catch more variants, possible false positives

### Redis Optimization

- **Enable persistence**: `redis-stack-server --appendonly yes`
- **Tune memory**: Adjust `maxmemory` and eviction policy
- **Use redis-cluster**: For multi-node deployments

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Integration Tests (requires Docker)

```bash
pytest tests/integration/ -v -m integration
```

All tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src/semantic_intent_cache --cov-report=html
```

## 🐳 Docker Deployment

### Development

```bash
docker compose up -d
```

Services:
- Redis Stack: `localhost:6379`
- Redis Insight: http://localhost:8001
- API: http://localhost:8080

### Production Build

```bash
docker build -t semantic-intent-cache:latest .
docker run -p 8080:8080 \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  semantic-intent-cache:latest
```

## 📋 API Reference

### `POST /cache/ingest`

Store an intent with semantic variants.

**Request:**

```json
{
  "intent_id": "UPGRADE_PLAN",
  "question": "How do I upgrade my plan?",
  "auto_variant_count": 12,
  "variants": []
}
```

**Response:**

```json
{
  "status": "ok",
  "intent_id": "UPGRADE_PLAN",
  "stored_variants": 12
}
```

### `POST /cache/match`

Search for best matching intent.

**Request:**

```json
{
  "query": "I want to change to a higher tier",
  "top_k": 5,
  "min_similarity": 0.75
}
```

**Response:**

```json
{
  "match": {
    "intent_id": "UPGRADE_PLAN",
    "question": "How do I upgrade my plan?",
    "similarity": 0.87
  },
  "alternates": []
}
```

### `GET /cache/variants/{intent_id}`

Retrieve all variant texts for an intent.

**Response:**

```json
{
  "intent_id": "UPGRADE_PLAN",
  "variants": [
    "How do I upgrade my plan?",
    "How can I change my subscription?",
    "What's the process for upgrading?"
  ],
  "count": 3
}
```

### `DELETE /cache/intent/{intent_id}`

Delete an intent and all its variants.

**Response:**

```json
{
  "intent_id": "UPGRADE_PLAN",
  "deleted_count": 5,
  "message": "Successfully deleted intent 'UPGRADE_PLAN' with 5 variant(s)"
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8080/cache/intent/UPGRADE_PLAN
```

### `GET /healthz`

Health check.

**Response:**

```json
{
  "status": "ok",
  "healthy": true
}
```

## 🔮 Roadmap

- [ ] Bulk ingest endpoint (`POST /cache/bulk-ingest`)
- [ ] Export/import intents (JSONL)
- [ ] TTL-based hot query cache for exact repeats
- [ ] Prometheus metrics endpoint
- [ ] Multi-tenant filters
- [ ] Async SDK methods
- [ ] Vector dimension auto-detection

## 🤝 Contributing

Contributions welcome! Please open an [issue](https://github.com/bssahu/semantic_intent_cache/issues) or [PR](https://github.com/bssahu/semantic_intent_cache/pulls).

Development setup:

```bash
git clone https://github.com/bssahu/semantic_intent_cache.git
cd semantic_intent_cache
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Linting:

```bash
ruff check .
black --check src/ tests/
mypy src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Redis Stack](https://redis.io/docs/stack/) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Anthropic Claude](https://www.anthropic.com/claude) for semantic variant generation

---

**Made with ❤️ for production-ready semantic search**

