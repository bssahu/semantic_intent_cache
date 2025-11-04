"""CLI for semantic intent cache."""

import logging
import sys

import typer

from semantic_intent_cache.config import settings

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="semantic-intent-cache",
    help="Semantic Intent Cache CLI",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host to bind"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """
    Start the FastAPI server.

    Example:
        semantic-intent-cache serve --host 0.0.0.0 --port 8080
    """
    import uvicorn

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "semantic_intent_cache.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def ingest(
    intent: str = typer.Option(..., "--intent", "-i", help="Intent identifier"),
    question: str = typer.Option(..., "--question", "-q", help="Question text"),
    auto_variants: int = typer.Option(
        12,
        "--auto-variants",
        "-v",
        help="Number of variants to auto-generate",
    ),
    redis_url: str = typer.Option(None, "--redis-url", "-r", help="Redis URL"),
) -> None:
    """
    Ingest an intent with variants.

    Example:
        semantic-intent-cache ingest --intent UPGRADE_PLAN --question "How do I upgrade my plan?" --auto-variants 12
    """
    from semantic_intent_cache.sdk import SemanticIntentCache

    try:
        sdk = SemanticIntentCache(redis_url=redis_url)
        sdk.ensure_index()

        result = sdk.ingest(
            intent_id=intent,
            question=question,
            auto_variant_count=auto_variants,
        )

        print(f"✓ Ingested intent: {result['intent_id']}")
        print(f"  Stored variants: {result['stored_variants']}")
        print(f"  Total generated: {result['total_generated']}")

        sdk.close()
    except Exception as e:
        logger.error(f"Error ingesting intent: {e}")
        sys.exit(1)


@app.command()
def match(
    query: str = typer.Option(..., "--query", "-q", help="Query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    min_sim: float = typer.Option(
        0.8,
        "--min-sim",
        "-m",
        help="Minimum similarity threshold",
    ),
    redis_url: str = typer.Option(None, "--redis-url", "-r", help="Redis URL"),
) -> None:
    """
    Match a query against stored intents.

    Example:
        semantic-intent-cache match --query "Change to higher tier" --top-k 5 --min-sim 0.8
    """
    from semantic_intent_cache.sdk import SemanticIntentCache

    try:
        sdk = SemanticIntentCache(redis_url=redis_url)

        result = sdk.match(
            query=query,
            top_k=top_k,
            min_similarity=min_sim,
        )

        if result["match"]:
            print("✓ Best match:")
            print(f"  Intent: {result['match']['intent_id']}")
            print(f"  Question: {result['match']['question']}")
            print(f"  Similarity: {result['match']['similarity']:.3f}")

            if result["alternates"]:
                print("\nAlternates:")
                for i, alt in enumerate(result["alternates"], 1):
                    print(f"  {i}. {alt['intent_id']}: {alt['similarity']:.3f}")
        else:
            print("✗ No match found")

        sdk.close()
    except Exception as e:
        logger.error(f"Error matching query: {e}")
        sys.exit(1)


@app.command()
def variants(
    intent: str = typer.Option(..., "--intent", "-i", help="Intent identifier"),
    redis_url: str = typer.Option(None, "--redis-url", "-r", help="Redis URL"),
) -> None:
    """List all variants for an intent."""
    try:
        from semantic_intent_cache.sdk import SemanticIntentCache

        sdk = SemanticIntentCache(redis_url=redis_url)

        variant_list = sdk.get_variants(intent)

        if variant_list:
            print(f"✓ Found {len(variant_list)} variants for '{intent}':\n")
            for i, v in enumerate(variant_list, 1):
                print(f"  {i}. {v['text']}")
        else:
            print(f"✗ No variants found for intent '{intent}'")

        sdk.close()
    except Exception as e:
        logger.error(f"Error retrieving variants: {e}")
        sys.exit(1)


@app.command()
def delete(
    intent: str = typer.Option(..., "--intent", "-i", help="Intent identifier to delete"),
    redis_url: str = typer.Option(None, "--redis-url", "-r", help="Redis URL"),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """Delete an intent and all its variants."""
    try:
        from semantic_intent_cache.sdk import SemanticIntentCache

        sdk = SemanticIntentCache(redis_url=redis_url)

        # Check if intent exists
        variant_list = sdk.get_variants(intent)
        variant_count = len(variant_list)

        if variant_count == 0:
            print(f"✗ No variants found for intent '{intent}'")
            sdk.close()
            return

        # Confirm deletion unless --confirm flag is used
        if not confirm:
            typer.confirm(
                f"Delete intent '{intent}' with {variant_count} variant(s)? This cannot be undone.",
                abort=True,
            )

        # Delete the intent
        deleted_count = sdk.delete_intent(intent)

        if deleted_count > 0:
            print(f"✓ Deleted intent '{intent}' with {deleted_count} variant(s)")
        else:
            print(f"✗ No variants deleted for intent '{intent}'")

        sdk.close()
    except typer.Abort:
        print("Deletion cancelled")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error deleting intent: {e}")
        sys.exit(1)


@app.command()
def info() -> None:
    """Display configuration information."""
    print(f"Redis URL: {settings.redis_url}")
    print(f"Index name: {settings.index_name}")
    print(f"Embed model: {settings.embed_model_name}")
    print(f"Vector dimension: {settings.vector_dim}")
    print(f"Variant provider: {settings.variant_provider}")


if __name__ == "__main__":
    app()

