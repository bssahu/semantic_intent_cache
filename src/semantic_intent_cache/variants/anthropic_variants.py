"""Anthropic/Bedrock variant provider (optional)."""

import asyncio
import logging

from semantic_intent_cache.embeddings.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


class AnthropicVariantProvider:
    """
    Anthropic variant provider using Bedrock API.

    Requires the 'anthropic' extra to be installed:
    pip install semantic-intent-cache[anthropic]
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        profile: str | None = None,
    ):
        """
        Initialize the Anthropic variant provider.

        Args:
            aws_region: AWS region for Bedrock.
            model_id: Bedrock model ID.
            aws_access_key_id: AWS access key ID (optional, uses credentials chain if not provided).
            aws_secret_access_key: AWS secret access key (optional, uses credentials chain if not provided).
            profile: AWS profile name (optional, deprecated - use credentials or default chain).
        """
        self.model_id = model_id

        # Initialize Bedrock client
        try:
            # Use explicit credentials if provided, otherwise use credentials chain
            if profile:
                logger.warning("profile parameter is deprecated, use aws_access_key_id/aws_secret_access_key or default credentials chain")
                # Fallback: try to use boto3 session for profile
                try:
                    import boto3
                    session = boto3.Session(profile_name=profile)
                    # Extract credentials from session
                    credentials = session.get_credentials()
                    if credentials:
                        aws_access_key_id = credentials.access_key
                        aws_secret_access_key = credentials.secret_key
                except Exception as e:
                    logger.warning(f"Could not use profile {profile}: {e}")

            self.bedrock_client = BedrockClient(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
            )
            logger.info(f"Initialized Anthropic variant provider with model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic variant provider: {e}")
            raise

    def generate(self, question: str, n: int) -> list[str]:
        """
        Generate semantic variants using Anthropic Claude via Bedrock.

        Args:
            question: The original question.
            n: Number of variants to generate.

        Returns:
            List of variant questions.
        """
        prompt = f"""Generate {n} distinct paraphrases of the following question, each expressing the same intent but with different wording and structure. Return only the paraphrases, one per line, without numbering or bullets.

Original question: {question}

Paraphrases:"""

        try:
            # Invoke Bedrock using async method (run in sync context)
            # Use asyncio.run() to execute the async call
            content = asyncio.run(
                self.bedrock_client.invoke_model_async(
                    model_id=self.model_id,
                    prompt=prompt,
                    max_tokens=1000,
                    agent_name="AnthropicVariantProvider",
                    temperature=0.7,
                    top_p=0.9,
                )
            )

            if not content:
                logger.warning(f"No content received from Bedrock response")
                return [question]

            logger.debug(f"Extracted content from Claude: {content[:200]}...")

            # Parse variants - split by newlines and filter empty lines and list markers
            variants = []
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "*", "•")):
                    # Clean up any remaining prefixes
                    line = line.lstrip("0123456789. -•*").strip()
                    if line:
                        variants.append(line)

            logger.info(f"Generated {len(variants)} variants from Claude response")

            # Always include original
            if question not in variants:
                variants.insert(0, question)

            # Limit to requested number
            result = variants[:n]
            logger.info(f"Returning {len(result)} variants (requested {n})")
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error generating variants with Bedrock: {error_msg}\n"
                f"Question: {question}, Requested variants: {n}\n"
                f"Falling back to original question only.",
                exc_info=True,
            )
            # Fallback to original
            return [question]

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicVariantProvider(model={self.model_id})"

