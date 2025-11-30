"""Anthropic/Bedrock variant provider (optional)."""

import logging
import json

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
            request_body = json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "anthropic_version": "bedrock-2023-05-31",
                }
            )

            response = self.bedrock_client.invoke_model(
                model_id=self.model_id,
                body=request_body,
                max_tokens=1000,
                agent_name="AnthropicVariantProvider",
            )

            raw_body = response.get("body")
            if raw_body is None:
                logger.warning("No body in Bedrock response")
                return [question]

            if hasattr(raw_body, "read"):
                raw_body = raw_body.read()

            if isinstance(raw_body, bytes):
                raw_body = raw_body.decode("utf-8")

            response_json = json.loads(raw_body)
            content_blocks = response_json.get("content", [])

            if not content_blocks:
                logger.warning("No content received from Bedrock response")
                return [question]

            content = content_blocks[0].get("text", "") if isinstance(content_blocks, list) else ""
            logger.debug(f"Extracted content from Claude: {content[:200]}...")

            # Parse variants - split by newlines and filter empty lines and list markers
            variants = []
            for line in content.split("\n"):
                line = line.strip()
                lower_line = line.lower()

                if lower_line.startswith(("here are", "paraphrases", "original question")):
                    continue

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

