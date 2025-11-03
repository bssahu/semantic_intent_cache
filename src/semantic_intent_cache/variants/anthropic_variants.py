"""Anthropic/Bedrock variant provider (optional)."""

import json
import logging

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
        profile: str = "default",
    ):
        """
        Initialize the Anthropic variant provider.

        Args:
            aws_region: AWS region for Bedrock.
            model_id: Bedrock model ID.
            profile: AWS profile name.
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 not installed. Install with: pip install semantic-intent-cache[anthropic]"
            ) from e

        self.aws_region = aws_region
        self.model_id = model_id
        self.profile = profile

        # Initialize Bedrock client
        try:
            session = boto3.Session(profile_name=profile)
            self.client = session.client("bedrock-runtime", region_name=aws_region)
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
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
            # Prepare request body for Claude
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            # Invoke Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response
            try:
                response_body = json.loads(response["body"].read())

                # Extract text from Claude response
                content = ""
                for message in response_body.get("content", []):
                    if message.get("type") == "text":
                        content = message.get("text", "")
                        break

                # Parse variants
                variants = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip()
                    and not line.strip().startswith(("1.", "2.", "3.", "-", "*"))
                ]

                # Always include original
                if question not in variants:
                    variants.insert(0, question)

                # Limit to requested number
                return variants[:n]

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing Bedrock response: {e}")
                return [question]

        except Exception as e:
            logger.error(f"Error generating variants with Bedrock: {e}")
            # Fallback to original
            return [question]

    def __repr__(self) -> str:
        """String representation."""
        return f"AnthropicVariantProvider(model={self.model_id})"

