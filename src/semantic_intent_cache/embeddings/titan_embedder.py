"""Amazon Titan embedding provider via Bedrock."""

import json
import logging

import numpy as np

from semantic_intent_cache.embeddings.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


class TitanEmbedder:
    """AWS Bedrock Titan embedding provider."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str = "us-east-1",
        vector_dim: int = 1024,
    ):
        """
        Initialize the Titan embedder.

        Args:
            model_id: Bedrock Titan model ID (default: amazon.titan-embed-text-v1).
            aws_access_key_id: AWS access key ID (optional).
            aws_secret_access_key: AWS secret access key (optional).
            aws_region: AWS region for Bedrock.
            vector_dim: Dimension of embeddings (default: 1024 for Titan v1).
        """
        self.model_id = model_id
        self._dim = vector_dim

        # Initialize Bedrock client
        try:
            self.bedrock_client = BedrockClient(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
            )
            logger.info(f"Initialized Titan embedder with model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Titan embedder: {e}")
            raise

    @property
    def dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embeddings using Titan.

        Args:
            texts: List of texts to encode.

        Returns:
            Numpy array of embeddings with shape (len(texts), dim).
        """
        embeddings = []

        for text in texts:
            try:
                # Prepare Titan embedding request
                body = json.dumps({"inputText": text})

                # Invoke Bedrock
                response = self.bedrock_client.invoke_model(
                    model_id=self.model_id,
                    body=body,
                    agent_name="TitanEmbedder",
                )

                # Parse response
                response_body = json.loads(response["body"].read())
                embedding = np.array(response_body["embedding"], dtype=np.float32)

                embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Error encoding text with Titan: {e}")
                # Return zero vector on error
                embeddings.append(np.zeros(self._dim, dtype=np.float32))

        return np.array(embeddings)
