"""Configuration management for semantic intent cache."""

from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"

    # Embedding configuration
    # Options: "st_local" (Sentence Transformers - local) or "titan" (AWS Bedrock Titan)
    embed_provider: Literal["st_local", "titan"] = "titan"
    # For st_local: sentence-transformers model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    # For titan: AWS Bedrock Titan model ID (e.g., "amazon.titan-embed-text-v1")
    embed_model_name: str = "amazon.titan-embed-text-v1"
    # Vector dimension must match your embedder:
    # - For titan: 1536 (amazon.titan-embed-text-v1)
    # - For st_local: typically 384 (all-MiniLM-L6-v2) or check your model's dimension
    vector_dim: int = 1536  # Default for Titan (1536), change to 384 for sentence-transformers

    # Variant provider configuration
    variant_provider: Literal["builtin", "anthropic"] = "anthropic"

    # AWS/Bedrock configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    bedrock_profile: str | None = None  # Optional, uses credentials chain if not set

    # Anthropic/Bedrock model configuration
    # Can be either a model ID or an inference profile ARN
    # Model ID format: anthropic.claude-3-haiku-20240307-v1:0
    # Inference profile ARN format: arn:aws:bedrock:REGION::inference-profile/MODEL_ID
    # Example ARN: arn:aws:bedrock:us-east-1::inference-profile/anthropic.claude-3-5-sonnet-20241022-v2:0
    # To use inference profiles: Set this to your inference profile ARN from AWS Bedrock Console
    anthropic_model: str = "anthropic.claude-3-haiku-20240307-v1:0"
    titan_embed_model: str = "amazon.titan-embed-text-v1"

    # API configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Redis index configuration
    index_name: str = "sc:idx"
    key_prefix: str = "sc:doc:"
    ef_construction: int = 200
    m: int = 16


# Global settings instance
settings = Settings()

