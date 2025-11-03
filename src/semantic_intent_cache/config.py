"""Configuration management for semantic intent cache."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"

    # Embedding configuration
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int = 384

    # Variant provider configuration
    variant_provider: Literal["builtin", "anthropic"] = "builtin"

    # Anthropic/Bedrock configuration
    aws_region: str = "us-east-1"
    anthropic_model: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    bedrock_profile: str = "default"

    # API configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Redis index configuration
    index_name: str = "sc:idx"
    key_prefix: str = "sc:doc:"
    ef_construction: int = 200
    m: int = 16

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

