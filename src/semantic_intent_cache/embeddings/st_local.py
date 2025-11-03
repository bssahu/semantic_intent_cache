"""Sentence Transformers local embedding provider."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Local embedding provider using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence transformer model to use.
        """
        self.model_name = model_name
        logger.info(f"Loading sentence transformer model: {model_name}")
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            # Get dimension from model
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        """Return the dimension of embeddings."""
        if self._dim is None:
            self._dim = self.model.get_sentence_embedding_dimension()
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (N, dim) with L2-normalized embeddings.
        """
        if not texts:
            return np.array([])

        # Encode using sentence transformers
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )

        # Ensure it's a 2D array even for single text
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    def __repr__(self) -> str:
        """String representation."""
        return f"SentenceTransformerEmbedder(model={self.model_name}, dim={self.dim})"

