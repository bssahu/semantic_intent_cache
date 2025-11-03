"""Base protocol for embedding providers."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for embedding providers."""

    @property
    def dim(self) -> int:
        """Return the dimension of embeddings."""
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (N, dim) with normalized embeddings.
        """
        ...

