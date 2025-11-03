"""Unit tests for embedding providers."""

import numpy as np

from semantic_intent_cache.embeddings.st_local import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    """Test sentence transformer embedder."""

    def test_encode_shape(self):
        """Test that encode returns correct shape."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world", "test embedding"]

        embeddings = embedder.encode(texts)

        assert embeddings.shape == (2, embedder.dim)

    def test_encode_single_text(self):
        """Test encoding of single text."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world"]

        embeddings = embedder.encode(texts)

        assert embeddings.ndim == 2
        assert embeddings.shape == (1, embedder.dim)

    def test_encode_empty_list(self):
        """Test encoding of empty list."""
        embedder = SentenceTransformerEmbedder()
        texts = []

        embeddings = embedder.encode(texts)

        assert embeddings.shape == (0,)

    def test_encode_normalization(self):
        """Test that embeddings are L2 normalized."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world", "test embedding", "another text"]

        embeddings = embedder.encode(texts)

        # Check each embedding is approximately unit norm
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, rtol=0.01), f"Norm: {norm}"

    def test_encode_dtype(self):
        """Test that embeddings have correct dtype."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world"]

        embeddings = embedder.encode(texts)

        assert embeddings.dtype == np.float32

    def test_dim_property(self):
        """Test dim property."""
        embedder = SentenceTransformerEmbedder()

        dim = embedder.dim

        assert isinstance(dim, int)
        assert dim > 0
        # all-MiniLM-L6-v2 has 384 dimensions
        assert dim == 384

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world", "completely different text"]

        embeddings = embedder.encode(texts)

        # Should be different (with high probability)
        cosine_sim = np.dot(embeddings[0], embeddings[1])
        # Should not be too similar for completely different texts
        assert abs(cosine_sim) < 0.9

    def test_similar_texts_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world", "hello world"]

        embeddings = embedder.encode(texts)

        # Should be identical
        cosine_sim = np.dot(embeddings[0], embeddings[1])
        assert np.isclose(cosine_sim, 1.0, rtol=0.01)

    def test_repr(self):
        """Test string representation."""
        embedder = SentenceTransformerEmbedder()
        assert "SentenceTransformerEmbedder" in repr(embedder)
        assert "dim=384" in repr(embedder)

