"""Unit tests for variant providers."""


from semantic_intent_cache.variants.builtin import BuiltinVariantProvider


class TestBuiltinVariantProvider:
    """Test builtin variant provider."""

    def test_generate_includes_original(self):
        """Test that generated variants include the original question."""
        provider = BuiltinVariantProvider(seed=42)
        question = "How do I upgrade my plan?"
        variants = provider.generate(question, n=5)

        assert len(variants) <= 5
        assert question in variants

    def test_generate_deterministic(self):
        """Test that generation is deterministic with same seed."""
        provider1 = BuiltinVariantProvider(seed=42)
        provider2 = BuiltinVariantProvider(seed=42)
        question = "How do I upgrade my plan?"

        variants1 = provider1.generate(question, n=10)
        variants2 = provider2.generate(question, n=10)

        assert variants1 == variants2

    def test_generate_uniqueness(self):
        """Test that generated variants are unique."""
        provider = BuiltinVariantProvider(seed=42)
        question = "How do I upgrade my plan?"
        variants = provider.generate(question, n=20)

        # All should be unique
        assert len(variants) == len(set(variants))

    def test_generate_caps_at_n(self):
        """Test that generation caps at requested n."""
        provider = BuiltinVariantProvider(seed=42)
        question = "How do I upgrade my plan?"

        # Request more than we can generate
        variants = provider.generate(question, n=1000)

        # Should be bounded by templates + rules
        assert len(variants) > 0
        # Realistic upper bound for builtin
        assert len(variants) <= 100

    def test_generate_empty_question(self):
        """Test handling of empty question."""
        provider = BuiltinVariantProvider(seed=42)
        variants = provider.generate("", n=5)

        assert len(variants) > 0

    def test_generate_single_char(self):
        """Test handling of single character."""
        provider = BuiltinVariantProvider(seed=42)
        variants = provider.generate("a", n=5)

        assert len(variants) > 0

    def test_generate_with_synonyms(self):
        """Test synonym-based generation."""
        provider = BuiltinVariantProvider(seed=42)
        question = "How do I upgrade my plan?"
        variants = provider.generate(question, n=20)

        # Should have some semantic variants
        assert any("upgrade" in v.lower() or "enhance" in v.lower() or "boost" in v.lower() for v in variants)
        assert any("plan" in v.lower() or "tier" in v.lower() or "level" in v.lower() for v in variants)

    def test_repr(self):
        """Test string representation."""
        provider = BuiltinVariantProvider(seed=42)
        assert "BuiltinVariantProvider" in repr(provider)

