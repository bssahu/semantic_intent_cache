"""Builtin variant provider using templates and rewrites."""

import logging
import random

logger = logging.getLogger(__name__)

# Templates for question paraphrasing
QUESTION_TEMPLATES = [
    # Direct variations
    "{question}",
    "Can you tell me {question_lower}?",
    "I need to know {question_lower}",
    "Could you explain {question_lower}?",
    # Politeness variations
    "Please help me with {question_lower}",
    "I'd like to understand {question_lower}",
    "What should I do to {question_lower}?",
    # Action-oriented
    "How can I {question_lower}?",
    "What's the process for {question_lower}?",
    "Steps to {question_lower}?",
    # Simplifications
    "How do I {question_lower}?",
    "Tell me about {question_lower}",
    "I want to {question_lower}",
]

# Alternative question words
QUESTION_WORDS_REPLACEMENTS = {
    "how": ["what", "which", "where", "when"],
    "what": ["how", "which"],
    "can": ["could", "would", "should"],
    "do": ["can", "should", "would"],
}

# Common synonyms for technical terms
SYNONYMS = {
    "upgrade": ["enhance", "improve", "boost", "increase"],
    "plan": ["tier", "level", "package", "subscription"],
    "change": ["switch", "modify", "alter", "update"],
    "cancel": ["stop", "end", "terminate", "discontinue"],
    "account": ["profile", "subscription", "settings"],
    "billing": ["payment", "charges", "invoice", "cost"],
}


class BuiltinVariantProvider:
    """Builtin variant provider using templates and rewrites."""

    def __init__(self, seed: int | None = 42):
        """
        Initialize the provider.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate(self, question: str, n: int) -> list[str]:
        """
        Generate semantic variants of a question.

        Args:
            question: The original question.
            n: Number of variants to generate.

        Returns:
            List of unique variant questions.
        """
        variants = set()
        variants.add(question)  # Always include original

        # Generate template-based variants
        question_lower = question.lower()
        for template in QUESTION_TEMPLATES:
            if len(variants) >= n:
                break
            try:
                variant = template.format(question=question, question_lower=question_lower)
                variants.add(variant.strip())
            except (KeyError, ValueError):
                continue

        # Generate replacement-based variants
        if len(variants) < n:
            variants.update(self._generate_replacements(question, n - len(variants)))

        # Generate synonym-based variants
        if len(variants) < n:
            variants.update(self._generate_synonym_variants(question, n - len(variants)))

        # Trim to requested number if exceeded
        result = list(variants)[:n]

        # Log if we couldn't generate enough
        if len(result) < n:
            logger.warning(
                f"Generated {len(result)} variants instead of requested {n} for: {question}"
            )

        return result

    def _generate_replacements(self, question: str, max_variants: int) -> set[str]:
        """Generate variants by replacing question words."""
        variants = set()
        words = question.lower().split()

        for i, word in enumerate(words):
            if len(variants) >= max_variants:
                break

            if word in QUESTION_WORDS_REPLACEMENTS:
                for replacement in QUESTION_WORDS_REPLACEMENTS[word]:
                    new_words = words.copy()
                    new_words[i] = replacement
                    variant = " ".join(new_words).capitalize()
                    if variant.strip():
                        variants.add(variant)

        return variants

    def _generate_synonym_variants(self, question: str, max_variants: int) -> set[str]:
        """Generate variants by replacing synonyms."""
        variants = set()
        words = question.lower().split()

        for i, word in enumerate(words):
            if len(variants) >= max_variants:
                break

            if word in SYNONYMS:
                for synonym in SYNONYMS[word]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    variant = " ".join(new_words).capitalize()
                    if variant.strip():
                        variants.add(variant)

        return variants

    def __repr__(self) -> str:
        """String representation."""
        return "BuiltinVariantProvider()"

