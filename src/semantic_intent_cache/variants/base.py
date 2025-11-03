"""Base protocol for variant providers."""

from typing import Protocol


class VariantProvider(Protocol):
    """Protocol for variant providers."""

    def generate(self, question: str, n: int) -> list[str]:
        """
        Generate semantic variants of a question.

        Args:
            question: The original question.
            n: Number of variants to generate (including original if included).

        Returns:
            List of variant questions.
        """
        ...

