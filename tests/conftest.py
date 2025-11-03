"""Pytest configuration and fixtures."""

import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for unit tests."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.ft.return_value = mock_client
    mock_client.info.return_value = {}
    return mock_client


@pytest.fixture
def sample_intents():
    """Sample intents for testing."""
    return {
        "UPGRADE_PLAN": "How do I upgrade my plan?",
        "CANCEL_ACCOUNT": "How do I cancel my account?",
        "CHANGE_BILLING": "How can I change my billing information?",
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "upgrade_similar": "I want to move to a higher tier",
        "cancel_similar": "How to terminate my subscription?",
        "billing_similar": "Update my payment details",
        "no_match": "What is the weather today?",
    }

