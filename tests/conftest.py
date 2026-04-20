"""Pytest configuration for SGOD tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires GPU + checkpoints")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless -m integration is specified."""
    if not config.getoption("-m", default=None):
        skip_integration = pytest.mark.skip(reason="needs -m integration flag")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
