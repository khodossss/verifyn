"""E2E test configuration.

These tests make real API calls (OpenAI LLM + embeddings, web search).
They are expensive, slow, and require OPENAI_API_KEY.

Run manually:
    pytest agent/tests/e2e/ --run-e2e -v

Never runs in CI by default.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", default=False, help="Run e2e tests (real API calls)")


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end test requiring real API keys")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="Need --run-e2e flag to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture(scope="session", autouse=True)
def check_api_key(request):
    """Skip all e2e tests if OPENAI_API_KEY is not set."""
    if not request.config.getoption("--run-e2e"):
        return
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — cannot run e2e tests")
