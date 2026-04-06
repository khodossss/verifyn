"""E2E streaming tests — verify SSE event flow with real LLM.

Checks that analyze_news_stream produces the correct sequence of events
and that all event types are present.

Run:
    pytest agent/tests/e2e/test_streaming.py --run-e2e -v

Requires: OPENAI_API_KEY
Cost: ~$0.05-0.10 per test
"""

from __future__ import annotations

import os
import tempfile

import pytest
from dotenv import load_dotenv

load_dotenv()

_E2E_DB_FILE = os.path.join(tempfile.gettempdir(), "verifyn_e2e_streaming.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_E2E_DB_FILE}"

from agent.db import Base, _get_engine

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module", autouse=True)
def e2e_db():
    import agent.db as db_mod

    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None
    os.environ["DATABASE_URL"] = f"sqlite:///{_E2E_DB_FILE}"
    db_mod.DATABASE_URL = f"sqlite:///{_E2E_DB_FILE}"

    if os.path.exists(_E2E_DB_FILE):
        os.remove(_E2E_DB_FILE)
    engine = _get_engine()
    Base.metadata.create_all(engine)
    yield
    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None
    if os.path.exists(_E2E_DB_FILE):
        try:
            os.remove(_E2E_DB_FILE)
        except OSError:
            pass


class TestStreamingEvents:
    """Verify the SSE event stream from analyze_news_stream."""

    def test_stream_produces_all_event_types(self):
        """A real claim should produce thinking → tool_call → tool_result → extracting → result."""
        from agent import analyze_news_stream

        events = list(
            analyze_news_stream(
                "The Eiffel Tower is located in Berlin, Germany.",
                reasoning_effort="low",
            )
        )

        event_types = [e["type"] for e in events]

        # Must have thinking (agent reasoning)
        assert "thinking" in event_types, f"No thinking event. Got: {event_types}"

        # Must have at least one tool call
        assert "tool_call" in event_types, f"No tool_call event. Got: {event_types}"

        # Must have extracting phase
        assert "extracting" in event_types, f"No extracting event. Got: {event_types}"

        # Must end with result (not error)
        assert "result" in event_types, f"No result event. Got: {event_types}"
        assert "error" not in event_types, f"Got error event: {[e for e in events if e['type'] == 'error']}"

    def test_stream_result_has_verdict(self):
        """The final result event must contain a valid FactCheckResult."""
        from agent import analyze_news_stream

        events = list(
            analyze_news_stream(
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                reasoning_effort="low",
            )
        )

        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1

        data = result_events[0]["data"]
        assert "verdict" in data
        assert "confidence" in data
        assert "summary" in data
        assert data["verdict"] in (
            "REAL",
            "FAKE",
            "MISLEADING",
            "PARTIALLY_FAKE",
            "UNVERIFIABLE",
            "SATIRE",
            "NO_CLAIMS",
        )
        assert 0.0 <= data["confidence"] <= 1.0

    def test_stream_tool_calls_have_labels(self):
        """Every tool_call event should have a human-readable label."""
        from agent import analyze_news_stream

        events = list(
            analyze_news_stream(
                "Breaking: Scientists discover that cats can fly using their whiskers.",
                reasoning_effort="low",
            )
        )

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1

        for tc in tool_calls:
            assert "tool" in tc, f"tool_call missing 'tool' field: {tc}"
            assert "label" in tc, f"tool_call missing 'label' field: {tc}"
            assert "query" in tc, f"tool_call missing 'query' field: {tc}"
            assert tc["label"], f"tool_call label is empty: {tc}"

    def test_stream_extracting_before_result(self):
        """The extracting event must come before the result event."""
        from agent import analyze_news_stream

        events = list(
            analyze_news_stream(
                "The Great Wall of China is visible from space with the naked eye.",
                reasoning_effort="low",
            )
        )

        types = [e["type"] for e in events]
        if "extracting" in types and "result" in types:
            assert types.index("extracting") < types.index("result")
