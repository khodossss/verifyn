"""Integration test — agent calls search_similar_queries and gets a real DB result.

Unlike unit tests that mock everything, this test:
- Uses a real in-memory SQLite DB seeded with a previous fact-check
- Uses real tools (search_similar_queries) with real DB queries
- Mocks only: LLM (to control agent behavior) and OpenAI embeddings API
- Verifies the full chain: agent → tool call → DB lookup → result back to agent
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Use a file-based temp DB so it's accessible across threads
# (in-memory SQLite is per-connection, breaks when LangGraph runs tools in threads)
_TEST_DB_FILE = os.path.join(tempfile.gettempdir(), "verifyn_test_integration.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_FILE}"

from verifyn.agent.db import Base, _get_engine, save_query
from verifyn.agent.models import FactCheckResult, Verdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536  # Simulated text-embedding-3-small output

PREVIOUS_RESULT = {
    "verdict": "FAKE",
    "confidence": 0.95,
    "confidence_level": "HIGH",
    "manipulation_type": "FABRICATED",
    "main_claims": ["5G towers cause COVID-19"],
    "evidence_for": [],
    "evidence_against": [
        {
            "source": "Reuters",
            "url": "https://reuters.com/fact-check/5g",
            "summary": "No scientific evidence links 5G to COVID-19",
            "supports_claim": False,
            "credibility": "HIGH",
        }
    ],
    "fact_checker_results": ["Snopes: FALSE"],
    "sources_checked": ["https://reuters.com/fact-check/5g", "https://snopes.com/5g-covid"],
    "reasoning": "Step 1: Claim is that 5G causes COVID. Step 2: Debunked by multiple sources.",
    "summary": "This claim is fabricated. No credible evidence supports it.",
}

FINAL_AGENT_JSON = json.dumps(
    {
        "verdict": "FAKE",
        "confidence": 0.93,
        "confidence_level": "HIGH",
        "manipulation_type": "FABRICATED",
        "main_claims": ["5G towers spread coronavirus"],
        "evidence_for": [],
        "evidence_against": [
            {
                "source": "Reuters",
                "url": "https://reuters.com/fact-check/5g",
                "summary": "Debunked",
                "supports_claim": False,
                "credibility": "HIGH",
            }
        ],
        "fact_checker_results": [],
        "sources_checked": ["https://reuters.com/fact-check/5g"],
        "reasoning": "Found in history. Previous check confirmed FAKE. Verified with search.",
        "summary": "This claim was previously debunked and remains false.",
    }
)


class ScriptedLLM(BaseChatModel):
    """A fake LLM that follows a script: first call → tool call, then → final JSON.

    Must be a proper BaseChatModel subclass so LangGraph accepts it as a Runnable.
    """

    final_json: str = ""
    call_count: int = 0
    tool_result_text: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedLLM":
        # LangGraph calls bind_tools; we just return self since tool calls are scripted
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.call_count += 1

        # Check if we received a tool result
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_similar_queries":
                self.tool_result_text = msg.content

        if self.call_count == 1:
            # First call: request tool call
            ai = AIMessage(content="Let me check history.", id=f"msg_{self.call_count}")
            ai.tool_calls = [
                {
                    "name": "search_similar_queries",
                    "args": {"query": "5G COVID"},
                    "id": "tc_sim_1",
                    "type": "tool_call",
                }
            ]
            return ChatResult(generations=[ChatGeneration(message=ai)])
        else:
            # Subsequent calls: produce final verdict
            ai = AIMessage(
                content=f"Based on history:\n\n```json\n{self.final_json}\n```",
                id=f"msg_{self.call_count}",
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset DB for each test using a file-based temp SQLite.

    Uses a file-based DB (not in-memory) because LangGraph's create_react_agent
    may execute tools in a different thread, and SQLite in-memory DBs are
    per-connection — a second thread would get an empty database.
    """
    import verifyn.agent.db as db_mod

    # Dispose previous engine if exists (releases file lock on Windows)
    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None

    # Force file-based URL (other test files may have set sqlite:// in-memory)
    os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_FILE}"
    db_mod.DATABASE_URL = f"sqlite:///{_TEST_DB_FILE}"

    # Remove old file if exists
    if os.path.exists(_TEST_DB_FILE):
        os.remove(_TEST_DB_FILE)
    engine = _get_engine()
    Base.metadata.create_all(engine)
    yield
    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None
    if os.path.exists(_TEST_DB_FILE):
        try:
            os.remove(_TEST_DB_FILE)
        except OSError:
            pass  # Best-effort cleanup on Windows


@pytest.fixture(autouse=True)
def _disable_claim_prefilter():
    """Block the classifier so the agent integration test does not load DeBERTa."""
    with patch("verifyn.agent.agent._claim_prefilter", return_value=None):
        yield


def _seed_previous_query():
    """Insert a previous fact-check into the DB with embedding."""
    result_mock = MagicMock()
    result_mock.verdict.value = "FAKE"
    result_mock.model_dump.return_value = PREVIOUS_RESULT
    save_query(
        "5G towers cause COVID-19 pandemic",
        "precise",
        result_mock,
        reputation_updated=1,
        embedding=FAKE_EMBEDDING,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSimilaritySearchIntegration:
    """Test that the search_similar_queries tool works end-to-end with the DB."""

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_tool_finds_previous_result_from_db(self, mock_embed):
        """Call the tool directly with a similar query and verify it hits the DB."""
        _seed_previous_query()

        from verifyn.agent.tools.similarity import search_similar_queries

        search_similar_queries._current_mode = "fast"
        result = search_similar_queries.invoke({"query": "Do 5G towers spread COVID?"})

        assert "Found 1 similar" in result
        assert "FAKE" in result
        assert "0.95" in result  # confidence from previous result
        assert "Reuters" in result
        assert "5G" in result

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_tool_returns_nothing_when_db_empty(self, mock_embed):
        """No previous queries in DB → tool returns 'no similar' message."""
        from verifyn.agent.tools.similarity import search_similar_queries

        result = search_similar_queries.invoke({"query": "Some random claim"})
        assert "No similar" in result

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_tool_respects_mode_precise(self, mock_embed):
        """In precise mode, tool should not return fast-mode results."""
        # Seed a fast-mode result
        result_mock = MagicMock()
        result_mock.verdict.value = "REAL"
        result_mock.model_dump.return_value = {"verdict": "REAL", "confidence": 0.8}
        save_query("test claim", "fast", result_mock, embedding=FAKE_EMBEDDING)

        from verifyn.agent.tools.similarity import search_similar_queries

        search_similar_queries._current_mode = "precise"
        result = search_similar_queries.invoke({"query": "test claim"})

        assert "No similar" in result

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_tool_in_precise_mode_finds_precise_results(self, mock_embed):
        """In precise mode, tool should find precise-mode results."""
        _seed_previous_query()  # seeds as "precise"

        from verifyn.agent.tools.similarity import search_similar_queries

        search_similar_queries._current_mode = "precise"
        result = search_similar_queries.invoke({"query": "5G COVID claim"})

        assert "Found 1 similar" in result

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_agent_calls_similarity_tool_and_receives_history(self, mock_embed):
        """Full agent integration: LLM calls search_similar_queries,
        tool executes against DB, result flows back to agent."""
        _seed_previous_query()

        # Use the project's wrapped import (warning suppressed in agent.agent)
        from verifyn.agent.agent import create_react_agent
        from verifyn.agent.prompts import SYSTEM_PROMPT
        from verifyn.agent.tools import ALL_TOOLS
        from verifyn.agent.tools.similarity import search_similar_queries

        search_similar_queries._current_mode = "fast"

        fake_llm = ScriptedLLM(final_json=FINAL_AGENT_JSON)
        agent = create_react_agent(model=fake_llm, tools=ALL_TOOLS, prompt=SYSTEM_PROMPT)

        # Run the agent
        research_messages = []
        for chunk in agent.stream(
            {"messages": [HumanMessage(content="5G towers are spreading COVID-19")]},
            stream_mode="values",
            config={"recursion_limit": 10},
        ):
            msgs = chunk.get("messages", [])
            if msgs:
                research_messages = msgs

        # Verify tool was called and returned DB data
        assert fake_llm.tool_result_text is not None, "Agent never received tool result"
        assert "Found 1 similar" in fake_llm.tool_result_text
        assert "FAKE" in fake_llm.tool_result_text
        assert "Reuters" in fake_llm.tool_result_text
        assert "5G" in fake_llm.tool_result_text

        # Verify there was a ToolMessage in the conversation
        tool_msgs = [m for m in research_messages if isinstance(m, ToolMessage)]
        assert any(m.name == "search_similar_queries" for m in tool_msgs)

        # Verify the agent produced a final answer with JSON
        final_ai = [m for m in research_messages if isinstance(m, AIMessage) and "```json" in (m.content or "")]
        assert len(final_ai) >= 1

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_full_analyze_news_with_seeded_history(self, mock_embed):
        """End-to-end: analyze_news with a ScriptedLLM that uses similarity search.
        Verifies embedding is computed, mode is set, and result is saved back."""
        _seed_previous_query()

        from verifyn.agent.agent import analyze_news

        with patch("verifyn.agent.agent._build_llm", return_value=ScriptedLLM(final_json=FINAL_AGENT_JSON)):
            with patch("verifyn.agent.agent.create_react_agent") as mock_create:
                # Import the real factory directly from langgraph (bypass our patched module ref).
                # The deprecation warning from this import is filtered globally in pytest.ini.
                import warnings as _w

                with _w.catch_warnings():
                    _w.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
                    from langgraph.prebuilt import create_react_agent as real_create

                def patched_create(model, tools, prompt):
                    return real_create(model=ScriptedLLM(final_json=FINAL_AGENT_JSON), tools=tools, prompt=prompt)

                mock_create.side_effect = patched_create

                result = analyze_news("5G towers cause COVID-19", reasoning_effort="low")

        assert isinstance(result, FactCheckResult)
        assert result.verdict == Verdict.FAKE

        # Verify a new query was saved to history (now 2 total)
        from verifyn.agent.db import get_query_history

        history = get_query_history(limit=10)
        assert len(history) == 2  # original seed + new query

    @patch("verifyn.agent.db.compute_embedding", return_value=FAKE_EMBEDDING)
    def test_stream_variant_calls_similarity_tool(self, mock_embed):
        """analyze_news_stream also triggers similarity search and yields tool_call events."""
        _seed_previous_query()

        from verifyn.agent.agent import analyze_news_stream

        with patch("verifyn.agent.agent._build_llm", return_value=ScriptedLLM(final_json=FINAL_AGENT_JSON)):
            with patch("verifyn.agent.agent.create_react_agent") as mock_create:
                # Import the real factory directly from langgraph (bypass our patched module ref).
                # The deprecation warning from this import is filtered globally in pytest.ini.
                import warnings as _w

                with _w.catch_warnings():
                    _w.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
                    from langgraph.prebuilt import create_react_agent as real_create

                def patched_create(model, tools, prompt):
                    return real_create(model=ScriptedLLM(final_json=FINAL_AGENT_JSON), tools=tools, prompt=prompt)

                mock_create.side_effect = patched_create

                events = list(analyze_news_stream("5G towers cause COVID", reasoning_effort="low"))

        event_types = [e["type"] for e in events]

        # Should have tool_call event for search_similar_queries
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        sim_calls = [e for e in tool_call_events if e["tool"] == "search_similar_queries"]
        assert len(sim_calls) >= 1, f"No search_similar_queries tool call found. Events: {event_types}"
        assert sim_calls[0]["label"] == "Searching previous fact-checks"

        # Should have tool_result event
        assert "tool_result" in event_types

        # Should end with result
        assert "result" in event_types
        result_event = next(e for e in events if e["type"] == "result")
        assert result_event["data"]["verdict"] == "FAKE"
