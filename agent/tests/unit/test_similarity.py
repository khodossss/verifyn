"""Tests for the similarity search tool and agent mode injection."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["DATABASE_URL"] = "sqlite://"

from agent.db import Base, _get_engine
from agent.tools.similarity import _format_previous_result, search_similar_queries


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset the DB before each test."""
    import agent.db as db_mod

    db_mod._engine = None
    db_mod._SessionFactory = None
    engine = _get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    db_mod._engine = None
    db_mod._SessionFactory = None


# ---------------------------------------------------------------------------
# _format_previous_result
# ---------------------------------------------------------------------------


class TestFormatPreviousResult:
    def test_includes_verdict_and_confidence(self):
        match = {
            "query": "Is earth flat?",
            "similarity": 0.95,
            "mode": "precise",
            "created_at": "2026-01-01T00:00:00",
            "result": {"verdict": "FAKE", "confidence": 0.92, "manipulation_type": "FABRICATED"},
        }
        output = _format_previous_result(match)
        assert "FAKE" in output
        assert "0.92" in output
        assert "0.95" in output
        assert "precise" in output

    def test_includes_evidence_summary(self):
        match = {
            "query": "test",
            "similarity": 0.9,
            "mode": "fast",
            "created_at": "2026-01-01T00:00:00",
            "result": {
                "verdict": "REAL",
                "confidence": 0.85,
                "evidence_for": [
                    {
                        "source": "Reuters",
                        "url": "https://reuters.com/a",
                        "summary": "Confirmed by Reuters",
                        "supports_claim": True,
                    }
                ],
                "evidence_against": [],
            },
        }
        output = _format_previous_result(match)
        assert "Reuters" in output
        assert "reuters.com" in output
        assert "FOR" in output

    def test_includes_summary_field(self):
        match = {
            "query": "test",
            "similarity": 0.9,
            "mode": "fast",
            "created_at": None,
            "result": {
                "verdict": "MISLEADING",
                "confidence": 0.6,
                "summary": "The claim is misleading due to context.",
            },
        }
        output = _format_previous_result(match)
        assert "misleading due to context" in output

    def test_includes_main_claims(self):
        match = {
            "query": "test",
            "similarity": 0.9,
            "mode": "precise",
            "created_at": "2026-01-01T00:00:00",
            "result": {"verdict": "FAKE", "confidence": 0.9, "main_claims": ["Claim A", "Claim B"]},
        }
        output = _format_previous_result(match)
        assert "Claim A" in output
        assert "Claim B" in output

    def test_handles_missing_optional_fields(self):
        """Minimal result dict — should not crash."""
        match = {
            "query": "test",
            "similarity": 0.88,
            "mode": "fast",
            "created_at": None,
            "result": {"verdict": "UNVERIFIABLE", "confidence": 0.3},
        }
        output = _format_previous_result(match)
        assert "UNVERIFIABLE" in output

    def test_evidence_against_shown_as_against(self):
        match = {
            "query": "test",
            "similarity": 0.9,
            "mode": "precise",
            "created_at": "2026-01-01T00:00:00",
            "result": {
                "verdict": "FAKE",
                "confidence": 0.95,
                "evidence_for": [],
                "evidence_against": [
                    {"source": "Snopes", "url": "https://snopes.com/x", "summary": "Debunked", "supports_claim": False}
                ],
            },
        }
        output = _format_previous_result(match)
        assert "AGAINST" in output
        assert "Snopes" in output


# ---------------------------------------------------------------------------
# search_similar_queries tool
# ---------------------------------------------------------------------------


class TestSearchSimilarQueriesTool:
    def _make_result(self, verdict="REAL"):
        result = MagicMock()
        result.verdict.value = verdict
        result.model_dump.return_value = {
            "verdict": verdict,
            "confidence": 0.9,
            "summary": f"This is {verdict}.",
        }
        return result

    @patch("agent.db.compute_embedding")
    @patch("agent.db.find_similar_queries")
    def test_returns_no_matches_message(self, mock_find, mock_embed):
        mock_embed.return_value = [0.1] * 8
        mock_find.return_value = []

        result = search_similar_queries.invoke({"query": "test claim"})
        assert "No similar" in result

    @patch("agent.db.compute_embedding")
    @patch("agent.db.find_similar_queries")
    def test_returns_formatted_matches(self, mock_find, mock_embed):
        mock_embed.return_value = [0.1] * 8
        mock_find.return_value = [
            {
                "id": 1,
                "query": "Is earth flat?",
                "mode": "precise",
                "result": {"verdict": "FAKE", "confidence": 0.95},
                "similarity": 0.93,
                "created_at": "2026-01-01T00:00:00",
            }
        ]

        result = search_similar_queries.invoke({"query": "is the earth flat"})
        assert "Found 1 similar" in result
        assert "FAKE" in result
        assert "0.93" in result

    @patch("agent.db.compute_embedding")
    def test_embedding_failure_returns_unavailable(self, mock_embed):
        mock_embed.side_effect = RuntimeError("API key missing")

        result = search_similar_queries.invoke({"query": "test"})
        assert "unavailable" in result.lower()

    @patch("agent.db.compute_embedding")
    @patch("agent.db.find_similar_queries")
    def test_mode_injected_from_attribute(self, mock_find, mock_embed):
        """The tool reads _current_mode and passes it to find_similar_queries."""
        mock_embed.return_value = [0.1] * 8
        mock_find.return_value = []

        search_similar_queries._current_mode = "precise"
        search_similar_queries.invoke({"query": "test"})

        mock_find.assert_called_once()
        _, kwargs = mock_find.call_args
        assert kwargs["mode"] == "precise"

        # Reset
        search_similar_queries._current_mode = "fast"

    @patch("agent.db.compute_embedding")
    @patch("agent.db.find_similar_queries")
    def test_default_mode_is_fast(self, mock_find, mock_embed):
        """When _current_mode is not set, default should be 'fast'."""
        mock_embed.return_value = [0.1] * 8
        mock_find.return_value = []

        # Remove attribute if it exists
        if hasattr(search_similar_queries, "_current_mode"):
            delattr(search_similar_queries, "_current_mode")

        search_similar_queries.invoke({"query": "test"})

        _, kwargs = mock_find.call_args
        assert kwargs["mode"] == "fast"


# ---------------------------------------------------------------------------
# Agent mode injection
# ---------------------------------------------------------------------------


class TestAgentModeInjection:
    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    @patch("agent.db.compute_embedding", return_value=[0.1] * 8)
    def test_fast_mode_sets_tool_mode(self, mock_embed, mock_build_llm, mock_create_agent):
        """analyze_news with reasoning_effort='low' should set tool mode to 'fast'."""
        from agent.tools.similarity import search_similar_queries as sim_tool

        mock_build_llm.return_value = MagicMock()

        import json

        valid_json = json.dumps(
            {
                "verdict": "FAKE",
                "confidence": 0.92,
                "confidence_level": "HIGH",
                "manipulation_type": "FABRICATED",
                "reasoning": "test",
                "summary": "test",
            }
        )
        agent = MagicMock()

        def fake_stream(inputs, stream_mode=None, config=None):
            from langchain_core.messages import AIMessage, HumanMessage

            yield {
                "messages": [
                    HumanMessage(content="test"),
                    AIMessage(content=f"```json\n{valid_json}\n```", id="msg1"),
                ]
            }

        agent.stream = fake_stream
        mock_create_agent.return_value = agent

        from agent.agent import analyze_news

        analyze_news("Test claim for mode injection", reasoning_effort="low")
        assert sim_tool._current_mode == "fast"

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    @patch("agent.db.compute_embedding", return_value=[0.1] * 8)
    def test_precise_mode_sets_tool_mode(self, mock_embed, mock_build_llm, mock_create_agent):
        """analyze_news with reasoning_effort='medium' should set tool mode to 'precise'."""
        from agent.tools.similarity import search_similar_queries as sim_tool

        mock_build_llm.return_value = MagicMock()

        import json

        valid_json = json.dumps(
            {
                "verdict": "REAL",
                "confidence": 0.85,
                "confidence_level": "HIGH",
                "manipulation_type": "NONE",
                "reasoning": "test",
                "summary": "test",
            }
        )
        agent = MagicMock()

        def fake_stream(inputs, stream_mode=None, config=None):
            from langchain_core.messages import AIMessage, HumanMessage

            yield {
                "messages": [
                    HumanMessage(content="test"),
                    AIMessage(content=f"```json\n{valid_json}\n```", id="msg1"),
                ]
            }

        agent.stream = fake_stream
        mock_create_agent.return_value = agent

        from agent.agent import analyze_news

        analyze_news("Test claim for precise mode", reasoning_effort="medium")
        assert sim_tool._current_mode == "precise"
