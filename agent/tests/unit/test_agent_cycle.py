"""Tests for the agent cycle — analyze_news, analyze_news_stream, helpers.

These tests mock the LLM and tools to test the orchestration logic
without making real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.agent import (
    _build_user_message,
    _extract_narrative,
    _sanitize_json,
    _try_parse,
    analyze_news,
    analyze_news_stream,
)
from agent.models import FactCheckResult, Verdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps(
    {
        "verdict": "FAKE",
        "confidence": 0.92,
        "confidence_level": "HIGH",
        "manipulation_type": "FABRICATED",
        "main_claims": ["Test claim"],
        "evidence_for": [],
        "evidence_against": [
            {
                "source": "Reuters",
                "url": "https://reuters.com",
                "summary": "Debunked",
                "supports_claim": False,
                "credibility": "HIGH",
            }
        ],
        "fact_checker_results": ["Snopes rated this False"],
        "sources_checked": ["https://reuters.com"],
        "reasoning": "Step 1: Extracted claim.\nStep 2: Found debunking.",
        "summary": "This claim is fabricated.",
    }
)


def _agent_final_message() -> str:
    """Simulates agent's final message with JSON block."""
    return f"After researching the claim, I found it to be fabricated.\n\n```json\n{VALID_JSON}\n```"


def _mock_react_agent(final_content: str):
    """Create a mock ReAct agent that yields a single chunk with the final message."""
    agent = MagicMock()

    def fake_stream(inputs, stream_mode=None, config=None):
        messages = [
            HumanMessage(content=inputs["messages"][0].content),
            AIMessage(content="Let me search for this claim.", id="msg1"),
            AIMessage(content=final_content, id="msg2"),
        ]
        yield {"messages": messages}

    agent.stream = fake_stream
    return agent


# ---------------------------------------------------------------------------
# _sanitize_json
# ---------------------------------------------------------------------------


class TestSanitizeJson:
    def test_pipe_in_manipulation_type(self):
        data = {"manipulation_type": "FABRICATED|CONTEXT_MANIPULATION", "confidence_level": "HIGH", "verdict": "FAKE"}
        result = _sanitize_json(data)
        assert result["manipulation_type"] == "FABRICATED"

    def test_confidence_level_lowercase(self):
        data = {"confidence_level": "high", "verdict": "REAL"}
        result = _sanitize_json(data)
        assert result["confidence_level"] == "HIGH"

    def test_confidence_level_mixed_case(self):
        data = {"confidence_level": " Medium ", "verdict": "MISLEADING"}
        result = _sanitize_json(data)
        assert result["confidence_level"] == "MEDIUM"

    def test_verdict_lowercase(self):
        data = {"verdict": "fake", "confidence_level": "HIGH"}
        result = _sanitize_json(data)
        assert result["verdict"] == "FAKE"

    def test_no_mutation_when_clean(self):
        data = {"verdict": "REAL", "confidence_level": "HIGH", "manipulation_type": "NONE"}
        result = _sanitize_json(data)
        assert result == data


# ---------------------------------------------------------------------------
# _try_parse
# ---------------------------------------------------------------------------


class TestTryParse:
    def test_valid_json_block(self):
        text = f"Some text\n```json\n{VALID_JSON}\n```\nMore text"
        result = _try_parse(text)
        assert result is not None
        assert result.verdict == Verdict.FAKE

    def test_json_without_code_fence(self):
        text = f"Some text\n{VALID_JSON}\nMore text"
        result = _try_parse(text)
        assert result is not None

    def test_completely_invalid(self):
        result = _try_parse("no json here at all")
        assert result is None

    def test_json_missing_required_field(self):
        bad = json.dumps({"verdict": "REAL"})  # missing confidence, reasoning, summary
        _try_parse(f"```json\n{bad}\n```")
        # json_repair + pydantic may or may not parse this — just shouldn't crash

    def test_truncated_json(self):
        truncated = '{"verdict": "FAKE", "confidence": 0.9, "confidence_level": "HIGH"'
        _try_parse(f"```json\n{truncated}\n```")
        # json_repair should handle truncated JSON

    def test_empty_string(self):
        result = _try_parse("")
        assert result is None


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_contains_news_text(self):
        msg = _build_user_message("Test news claim")
        assert "Test news claim" in msg

    def test_contains_date(self):
        msg = _build_user_message("anything")
        assert "TODAY'S DATE:" in msg

    def test_contains_search_limit(self):
        msg = _build_user_message("anything")
        assert "5 unique searches" in msg

    def test_contains_stop_rule(self):
        msg = _build_user_message("anything")
        assert "STOP RULE" in msg


# ---------------------------------------------------------------------------
# _extract_narrative
# ---------------------------------------------------------------------------


class TestExtractNarrative:
    def test_prefers_message_with_json_block(self):
        msgs = [
            AIMessage(content="Thinking about the claim...", id="1"),
            AIMessage(content="No JSON here", id="2"),
            AIMessage(content=f"Conclusion:\n```json\n{VALID_JSON}\n```", id="3"),
        ]
        narrative = _extract_narrative(msgs)
        assert "```json" in narrative

    def test_prefers_last_json_message_when_multiple(self):
        msgs = [
            AIMessage(content='```json\n{"verdict":"REAL"}\n```', id="1"),
            HumanMessage(content="test"),
            AIMessage(content=f"```json\n{VALID_JSON}\n```", id="2"),
        ]
        narrative = _extract_narrative(msgs)
        assert "FAKE" in narrative  # from VALID_JSON

    def test_falls_back_to_last_ai_message(self):
        msgs = [
            AIMessage(content="First analysis", id="1"),
            AIMessage(content="Final analysis without JSON", id="2"),
        ]
        narrative = _extract_narrative(msgs)
        assert narrative == "Final analysis without JSON"

    def test_skips_non_ai_messages(self):
        msgs = [
            HumanMessage(content="check this"),
            ToolMessage(content="search results", name="web_search", tool_call_id="tc1"),
            AIMessage(content="Based on the search results...", id="1"),
        ]
        narrative = _extract_narrative(msgs)
        assert "Based on the search results" in narrative

    def test_concatenates_when_no_single_good_message(self):
        msgs = [
            AIMessage(content="", id="1"),  # empty
            AIMessage(content="Part one.", id="2"),
            AIMessage(content="Part two.", id="3"),
        ]
        # Empty content first → falls through to concatenation
        narrative = _extract_narrative(msgs)
        assert "Part" in narrative

    def test_empty_messages_list(self):
        narrative = _extract_narrative([])
        assert narrative == ""


# ---------------------------------------------------------------------------
# analyze_news (mocked)
# ---------------------------------------------------------------------------


class TestAnalyzeNews:
    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_returns_fact_check_result(self, mock_build_llm, mock_create_agent):
        mock_create_agent.return_value = _mock_react_agent(_agent_final_message())
        mock_build_llm.return_value = MagicMock()

        result = analyze_news("NASA confirms bleach cures COVID")

        assert isinstance(result, FactCheckResult)
        assert result.verdict == Verdict.FAKE
        assert result.confidence == 0.92

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_passes_reasoning_effort(self, mock_build_llm, mock_create_agent):
        mock_create_agent.return_value = _mock_react_agent(_agent_final_message())
        mock_build_llm.return_value = MagicMock()

        analyze_news("test claim", reasoning_effort="low")

        mock_build_llm.assert_called_with(reasoning_effort="low")

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_agent_returns_no_json_triggers_repair(self, mock_build_llm, mock_create_agent):
        """When the agent doesn't produce JSON, extraction falls back to repair LLM."""
        mock_create_agent.return_value = _mock_react_agent("The claim is clearly fake based on my research.")
        # First call = agent LLM, subsequent calls = repair LLM
        repair_llm = MagicMock()
        repair_response = MagicMock()
        repair_response.content = VALID_JSON
        repair_llm.invoke.return_value = repair_response
        mock_build_llm.return_value = repair_llm

        result = analyze_news("test claim")

        assert isinstance(result, FactCheckResult)

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_agent_empty_response_triggers_repair(self, mock_build_llm, mock_create_agent):
        """When the agent produces empty messages, repair is attempted."""
        agent = MagicMock()

        def fake_stream(inputs, stream_mode=None, config=None):
            yield {"messages": [HumanMessage(content="test"), AIMessage(content="", id="1")]}

        agent.stream = fake_stream
        mock_create_agent.return_value = agent

        repair_llm = MagicMock()
        repair_response = MagicMock()
        repair_response.content = VALID_JSON
        repair_llm.invoke.return_value = repair_response
        mock_build_llm.return_value = repair_llm

        result = analyze_news("test")
        assert isinstance(result, FactCheckResult)


# ---------------------------------------------------------------------------
# analyze_news_stream (mocked)
# ---------------------------------------------------------------------------


class TestAnalyzeNewsStream:
    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_yields_result_event(self, mock_build_llm, mock_create_agent):
        mock_create_agent.return_value = _mock_react_agent(_agent_final_message())
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test claim"))

        event_types = [e["type"] for e in events]
        assert "result" in event_types
        result_event = next(e for e in events if e["type"] == "result")
        assert result_event["data"]["verdict"] == "FAKE"

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_yields_extracting_before_result(self, mock_build_llm, mock_create_agent):
        mock_create_agent.return_value = _mock_react_agent(_agent_final_message())
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test"))

        types = [e["type"] for e in events]
        assert "extracting" in types
        assert types.index("extracting") < types.index("result")

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_yields_thinking_events(self, mock_build_llm, mock_create_agent):
        mock_create_agent.return_value = _mock_react_agent(_agent_final_message())
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test"))

        thinking = [e for e in events if e["type"] == "thinking"]
        assert len(thinking) >= 1

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_tool_call_events(self, mock_build_llm, mock_create_agent):
        """Tool calls in AIMessage should yield tool_call events."""
        agent = MagicMock()

        ai_with_tool = AIMessage(content="Let me search.", id="msg1")
        ai_with_tool.tool_calls = [{"name": "web_search", "args": {"query": "test claim"}, "id": "tc1"}]

        def fake_stream(inputs, stream_mode=None, config=None):
            yield {"messages": [HumanMessage(content="test"), ai_with_tool]}
            yield {
                "messages": [
                    HumanMessage(content="test"),
                    ai_with_tool,
                    ToolMessage(content="results", name="web_search", tool_call_id="tc1"),
                ]
            }
            yield {
                "messages": [
                    HumanMessage(content="test"),
                    ai_with_tool,
                    ToolMessage(content="results", name="web_search", tool_call_id="tc1"),
                    AIMessage(content=_agent_final_message(), id="msg2"),
                ]
            }

        agent.stream = fake_stream
        mock_create_agent.return_value = agent
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test"))

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1
        assert tool_calls[0]["tool"] == "web_search"
        assert tool_calls[0]["query"] == "test claim"

    @patch("agent.agent._build_llm")
    def test_llm_init_error_yields_error_event(self, mock_build_llm):
        mock_build_llm.side_effect = ValueError("Invalid API key")

        events = list(analyze_news_stream("test"))

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "Invalid API key" in events[0]["message"]

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_agent_exception_yields_error_event(self, mock_build_llm, mock_create_agent):
        agent = MagicMock()
        agent.stream.side_effect = RuntimeError("LLM timeout")
        mock_create_agent.return_value = agent
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test"))

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "LLM timeout" in error_events[0]["message"]

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_extraction_failure_yields_error_event(self, mock_build_llm, mock_create_agent):
        """When agent produces no JSON and repair also fails → error event."""
        mock_create_agent.return_value = _mock_react_agent("no json at all, just text")

        # Repair LLM also returns garbage
        repair_llm = MagicMock()
        repair_response = MagicMock()
        repair_response.content = "still not json"
        repair_llm.invoke.return_value = repair_response
        mock_build_llm.return_value = repair_llm

        events = list(analyze_news_stream("test"))

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "Failed to structure result" in error_events[0]["message"]

    @patch("agent.agent.create_react_agent")
    @patch("agent.agent._build_llm")
    def test_stream_deduplicates_messages(self, mock_build_llm, mock_create_agent):
        """Messages with same id should not produce duplicate events."""
        agent = MagicMock()
        msg = AIMessage(content="thinking...", id="same_id")

        def fake_stream(inputs, stream_mode=None, config=None):
            yield {"messages": [msg]}
            yield {"messages": [msg]}  # duplicate
            yield {"messages": [msg, AIMessage(content=_agent_final_message(), id="final")]}

        agent.stream = fake_stream
        mock_create_agent.return_value = agent
        mock_build_llm.return_value = MagicMock()

        events = list(analyze_news_stream("test"))
        thinking = [e for e in events if e["type"] == "thinking" and e["text"] == "thinking..."]
        assert len(thinking) == 1  # not duplicated
