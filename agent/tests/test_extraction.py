"""Tests for the _extract_result function (Phase 2 parsing)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import _extract_result
from agent.models import FactCheckResult, Verdict


def _valid_payload() -> dict:
    return {
        "verdict": "REAL",
        "confidence": 0.9,
        "confidence_level": "HIGH",
        "manipulation_type": "NONE",
        "main_claims": ["Test claim"],
        "reasoning": "Full reasoning text.",
        "summary": "Plain language summary.",
    }


def _narrative_with_json(payload: dict | None = None) -> str:
    data = payload or _valid_payload()
    return f"Analysis complete.\n\n```json\n{json.dumps(data)}\n```"


class TestDirectParse:
    def test_valid_json_block_parsed_without_llm(self):
        narrative = _narrative_with_json()
        with patch("agent.agent._build_llm") as mock_llm:
            result = _extract_result(narrative)
        mock_llm.assert_not_called()
        assert isinstance(result, FactCheckResult)
        assert result.verdict == Verdict.REAL
        assert result.confidence == 0.9

    def test_all_verdict_types_parsed(self):
        for verdict in ("REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"):
            payload = {**_valid_payload(), "verdict": verdict}
            result = _extract_result(_narrative_with_json(payload))
            assert result.verdict.value == verdict

    def test_json_block_with_evidence(self):
        payload = {
            **_valid_payload(),
            "evidence_for": [
                {"source": "BBC", "url": None, "summary": "Confirmed", "supports_claim": True, "credibility": "high"}
            ],
            "sources_checked": ["https://bbc.com"],
        }
        result = _extract_result(_narrative_with_json(payload))
        assert len(result.evidence_for) == 1
        assert result.sources_checked == ["https://bbc.com"]

    def test_json_with_surrounding_text(self):
        narrative = (
            "Step 1: Extracted claim.\nStep 2: Searched web.\n"
            "```json\n" + json.dumps(_valid_payload()) + "\n```\n"
            "End of analysis."
        )
        result = _extract_result(narrative)
        assert result.verdict == Verdict.REAL


class TestRepairFallback:
    def _mock_llm_response(self, content: str) -> MagicMock:
        llm = MagicMock()
        response = MagicMock()
        response.content = content
        llm.invoke.return_value = response
        return llm

    def test_invalid_json_block_triggers_repair(self):
        narrative = "Analysis.\n```json\n{invalid json!!}\n```"
        valid_json = json.dumps(_valid_payload())
        with patch("agent.agent._build_llm", return_value=self._mock_llm_response(valid_json)):
            result = _extract_result(narrative)
        assert result.verdict == Verdict.REAL

    def test_no_json_block_triggers_repair(self):
        narrative = "The claim appears to be real based on multiple sources."
        valid_json = json.dumps(_valid_payload())
        with patch("agent.agent._build_llm", return_value=self._mock_llm_response(valid_json)):
            result = _extract_result(narrative)
        assert isinstance(result, FactCheckResult)

    def test_repair_failure_raises_runtime_error(self):
        narrative = "No JSON block here."
        with patch("agent.agent._build_llm", return_value=self._mock_llm_response("not json at all")):
            with pytest.raises(RuntimeError, match="Extraction failed"):
                _extract_result(narrative)

    def test_repair_call_receives_broken_json(self):
        broken = '{"verdict": "FAKE", "missing_fields": true}'
        narrative = f"```json\n{broken}\n```"
        valid_json = json.dumps(_valid_payload())
        llm = self._mock_llm_response(valid_json)
        with patch("agent.agent._build_llm", return_value=llm):
            _extract_result(narrative)
        # Repair LLM was called with the broken JSON in the message
        call_args = llm.invoke.call_args[0][0]
        assert any(broken in str(m.content) for m in call_args)
