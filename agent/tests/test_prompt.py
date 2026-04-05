"""Tests for the system prompt content and structure."""

from agent.prompts import SYSTEM_PROMPT


class TestSystemPromptContent:
    def test_prompt_nonempty(self):
        assert len(SYSTEM_PROMPT) > 500

    def test_all_eight_steps_present(self):
        for i in range(1, 9):
            assert f"Step {i}" in SYSTEM_PROMPT, f"Step {i} missing from system prompt"

    def test_all_tool_names_mentioned(self):
        tools = [
            "web_search",
            "search_fact_checkers",
            "check_if_old_news",
            "extract_article_content",
            "check_domain_reputation",
        ]
        for tool in tools:
            assert tool in SYSTEM_PROMPT, f"Tool '{tool}' not mentioned in prompt"

    def test_all_verdicts_defined(self):
        for verdict in ("REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"):
            assert verdict in SYSTEM_PROMPT, f"Verdict '{verdict}' not in prompt"

    def test_tool_limit_mentioned(self):
        assert "tool call" in SYSTEM_PROMPT.lower() or "tool calls" in SYSTEM_PROMPT.lower()

    def test_json_output_schema_present(self):
        assert "```json" in SYSTEM_PROMPT, "JSON output schema block missing from prompt"
        assert "verdict" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT
        assert "summary" in SYSTEM_PROMPT

    def test_confidence_calibration_guidance_present(self):
        assert "confidence" in SYSTEM_PROMPT.lower()
        assert "HIGH" in SYSTEM_PROMPT

    def test_early_stopping_instruction_present(self):
        assert "stop" in SYSTEM_PROMPT.lower()
