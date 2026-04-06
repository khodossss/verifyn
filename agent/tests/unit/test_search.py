"""Tests for web search tools."""

from unittest.mock import patch

from agent.tools.search import _format_results, check_if_old_news, search_fact_checkers, web_search

SAMPLE_RESULTS = [
    {"title": "BBC News", "url": "https://bbc.com/article", "content": "Russia invaded Ukraine on Feb 24."},
    {"title": "Reuters", "url": "https://reuters.com/article", "content": "Confirmed by multiple sources."},
]


class TestFormatResults:
    def test_empty_returns_no_results(self):
        assert _format_results([]) == "No results found."

    def test_single_result_contains_title_and_url(self):
        out = _format_results([SAMPLE_RESULTS[0]])
        assert "BBC News" in out
        assert "https://bbc.com/article" in out
        assert "Russia invaded" in out

    def test_multiple_results_separated(self):
        out = _format_results(SAMPLE_RESULTS)
        assert "---" in out
        assert "BBC News" in out
        assert "Reuters" in out

    def test_content_truncated_at_600(self):
        long_content = "x" * 700
        out = _format_results([{"title": "T", "url": "http://x.com", "content": long_content}])
        assert len(out) < 700

    def test_missing_fields_handled(self):
        out = _format_results([{}])
        assert "No title" in out


class TestWebSearch:
    @patch("agent.tools.search._tavily_search", return_value=SAMPLE_RESULTS)
    def test_tavily_success(self, mock_tavily):
        result = web_search.invoke({"query": "Russia Ukraine"})
        mock_tavily.assert_called_once_with("Russia Ukraine", 6)
        assert "BBC News" in result

    @patch("agent.tools.search._ddg_search", return_value=SAMPLE_RESULTS)
    @patch("agent.tools.search._tavily_search", side_effect=Exception("no key"))
    def test_ddg_fallback_on_tavily_failure(self, mock_tavily, mock_ddg):
        result = web_search.invoke({"query": "test query"})
        mock_ddg.assert_called_once()
        assert "BBC News" in result

    @patch("agent.tools.search._ddg_search", side_effect=Exception("ddg down"))
    @patch("agent.tools.search._tavily_search", side_effect=Exception("tavily down"))
    def test_both_fail_returns_error_string(self, mock_tavily, mock_ddg):
        result = web_search.invoke({"query": "test"})
        assert "Search failed" in result

    @patch("agent.tools.search._tavily_search", return_value=[])
    def test_empty_results(self, _):
        result = web_search.invoke({"query": "nothing found"})
        assert "No results found" in result


class TestSearchFactCheckers:
    @patch("agent.tools.search._tavily_search", return_value=SAMPLE_RESULTS)
    def test_builds_site_scoped_query(self, mock_tavily):
        search_fact_checkers.invoke({"claim": "vaccines cause autism"})
        call_query = mock_tavily.call_args[0][0]
        assert "vaccines cause autism" in call_query
        assert "snopes.com" in call_query

    @patch("agent.tools.search._ddg_search", return_value=SAMPLE_RESULTS)
    @patch("agent.tools.search._tavily_search", side_effect=Exception("fail"))
    def test_fallback_works(self, _tavily, mock_ddg):
        result = search_fact_checkers.invoke({"claim": "test claim"})
        mock_ddg.assert_called_once()
        assert "BBC News" in result


class TestCheckIfOldNews:
    @patch("agent.tools.search._tavily_search", return_value=SAMPLE_RESULTS)
    def test_returns_formatted_results(self, mock_tavily):
        result = check_if_old_news.invoke({"claim_or_title": "Some old story"})
        assert "BBC News" in result or "No results" in result

    @patch("agent.tools.search._tavily_search", side_effect=Exception("fail"))
    @patch("agent.tools.search._ddg_search", side_effect=Exception("fail"))
    def test_both_fail_returns_error(self, _ddg, _tavily):
        result = check_if_old_news.invoke({"claim_or_title": "test"})
        assert "failed" in result.lower()
