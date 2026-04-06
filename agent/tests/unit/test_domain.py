"""Tests for the domain reputation tool."""

from unittest.mock import patch

from agent.tools.domain import _extract_domain, _web_reputation_search, check_domain_reputation


class TestExtractDomain:
    def test_full_url(self):
        assert _extract_domain("https://www.bbc.com/news/article") == "bbc.com"

    def test_url_without_www(self):
        assert _extract_domain("https://reuters.com/article") == "reuters.com"

    def test_bare_domain(self):
        assert _extract_domain("snopes.com") == "snopes.com"

    def test_www_stripped(self):
        assert _extract_domain("www.nytimes.com") == "nytimes.com"

    def test_subdomain_kept(self):
        assert _extract_domain("https://fact-check.reuters.com/path") == "fact-check.reuters.com"

    def test_lowercase(self):
        assert _extract_domain("https://BBC.COM/news") == "bbc.com"


class TestWebReputationSearch:
    def test_returns_snippets_on_success(self):
        fake_results = [
            {
                "title": "MBFC: Example Site",
                "url": "https://mediabiasfactcheck.com/example",
                "content": "Rated LOW credibility. Known for publishing propaganda.",
            },
        ]
        with patch("agent.tools.search._search_with_fallback", return_value=fake_results):
            result = _web_reputation_search("example.com")
        assert "MBFC" in result or "propaganda" in result

    def test_falls_back_to_broader_query(self):
        broad_results = [
            {"title": "AllSides: Bias rating", "url": "https://allsides.com/x", "content": "Leans far right."},
        ]
        with patch("agent.tools.search._search_with_fallback", side_effect=[[], broad_results]):
            result = _web_reputation_search("biasedsite.com")
        assert "far right" in result or "AllSides" in result

    def test_returns_empty_string_when_all_fail(self):
        with patch("agent.tools.search._search_with_fallback", return_value="Search failed: error"):
            result = _web_reputation_search("unknown.com")
        assert result == ""

    def test_returns_empty_on_no_results(self):
        with patch("agent.tools.search._search_with_fallback", return_value=[]):
            result = _web_reputation_search("fakery.com")
        assert result == ""

    def test_limits_output_to_3_snippets(self):
        many_results = [
            {"title": f"Source {i}", "url": f"http://s{i}.com", "content": f"Content {i}"} for i in range(10)
        ]
        with patch("agent.tools.search._search_with_fallback", return_value=many_results):
            result = _web_reputation_search("example.com")
        assert result.count("Source") <= 3


class TestCheckDomainReputation:
    def _invoke(self, value):
        return check_domain_reputation.invoke({"url_or_domain": value})

    def test_empty_input_returns_error(self):
        result = self._invoke("")
        assert "Could not parse" in result

    def test_unknown_domain_triggers_web_search(self):
        with patch("agent.tools.domain._web_reputation_search", return_value="Rated LOW credibility") as mock_web:
            result = self._invoke("randomsite123.com")
        mock_web.assert_called_once_with("randomsite123.com")
        assert "LOW credibility" in result

    def test_unknown_domain_no_web_results(self):
        with patch("agent.tools.domain._web_reputation_search", return_value=""):
            result = self._invoke("randomsite123.com")
        assert "No watchdog coverage found" in result

    def test_db_confident_skips_web_search(self):
        """When DB has above-threshold data, web search is skipped."""
        db_info = {
            "domain": "trusted.com",
            "true_points": 45.0,
            "false_points": 5.0,
            "total_checks": 50,
            "credibility": 0.9,
            "above_threshold": True,
            "comment": "",
        }
        with (
            patch("agent.db.get_domain_credibility", return_value=db_info),
            patch("agent.db.CREDIBILITY_THRESHOLD", 50),
            patch("agent.tools.domain._web_reputation_search") as mock_web,
        ):
            result = self._invoke("trusted.com")
        mock_web.assert_not_called()
        assert "90.0%" in result
        assert "HIGH" in result

    def test_db_below_threshold_still_runs_web_search(self):
        """When DB has below-threshold data, web search still runs."""
        db_info = {
            "domain": "new-site.com",
            "true_points": 3.0,
            "false_points": 1.0,
            "total_checks": 4,
            "credibility": 0.75,
            "above_threshold": False,
            "comment": "",
        }
        with (
            patch("agent.db.get_domain_credibility", return_value=db_info),
            patch("agent.db.CREDIBILITY_THRESHOLD", 50),
            patch("agent.tools.domain._web_reputation_search", return_value="Some info") as mock_web,
        ):
            result = self._invoke("new-site.com")
        mock_web.assert_called_once()
        assert "preliminary" in result
        assert "Some info" in result

    def test_output_contains_domain(self):
        with patch("agent.tools.domain._web_reputation_search", return_value=""):
            result = self._invoke("https://bbc.com/news")
        assert "bbc.com" in result
