"""Tests for the article content extractor tool."""

import json
import sys
from unittest.mock import MagicMock, patch

from agent.tools.extractor import extract_article_content

SAMPLE_PAYLOAD = {
    "title": "Test Article",
    "date": "2024-01-01",
    "author": "Jane Doe",
    "text": "This is the article body text.",
}


def _make_trafilatura(fetch_return=b"<html/>", extract_return=None):
    mock = MagicMock()
    mock.fetch_url.return_value = fetch_return
    mock.extract.return_value = extract_return or json.dumps(SAMPLE_PAYLOAD)
    return mock


def _invoke(url="https://example.com/article"):
    return extract_article_content.invoke({"url": url})


class TestExtractArticleContent:
    def test_successful_extraction(self):
        mock_traf = _make_trafilatura()
        with patch.dict(sys.modules, {"trafilatura": mock_traf}):
            result = _invoke()
        assert "Test Article" in result
        assert "Jane Doe" in result
        assert "2024-01-01" in result
        assert "article body text" in result

    def test_fetch_returns_none(self):
        mock_traf = _make_trafilatura(fetch_return=None)
        with patch.dict(sys.modules, {"trafilatura": mock_traf}):
            result = _invoke()
        assert "Could not fetch content" in result

    def test_extract_returns_none(self):
        mock_traf = _make_trafilatura(extract_return=None)
        mock_traf.extract.return_value = None
        with patch.dict(sys.modules, {"trafilatura": mock_traf}):
            result = _invoke()
        assert "Could not extract" in result

    def test_long_text_truncated(self):
        long_payload = {**SAMPLE_PAYLOAD, "text": "word " * 2000}
        mock_traf = _make_trafilatura(extract_return=json.dumps(long_payload))
        with patch.dict(sys.modules, {"trafilatura": mock_traf}):
            result = _invoke()
        assert "truncated" in result

    def test_url_included_in_output(self):
        mock_traf = _make_trafilatura()
        with patch.dict(sys.modules, {"trafilatura": mock_traf}):
            result = _invoke("https://bbc.com/news/123")
        assert "https://bbc.com/news/123" in result

    def test_trafilatura_failure_triggers_bs4_fallback(self):
        mock_traf = _make_trafilatura()
        mock_traf.fetch_url.side_effect = Exception("network error")

        mock_resp = MagicMock()
        mock_resp.text = "<html><head><title>Fallback</title></head><body>fallback body</body></html>"
        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        mock_soup_instance = MagicMock()
        mock_soup_instance.title.get_text.return_value = "Fallback"
        mock_soup_instance.get_text.return_value = "fallback body"
        mock_soup_instance.__call__ = lambda *a, **kw: []
        mock_bs4 = MagicMock()
        mock_bs4.BeautifulSoup.return_value = mock_soup_instance

        with patch.dict(sys.modules, {"trafilatura": mock_traf, "requests": mock_requests, "bs4": mock_bs4}):
            result = _invoke()
        assert "fallback" in result.lower() or "Fallback" in result

    def test_both_methods_fail_returns_error(self):
        mock_traf = _make_trafilatura()
        mock_traf.fetch_url.side_effect = Exception("traf error")
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("network error")

        with patch.dict(sys.modules, {"trafilatura": mock_traf, "requests": mock_requests}):
            result = _invoke()
        assert "Failed to extract" in result
