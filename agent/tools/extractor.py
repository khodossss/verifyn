"""Article content extraction via trafilatura, with BeautifulSoup fallback."""

from __future__ import annotations

import re

from langchain_core.tools import tool

MAX_ARTICLE_CHARS = 4000
FALLBACK_HTTP_TIMEOUT_SECONDS = 10


@tool
def extract_article_content(url: str) -> str:
    """Fetch and extract the full text content of a web article at the given URL.
    Use this to read the actual content of sources found via web_search.
    Returns the article title, publish date, author, and body text."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch content from {url}"

        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            with_metadata=True,
            output_format="json",
        )

        if not result:
            return f"Could not extract readable content from {url}"

        import json

        data = json.loads(result)

        title = data.get("title", "")
        date = data.get("date", "unknown date")
        author = data.get("author", "unknown author")
        text = data.get("text", "")

        # Truncate very long articles to save context
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS] + "\n\n[... truncated ...]"

        return f"**Title:** {title}\n**Date:** {date}\n**Author:** {author}\n**URL:** {url}\n\n**Content:**\n{text}"

    except Exception as e:
        # Fallback: basic requests + BeautifulSoup
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "Mozilla/5.0 (compatible; FactCheckBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=FALLBACK_HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts/styles
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()

            title = soup.title.get_text(strip=True) if soup.title else ""
            body = soup.get_text(separator="\n", strip=True)
            body = re.sub(r"\n{3,}", "\n\n", body)[:MAX_ARTICLE_CHARS]

            return f"**Title:** {title}\n**URL:** {url}\n\n**Content (fallback extraction):**\n{body}"

        except Exception as e2:
            return f"Failed to extract content from {url}: {e} / {e2}"
