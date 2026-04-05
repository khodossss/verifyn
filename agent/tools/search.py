"""Web search tools — Tavily or DuckDuckGo + page fetch.

Set ``TAVILY_USE=false`` in ``.env`` to skip Tavily entirely and use
DuckDuckGo as the primary search engine with automatic page content
fetching (since DDG only returns short snippets).
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger("verifyn.search")

_TAVILY_ENABLED = os.environ.get("TAVILY_USE", "true").lower().strip() not in ("false", "0", "no")


def _tavily_search(query: str, max_results: int = 6) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(query, max_results=max_results, search_depth="advanced")
    return resp.get("results", [])


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    from ddgs import DDGS

    results = []
    for r in DDGS().text(query, max_results=max_results):
        results.append({"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")})
    return results


def _fetch_page_text(url: str, max_chars: int = 2000) -> str:
    """Fetch a URL and extract readable text. Returns empty string on failure."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text:
                return text[:max_chars]
    except Exception:
        pass

    # Fallback: requests + BeautifulSoup
    try:
        import re

        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; FactCheckBot/1.0)"}, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars]
    except Exception:
        return ""


def _enrich_ddg_results(results: list[dict], max_fetch: int = 3) -> list[dict]:
    """Fetch full page content for top DDG results (snippets are often too short).

    DDG typically returns 100-300 char snippets vs Tavily's 600+ char extracts.
    We fetch the actual page for any result with < 500 chars of content.
    """
    for i, r in enumerate(results):
        if i >= max_fetch:
            break
        url = r.get("url", "")
        snippet = r.get("content", "")
        # Skip YouTube and other non-article URLs
        if not url or any(skip in url for skip in ("youtube.com", "youtu.be", "twitter.com", "x.com")):
            continue
        if len(snippet) < 500:
            page_text = _fetch_page_text(url, max_chars=3000)
            if page_text and len(page_text) > len(snippet):
                r["content"] = page_text
                logger.debug("Enriched DDG result %s (%d -> %d chars)", url, len(snippet), len(page_text))
    return results


def _search_with_fallback(query: str, max_results: int = 6) -> list[dict] | str:
    """Run search and return results list or error string.

    If TAVILY_USE=false, uses DuckDuckGo as primary with page content fetching.
    Otherwise tries Tavily first, falls back to DuckDuckGo.
    """
    if _TAVILY_ENABLED:
        try:
            return _tavily_search(query, max_results)
        except Exception:
            logger.debug("Tavily failed, falling back to DuckDuckGo")

    # DuckDuckGo (primary when Tavily disabled, fallback otherwise)
    try:
        results = _ddg_search(query, max_results)
        if results:
            results = _enrich_ddg_results(results)
        return results
    except Exception as e:
        return f"Search failed: {e}"


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    parts = []
    for r in results:
        parts.append(f"**{r.get('title', 'No title')}**\nURL: {r.get('url', '')}\n{r.get('content', '')[:600]}")
    return "\n\n---\n\n".join(parts)


@tool
def web_search(query: str) -> str:
    """Search the web for information about a claim, event, or person.
    Use this for lateral reading — run multiple searches with different query angles.
    Returns titles, URLs, and content snippets of the top results."""
    results = _search_with_fallback(query)
    if isinstance(results, str):
        return results
    return _format_results(results)


@tool
def search_fact_checkers(claim: str) -> str:
    """Search specifically on professional fact-checking websites for a given claim or key phrase.
    Checks: Snopes, PolitiFact, FactCheck.org, Reuters Fact Check, AP Fact Check, Full Fact, AFP Fact Check.
    Always run this for any significant claim before concluding."""
    sites = (
        "site:snopes.com OR site:politifact.com OR site:factcheck.org "
        "OR site:reuters.com/fact-check OR site:apnews.com/fact-checking "
        "OR site:fullfact.org OR site:factcheck.afp.com"
    )
    query = f"{claim} {sites}"
    results = _search_with_fallback(query, max_results=5)
    if isinstance(results, str):
        return results
    return _format_results(results)


@tool
def check_if_old_news(claim_or_title: str) -> str:
    """Check whether this is old/recycled content being presented as new.
    Searches for the earliest appearances of this story and compares dates.
    Also useful for detecting out-of-context photos or videos."""
    query = f'"{claim_or_title}" earliest original date site:web.archive.org OR "first reported" OR "original"'
    results = _search_with_fallback(query, max_results=5)
    if isinstance(results, str):
        return results

    # Also search without site filter
    query2 = f"{claim_or_title} original source when first reported"
    results2 = _search_with_fallback(query2, max_results=3)
    if isinstance(results2, list):
        results = (results + results2)[:7]

    return _format_results(results)
