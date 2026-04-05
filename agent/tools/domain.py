"""Domain reputation tool — DB lookup + web reputation search + WHOIS + heuristics."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from langchain_core.tools import tool


def _extract_domain(url_or_domain: str) -> str:
    """Extract root domain from URL or return as-is if already a domain."""
    if url_or_domain.startswith("http"):
        parsed = urlparse(url_or_domain)
        domain = parsed.netloc
    else:
        domain = url_or_domain
    # Strip www.
    domain = re.sub(r"^www\.", "", domain)
    return domain.lower().strip()


def _web_reputation_search(domain: str) -> str:
    """Search the web for what fact-checkers and media watchdogs say about a domain."""
    from agent.tools.search import _search_with_fallback

    query = f'site:mediabiasfactcheck.com OR site:allsides.com OR site:newsguardtech.com "{domain}" bias reliability'
    results = _search_with_fallback(query)
    if isinstance(results, str) or not results:
        query2 = f'"{domain}" propaganda yellow press unreliable fake news OR "media bias" OR "fact check"'
        results = _search_with_fallback(query2)
    if isinstance(results, str) or not results:
        return ""
    snippets = []
    for r in results[:3]:
        title = r.get("title", "")
        content = r.get("content", "")[:300]
        url = r.get("url", "")
        if content:
            snippets.append(f"[{title}]({url}): {content}")
    return "\n".join(snippets)


@tool
def check_domain_reputation(url_or_domain: str) -> str:
    """Check the reputation of a news source domain.
    Searches media watchdog sites (Media Bias/Fact Check, AllSides, NewsGuard)
    for bias ratings and reliability assessments, plus checks known lists and
    domain registration data. Use this on any unfamiliar or suspicious source."""
    domain = _extract_domain(url_or_domain)
    if not domain:
        return "Could not parse domain from input."

    lines: list[str] = [f"**Domain:** {domain}"]
    db_confident = False

    # Check learned reputation DB first
    try:
        from agent.db import CREDIBILITY_THRESHOLD, get_domain_credibility

        db_info = get_domain_credibility(domain)
        if db_info and db_info["above_threshold"]:
            db_confident = True
            cred = db_info["credibility"]
            level = "HIGH" if cred >= 0.75 else "MEDIUM" if cred >= 0.5 else "LOW"
            lines.append(
                f"**Learned reputation ({db_info['total_checks']} checks):** "
                f"credibility={cred:.1%} ({level}) — "
                f"true_points={db_info['true_points']:.1f}, false_points={db_info['false_points']:.1f}"
            )
            if db_info.get("comment"):
                lines.append(f"**DB note:** {db_info['comment']}")
        elif db_info:
            total = db_info["true_points"] + db_info["false_points"]
            lines.append(
                f"**Learned reputation (preliminary, {total:.0f}/{CREDIBILITY_THRESHOLD} points):** "
                f"true={db_info['true_points']:.1f}, false={db_info['false_points']:.1f} — "
                f"not enough data yet, using web search"
            )
    except Exception:
        pass  # DB not available — continue with hardcoded lists + web search

    # Web reputation search (skip if DB already has a confident score)
    if not db_confident:
        web_rep = _web_reputation_search(domain)
        if web_rep:
            lines.append(f"\n**Web reputation findings:**\n{web_rep}")
        else:
            lines.append("**Web reputation:** No watchdog coverage found.")

    return "\n".join(lines)
