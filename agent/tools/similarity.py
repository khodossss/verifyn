"""Similarity search tool: embedding-based lookup of previously verified claims."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger("verifyn.similarity")

# Display limits when formatting a previous fact-check for the agent
MAX_CLAIMS_PREVIEW = 3
MAX_EVIDENCE_PREVIEW = 4
MAX_SOURCES_PREVIEW = 5
EVIDENCE_SUMMARY_PREVIEW_CHARS = 120

# How many similar past queries to return from the DB
SIMILARITY_TOP_K = 3


def _format_previous_result(match: dict) -> str:
    """Render a similarity match as a markdown block for the agent."""
    r = match["result"]
    lines = [
        f"**Previous query** (similarity={match['similarity']:.2f}, mode={match['mode']}, date={match['created_at']}):",
        f"> {match['query']}",
        "",
        f"**Verdict:** {r.get('verdict', '?')} (confidence={r.get('confidence', '?')})",
        f"**Manipulation type:** {r.get('manipulation_type', 'NONE')}",
    ]

    if r.get("main_claims"):
        lines.append(f"**Claims:** {'; '.join(r['main_claims'][:MAX_CLAIMS_PREVIEW])}")

    if r.get("summary"):
        lines.append(f"**Summary:** {r['summary']}")

    evidence_for = r.get("evidence_for", [])
    evidence_against = r.get("evidence_against", [])
    if evidence_for or evidence_against:
        lines.append(f"**Evidence:** {len(evidence_for)} for, {len(evidence_against)} against")
        for e in (evidence_for + evidence_against)[:MAX_EVIDENCE_PREVIEW]:
            direction = "FOR" if e.get("supports_claim") else "AGAINST"
            url = e.get("url", "")
            summary_preview = e.get("summary", "")[:EVIDENCE_SUMMARY_PREVIEW_CHARS]
            lines.append(f"  - [{direction}] {e.get('source', '?')}: {summary_preview} {url}")

    sources = r.get("sources_checked", [])
    if sources:
        lines.append(f"**Sources checked:** {', '.join(sources[:MAX_SOURCES_PREVIEW])}")

    return "\n".join(lines)


@tool
def search_similar_queries(query: str) -> str:
    """Search the history of previously fact-checked queries for similar claims.

    Returns past verdicts with evidence, sources, and confidence scores.
    Use this BEFORE doing web searches — if a very similar claim was already
    checked recently with high confidence, you can build on that result
    instead of starting from scratch.

    IMPORTANT: Previous results may be outdated if the event has evolved.
    Always verify that the result is still current before reusing it.
    """
    from agent.db import compute_embedding, find_similar_queries

    try:
        embedding = compute_embedding(query)
    except Exception as exc:
        logger.warning("Failed to compute embedding for similarity search: %s", exc)
        return "Similarity search unavailable (embedding computation failed)."

    # Mode is injected at runtime via tool metadata; default to fast
    mode = getattr(search_similar_queries, "_current_mode", "fast")

    matches = find_similar_queries(embedding, mode=mode, top_k=SIMILARITY_TOP_K)

    if not matches:
        return "No similar previously checked claims found in history."

    parts = [f"Found {len(matches)} similar previous fact-check(s):\n"]
    for match in matches:
        parts.append(_format_previous_result(match))
        parts.append("\n---\n")

    return "\n".join(parts)
