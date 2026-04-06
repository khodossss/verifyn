"""Similarity search tool — find previously verified claims in query history.

Uses OpenAI embeddings + cosine similarity to locate past fact-checks
that are semantically close to the current query.  The agent can then
decide whether the previous result is still valid or needs updating.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger("verifyn.similarity")


def _format_previous_result(match: dict) -> str:
    """Format a single similarity match for the agent."""
    r = match["result"]
    lines = [
        f"**Previous query** (similarity={match['similarity']:.2f}, mode={match['mode']}, date={match['created_at']}):",
        f"> {match['query']}",
        "",
        f"**Verdict:** {r.get('verdict', '?')} (confidence={r.get('confidence', '?')})",
        f"**Manipulation type:** {r.get('manipulation_type', 'NONE')}",
    ]

    if r.get("main_claims"):
        lines.append(f"**Claims:** {'; '.join(r['main_claims'][:3])}")

    if r.get("summary"):
        lines.append(f"**Summary:** {r['summary']}")

    evidence_for = r.get("evidence_for", [])
    evidence_against = r.get("evidence_against", [])
    if evidence_for or evidence_against:
        lines.append(f"**Evidence:** {len(evidence_for)} for, {len(evidence_against)} against")
        for e in (evidence_for + evidence_against)[:4]:
            direction = "FOR" if e.get("supports_claim") else "AGAINST"
            url = e.get("url", "")
            lines.append(f"  - [{direction}] {e.get('source', '?')}: {e.get('summary', '')[:120]} {url}")

    sources = r.get("sources_checked", [])
    if sources:
        lines.append(f"**Sources checked:** {', '.join(sources[:5])}")

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

    matches = find_similar_queries(embedding, mode=mode, top_k=3)

    if not matches:
        return "No similar previously checked claims found in history."

    parts = [f"Found {len(matches)} similar previous fact-check(s):\n"]
    for match in matches:
        parts.append(_format_previous_result(match))
        parts.append("\n---\n")

    return "\n".join(parts)
