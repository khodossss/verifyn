"""Domain reputation database — learns from agent verdicts over time.

After each fact-check, the agent updates reputation scores for every source
domain referenced in evidence_for / evidence_against.  Over time, domains
accumulate true_points and false_points that reflect how often they appear
in reliable vs unreliable contexts.

Schema
------
    domain_reputation (
        domain          TEXT PRIMARY KEY,
        true_points     REAL DEFAULT 0,
        false_points    REAL DEFAULT 0,
        total_checks    INTEGER DEFAULT 0,
        first_seen      TIMESTAMP,
        last_checked    TIMESTAMP,
        comment         TEXT
    )

Credibility
-----------
    Once ``true_points + false_points >= CREDIBILITY_THRESHOLD`` (default 50),
    the domain's credibility score is computed as:

        credibility = true_points / (true_points + false_points)

    Below the threshold, the DB score is treated as preliminary and the agent
    falls back to web-based reputation search.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger("verifyn.db")

Base = declarative_base()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CREDIBILITY_THRESHOLD = int(os.environ.get("CREDIBILITY_THRESHOLD", "50"))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/verifyn.db")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DomainReputation(Base):
    __tablename__ = "domain_reputation"

    domain = Column(String, primary_key=True)
    true_points = Column(Float, default=0.0, nullable=False)
    false_points = Column(Float, default=0.0, nullable=False)
    total_checks = Column(Integer, default=0, nullable=False)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_checked = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    comment = Column(Text, default="")


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    query_hash = Column(String, nullable=False, index=True)
    mode = Column(String, default="precise")
    result = Column(Text, nullable=False)  # JSON string of FactCheckResult
    reputation_updated = Column(Integer, default=0)  # 1 if this query updated reputation
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

# (verdict, supports_claim) → (true_delta, false_delta)
SCORING_TABLE: dict[tuple[str, bool], tuple[float, float]] = {
    # evidence_for (supports_claim=True)
    ("REAL", True): (1.0, 0.0),
    ("FAKE", True): (0.0, 1.0),
    ("PARTIALLY_FAKE", True): (0.5, 0.5),
    ("MISLEADING", True): (0.3, 0.7),
    ("SATIRE", True): (0.0, 0.0),
    ("UNVERIFIABLE", True): (0.0, 0.0),
    # evidence_against (supports_claim=False)
    ("REAL", False): (0.0, 1.0),
    ("FAKE", False): (1.0, 0.0),
    ("PARTIALLY_FAKE", False): (0.5, 0.5),
    ("MISLEADING", False): (0.7, 0.3),
    ("SATIRE", False): (0.0, 0.0),
    ("UNVERIFIABLE", False): (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, echo=False)
        Base.metadata.create_all(_engine)
        logger.info("Database connected: %s", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine())
    return _SessionFactory()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_domain(url: str) -> str | None:
    """Extract root domain from a URL. Returns None for invalid URLs."""
    if not url or not url.startswith("http"):
        return None
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def get_domain(domain: str) -> DomainReputation | None:
    """Look up a domain's reputation record. Returns None if not found."""
    with get_session() as session:
        return session.query(DomainReputation).filter_by(domain=domain.lower()).first()


def get_domain_credibility(domain: str) -> dict[str, Any] | None:
    """Get domain credibility if enough data has accumulated.

    Returns
    -------
    dict with keys: domain, true_points, false_points, total_checks,
                    credibility (float 0-1), above_threshold (bool)
    None if domain not found in DB.
    """
    record = get_domain(domain)
    if record is None:
        return None

    total_points = record.true_points + record.false_points
    above_threshold = total_points >= CREDIBILITY_THRESHOLD
    credibility = (record.true_points / total_points) if total_points > 0 else 0.5

    return {
        "domain": record.domain,
        "true_points": record.true_points,
        "false_points": record.false_points,
        "total_checks": record.total_checks,
        "credibility": round(credibility, 3),
        "above_threshold": above_threshold,
        "comment": record.comment,
    }


def update_domain_scores(
    domain: str,
    true_delta: float,
    false_delta: float,
    comment: str = "",
) -> DomainReputation:
    """Upsert a domain's reputation scores."""
    with get_session() as session:
        record = session.query(DomainReputation).filter_by(domain=domain.lower()).first()
        now = datetime.now(timezone.utc)

        if record is None:
            record = DomainReputation(
                domain=domain.lower(),
                true_points=true_delta,
                false_points=false_delta,
                total_checks=1,
                first_seen=now,
                last_checked=now,
                comment=comment,
            )
            session.add(record)
        else:
            record.true_points += true_delta
            record.false_points += false_delta
            record.total_checks += 1
            record.last_checked = now
            if comment:
                record.comment = comment

        session.commit()
        session.refresh(record)
        logger.debug(
            "Domain %s updated: true=%.1f false=%.1f checks=%d",
            domain,
            record.true_points,
            record.false_points,
            record.total_checks,
        )
        return record


# ---------------------------------------------------------------------------
# Post-verdict update
# ---------------------------------------------------------------------------

# Mode multipliers — fast mode uses fewer searches so scores are less reliable
MODE_MULTIPLIER: dict[str, float] = {
    "fast": 0.5,
    "precise": 1.0,
}


def _normalize_query(text: str) -> str:
    """Normalize query text for dedup hashing — lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _query_hash(text: str) -> str:
    """Compute a stable hash for deduplication of similar queries."""
    return hashlib.sha256(_normalize_query(text).encode("utf-8")).hexdigest()[:16]


def _is_duplicate_query(query_text: str) -> bool:
    """Check if a query with the same hash already updated reputation."""
    qh = _query_hash(query_text)
    with get_session() as session:
        existing = session.query(QueryHistory).filter_by(query_hash=qh, reputation_updated=1).first()
        return existing is not None


def update_reputation_from_result(result: Any, *, mode: str = "precise", query_text: str = "") -> int:
    """Update domain reputation DB from a FactCheckResult.

    Deduplication: if the same query (by normalized hash) already updated
    reputation scores, this call is skipped. This prevents a single article
    from inflating scores when the same claim is checked repeatedly.

    Parameters
    ----------
    result:
        FactCheckResult with evidence and verdict.
    mode:
        Inference mode ("fast" or "precise"). Fast mode applies a 0.5x
        multiplier to all score updates since fewer sources were checked.
    query_text:
        Original query text for deduplication.

    Returns the number of domain records updated (0 if deduplicated).
    """
    # Dedup check
    if query_text and _is_duplicate_query(query_text):
        logger.info("Skipping reputation update — duplicate query (hash=%s)", _query_hash(query_text))
        return 0

    verdict = result.verdict.value
    multiplier = MODE_MULTIPLIER.get(mode, 1.0)
    updated = 0

    # Collect (domain, supports_claim) pairs from evidence
    evidence_items: list[tuple[str, bool]] = []

    for item in result.evidence_for or []:
        if item.url:
            domain = extract_domain(item.url)
            if domain:
                evidence_items.append((domain, True))

    for item in result.evidence_against or []:
        if item.url:
            domain = extract_domain(item.url)
            if domain:
                evidence_items.append((domain, False))

    # Also include sources_checked that weren't in evidence
    evidence_domains = {d for d, _ in evidence_items}
    for url in result.sources_checked or []:
        domain = extract_domain(url)
        if domain and domain not in evidence_domains:
            evidence_items.append((domain, True))

    for domain, supports_claim in evidence_items:
        score_key = (verdict, supports_claim)
        true_delta, false_delta = SCORING_TABLE.get(score_key, (0.0, 0.0))

        true_delta *= multiplier
        false_delta *= multiplier

        if true_delta == 0.0 and false_delta == 0.0:
            continue

        update_domain_scores(domain, true_delta, false_delta)
        updated += 1

    if updated:
        logger.info("Updated %d domain reputation records for verdict=%s", updated, verdict)

    return updated


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------


def save_query(query_text: str, mode: str, result: Any, *, reputation_updated: int = 0) -> QueryHistory:
    """Save a fact-check query and its result to the history table."""
    import json as _json

    if hasattr(result, "model_dump"):
        result_json = _json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    elif isinstance(result, dict):
        result_json = _json.dumps(result, ensure_ascii=False)
    else:
        result_json = str(result)

    with get_session() as session:
        record = QueryHistory(
            query=query_text,
            query_hash=_query_hash(query_text),
            mode=mode,
            result=result_json,
            reputation_updated=reputation_updated,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(
            "Saved query #%d: mode=%s verdict=%s rep_updated=%d",
            record.id,
            mode,
            result.verdict.value if hasattr(result, "verdict") else "?",
            reputation_updated,
        )
        return record


def get_query_history(limit: int = 50) -> list[dict]:
    """Retrieve recent query history."""
    import json as _json

    with get_session() as session:
        records = session.query(QueryHistory).order_by(QueryHistory.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "query": r.query,
                "mode": r.mode,
                "result": _json.loads(r.result),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
