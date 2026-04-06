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

    query_history (
        id              INTEGER PRIMARY KEY,
        query           TEXT,
        embedding       TEXT,          -- JSON array of floats (OpenAI embedding)
        mode            TEXT,          -- 'fast' or 'precise'
        result          TEXT,          -- JSON FactCheckResult
        reputation_updated INTEGER,
        created_at      TIMESTAMP
    )

Credibility
-----------
    Once ``true_points + false_points >= CREDIBILITY_THRESHOLD`` (default 50),
    the domain's credibility score is computed as:

        credibility = true_points / (true_points + false_points)

    Below the threshold, the DB score is treated as preliminary and the agent
    falls back to web-based reputation search.

Similarity search
-----------------
    Each saved query stores an OpenAI embedding (text-embedding-3-small).
    Before running the full agent pipeline, the system searches for similar
    past queries using cosine similarity.  This enables the agent to leverage
    previous analyses and avoid redundant work.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import numpy as np
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
    embedding = Column(Text, nullable=True)  # JSON array of floats (OpenAI embedding)
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
    ("NO_CLAIMS", True): (0.0, 0.0),
    # evidence_against (supports_claim=False)
    ("REAL", False): (0.0, 1.0),
    ("FAKE", False): (1.0, 0.0),
    ("PARTIALLY_FAKE", False): (0.5, 0.5),
    ("MISLEADING", False): (0.7, 0.3),
    ("SATIRE", False): (0.0, 0.0),
    ("UNVERIFIABLE", False): (0.0, 0.0),
    ("NO_CLAIMS", False): (0.0, 0.0),
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
    """Normalize query text — lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def update_reputation_from_result(result: Any, *, mode: str = "precise") -> int:
    """Update domain reputation DB from a FactCheckResult.

    Parameters
    ----------
    result:
        FactCheckResult with evidence and verdict.
    mode:
        Inference mode ("fast" or "precise"). Fast mode applies a 0.5x
        multiplier to all score updates since fewer sources were checked.

    Returns the number of domain records updated.
    """
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


def save_query(
    query_text: str,
    mode: str,
    result: Any,
    *,
    reputation_updated: int = 0,
    embedding: list[float] | None = None,
) -> QueryHistory:
    """Save a fact-check query and its result to the history table.

    Parameters
    ----------
    embedding:
        OpenAI embedding vector for the query text.  Stored as JSON array
        for later similarity search.
    """
    if hasattr(result, "model_dump"):
        result_json = _json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    elif isinstance(result, dict):
        result_json = _json.dumps(result, ensure_ascii=False)
    else:
        result_json = str(result)

    embedding_json = _json.dumps(embedding) if embedding else None

    with get_session() as session:
        record = QueryHistory(
            query=query_text,
            embedding=embedding_json,
            mode=mode,
            result=result_json,
            reputation_updated=reputation_updated,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(
            "Saved query #%d: mode=%s verdict=%s rep_updated=%d has_embedding=%s",
            record.id,
            mode,
            result.verdict.value if hasattr(result, "verdict") else "?",
            reputation_updated,
            embedding is not None,
        )
        return record


def get_query_history(limit: int = 50) -> list[dict]:
    """Retrieve recent query history."""
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


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").lower().strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.75"))


def compute_embedding(text: str) -> list[float]:
    """Compute an embedding vector for *text*.

    Supported providers (EMBEDDING_PROVIDER env var):
        openai   — OpenAI API (default). Model: text-embedding-3-small.
        ollama   — Local Ollama server. Model: nomic-embed-text, mxbai-embed-large, etc.
        custom   — Any OpenAI-compatible endpoint via EMBEDDING_BASE_URL.

    All providers return a list[float] embedding vector.
    """
    if EMBEDDING_PROVIDER == "ollama":
        return _embedding_ollama(text)
    if EMBEDDING_PROVIDER == "custom":
        return _embedding_custom(text)
    return _embedding_openai(text)


def _embedding_openai(text: str) -> list[float]:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
    if EMBEDDING_BASE_URL:
        kwargs["base_url"] = EMBEDDING_BASE_URL
    client = OpenAI(**kwargs)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _embedding_ollama(text: str) -> list[float]:
    import requests

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    resp = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _embedding_custom(text: str) -> list[float]:
    """Any OpenAI-compatible embeddings endpoint (vLLM, LiteLLM, HuggingFace TEI, etc.)."""
    from openai import OpenAI

    base_url = EMBEDDING_BASE_URL
    if not base_url:
        raise ValueError("EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=custom")
    api_key = os.environ.get("EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", "no-key"))
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_similar_queries(
    query_embedding: list[float],
    *,
    mode: str = "fast",
    top_k: int = 3,
    threshold: float | None = None,
) -> list[dict]:
    """Find past queries whose embeddings are similar to *query_embedding*.

    Mode logic
    ----------
    - mode='fast'   → return results from 'precise' first, then 'fast'
    - mode='precise' → return only 'precise' results

    Only the most recent result per unique normalized query is considered.

    Returns list of dicts sorted by similarity (highest first):
        [{"id", "query", "mode", "result", "similarity", "created_at"}, ...]
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    with get_session() as session:
        # Determine which modes to accept
        if mode == "precise":
            allowed_modes = ("precise",)
        else:
            allowed_modes = ("precise", "fast")

        records = (
            session.query(QueryHistory)
            .filter(
                QueryHistory.embedding.isnot(None),
                QueryHistory.mode.in_(allowed_modes),
            )
            .order_by(QueryHistory.created_at.desc(), QueryHistory.id.desc())
            .all()
        )

    if not records:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)

    # Deduplicate: keep only the latest result per normalized query text
    seen_queries: set[str] = set()
    unique_records: list[Any] = []
    for r in records:
        normalized = _normalize_query(r.query)
        if normalized not in seen_queries:
            seen_queries.add(normalized)
            unique_records.append(r)

    # Compute similarities
    scored: list[tuple[float, Any]] = []
    for r in unique_records:
        stored_vec = np.array(_json.loads(r.embedding), dtype=np.float32)
        sim = _cosine_similarity(query_vec, stored_vec)
        if sim >= threshold:
            scored.append((sim, r))

    # Sort by similarity descending, prefer precise over fast at equal similarity
    mode_rank = {"precise": 0, "fast": 1}
    scored.sort(key=lambda x: (-x[0], mode_rank.get(x[1].mode, 2)))

    results = []
    for sim, r in scored[:top_k]:
        results.append(
            {
                "id": r.id,
                "query": r.query,
                "mode": r.mode,
                "result": _json.loads(r.result),
                "similarity": round(sim, 4),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )

    return results
