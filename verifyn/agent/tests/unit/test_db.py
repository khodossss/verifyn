"""Tests for domain reputation database and similarity search."""

from __future__ import annotations

import os

import pytest

# Force SQLite in-memory for tests
os.environ["DATABASE_URL"] = "sqlite://"

import numpy as np

from verifyn.agent.db import (
    SCORING_TABLE,
    Base,
    _cosine_similarity,
    _get_engine,
    _normalize_query,
    extract_domain,
    find_similar_queries,
    get_domain,
    get_domain_credibility,
    get_query_history,
    save_query,
    update_domain_scores,
    update_reputation_from_result,
)


@pytest.fixture(autouse=True)
def fresh_db():
    """Reset the DB before each test."""
    import verifyn.agent.db as db_mod

    db_mod._engine = None
    db_mod._SessionFactory = None
    engine = _get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    db_mod._engine = None
    db_mod._SessionFactory = None


# ---------------------------------------------------------------------------
# extract_domain
# ---------------------------------------------------------------------------


class TestExtractDomain:
    def test_full_url(self):
        assert extract_domain("https://www.reuters.com/article/123") == "reuters.com"

    def test_no_www(self):
        assert extract_domain("https://bbc.com/news") == "bbc.com"

    def test_invalid(self):
        assert extract_domain("not a url") is None

    def test_empty(self):
        assert extract_domain("") is None

    def test_none_input(self):
        assert extract_domain(None) is None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestUpdateDomainScores:
    def test_creates_new_record(self):
        record = update_domain_scores("example.com", 1.0, 0.0)
        assert record.domain == "example.com"
        assert record.true_points == 1.0
        assert record.false_points == 0.0
        assert record.total_checks == 1

    def test_updates_existing_record(self):
        update_domain_scores("example.com", 1.0, 0.0)
        record = update_domain_scores("example.com", 0.5, 0.5)
        assert record.true_points == 1.5
        assert record.false_points == 0.5
        assert record.total_checks == 2

    def test_case_insensitive(self):
        update_domain_scores("Reuters.COM", 1.0, 0.0)
        record = get_domain("reuters.com")
        assert record is not None
        assert record.true_points == 1.0

    def test_comment_stored(self):
        update_domain_scores("example.com", 1.0, 0.0, comment="major outlet")
        record = get_domain("example.com")
        assert record.comment == "major outlet"


class TestGetDomainCredibility:
    def test_not_found(self):
        assert get_domain_credibility("nonexistent.com") is None

    def test_below_threshold(self):
        update_domain_scores("new-site.com", 3.0, 1.0)
        info = get_domain_credibility("new-site.com")
        assert info is not None
        assert info["above_threshold"] is False
        assert info["total_checks"] == 1

    def test_above_threshold(self):
        # Accumulate enough points
        for _ in range(60):
            update_domain_scores("trusted.com", 1.0, 0.0)
        info = get_domain_credibility("trusted.com")
        assert info["above_threshold"] is True
        assert info["credibility"] == 1.0

    def test_credibility_calculation(self):
        # 30 true + 20 false = 50 total → above threshold
        update_domain_scores("mixed.com", 30.0, 20.0)
        info = get_domain_credibility("mixed.com")
        assert info["above_threshold"] is True
        assert info["credibility"] == 0.6


# ---------------------------------------------------------------------------
# Scoring table
# ---------------------------------------------------------------------------


class TestScoringTable:
    def test_all_verdicts_covered_for_supports(self):
        for verdict in ("REAL", "FAKE", "PARTIALLY_FAKE", "MISLEADING", "SATIRE", "UNVERIFIABLE"):
            assert (verdict, True) in SCORING_TABLE
            assert (verdict, False) in SCORING_TABLE

    def test_real_supports_gives_true_point(self):
        assert SCORING_TABLE[("REAL", True)] == (1.0, 0.0)

    def test_fake_supports_gives_false_point(self):
        assert SCORING_TABLE[("FAKE", True)] == (0.0, 1.0)

    def test_real_against_gives_false_point(self):
        assert SCORING_TABLE[("REAL", False)] == (0.0, 1.0)

    def test_fake_against_gives_true_point(self):
        assert SCORING_TABLE[("FAKE", False)] == (1.0, 0.0)

    def test_partially_fake_is_symmetric(self):
        assert SCORING_TABLE[("PARTIALLY_FAKE", True)] == (0.5, 0.5)
        assert SCORING_TABLE[("PARTIALLY_FAKE", False)] == (0.5, 0.5)

    def test_satire_is_zero(self):
        assert SCORING_TABLE[("SATIRE", True)] == (0.0, 0.0)
        assert SCORING_TABLE[("SATIRE", False)] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Post-verdict update
# ---------------------------------------------------------------------------


class TestUpdateReputationFromResult:
    def _make_result(self, verdict="REAL", evidence_for=None, evidence_against=None, sources_checked=None):
        """Create a mock FactCheckResult-like object."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.verdict.value = verdict
        result.evidence_for = evidence_for or []
        result.evidence_against = evidence_against or []
        result.sources_checked = sources_checked or []
        return result

    def _make_evidence(self, url, supports=True):
        from unittest.mock import MagicMock

        item = MagicMock()
        item.url = url
        item.supports_claim = supports
        return item

    def test_updates_evidence_for_domains(self):
        result = self._make_result(
            verdict="REAL",
            evidence_for=[self._make_evidence("https://reuters.com/article/1")],
        )
        count = update_reputation_from_result(result)
        assert count == 1
        record = get_domain("reuters.com")
        assert record.true_points == 1.0
        assert record.false_points == 0.0

    def test_updates_evidence_against_domains(self):
        result = self._make_result(
            verdict="FAKE",
            evidence_against=[self._make_evidence("https://snopes.com/factcheck/1", supports=False)],
        )
        count = update_reputation_from_result(result)
        assert count == 1
        record = get_domain("snopes.com")
        assert record.true_points == 1.0  # Correctly debunked fake

    def test_fake_verdict_with_supporting_source(self):
        result = self._make_result(
            verdict="FAKE",
            evidence_for=[self._make_evidence("https://shady-news.com/article")],
        )
        update_reputation_from_result(result)
        record = get_domain("shady-news.com")
        assert record.false_points == 1.0  # Supported a fake claim

    def test_sources_checked_fallback(self):
        result = self._make_result(
            verdict="REAL",
            sources_checked=["https://bbc.com/news/1"],
        )
        count = update_reputation_from_result(result)
        assert count == 1
        record = get_domain("bbc.com")
        assert record.true_points == 1.0

    def test_no_urls_returns_zero(self):
        result = self._make_result(verdict="REAL")
        count = update_reputation_from_result(result)
        assert count == 0

    def test_unverifiable_skips_scoring(self):
        result = self._make_result(
            verdict="UNVERIFIABLE",
            evidence_for=[self._make_evidence("https://example.com/1")],
        )
        count = update_reputation_from_result(result)
        assert count == 0  # UNVERIFIABLE → (0, 0)

    def test_fast_mode_halves_scores(self):
        result = self._make_result(
            verdict="REAL",
            evidence_for=[self._make_evidence("https://reuters.com/article/1")],
        )
        count = update_reputation_from_result(result, mode="fast")
        assert count == 1
        record = get_domain("reuters.com")
        assert record.true_points == 0.5  # 1.0 * 0.5
        assert record.false_points == 0.0

    def test_precise_mode_full_scores(self):
        result = self._make_result(
            verdict="REAL",
            evidence_for=[self._make_evidence("https://reuters.com/article/1")],
        )
        update_reputation_from_result(result, mode="precise")
        record = get_domain("reuters.com")
        assert record.true_points == 1.0

    def test_fast_mode_accumulates_correctly(self):
        result = self._make_result(
            verdict="FAKE",
            evidence_for=[self._make_evidence("https://shady.com/article")],
        )
        # Two fast checks
        update_reputation_from_result(result, mode="fast")
        update_reputation_from_result(result, mode="fast")
        record = get_domain("shady.com")
        assert record.false_points == 1.0  # 2 * (1.0 * 0.5)

    def test_multiple_sources_accumulated(self):
        result = self._make_result(
            verdict="REAL",
            evidence_for=[
                self._make_evidence("https://reuters.com/a"),
                self._make_evidence("https://bbc.com/b"),
            ],
            evidence_against=[
                self._make_evidence("https://shady.com/c", supports=False),
            ],
        )
        count = update_reputation_from_result(result)
        assert count == 3

        assert get_domain("reuters.com").true_points == 1.0
        assert get_domain("bbc.com").true_points == 1.0
        assert get_domain("shady.com").false_points == 1.0


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------


class TestQueryHistory:
    def _make_result(self, verdict="REAL"):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.verdict.value = verdict
        result.model_dump.return_value = {"verdict": verdict, "confidence": 0.9}
        return result

    def test_save_and_retrieve(self):
        result = self._make_result("FAKE")
        save_query("Is the earth flat?", "fast", result)

        history = get_query_history(limit=10)
        assert len(history) == 1
        assert history[0]["query"] == "Is the earth flat?"
        assert history[0]["mode"] == "fast"
        assert history[0]["result"]["verdict"] == "FAKE"

    def test_multiple_queries_ordered_by_newest(self):
        save_query("query 1", "fast", self._make_result("REAL"))
        save_query("query 2", "precise", self._make_result("FAKE"))

        history = get_query_history(limit=10)
        assert len(history) == 2
        queries = {h["query"] for h in history}
        assert queries == {"query 1", "query 2"}

    def test_limit_works(self):
        for i in range(5):
            save_query(f"query {i}", "fast", self._make_result())

        history = get_query_history(limit=3)
        assert len(history) == 3

    def test_result_stored_as_json(self):
        result = self._make_result("MISLEADING")
        save_query("test", "precise", result)

        history = get_query_history()
        assert isinstance(history[0]["result"], dict)
        assert history[0]["result"]["confidence"] == 0.9

    def test_reputation_updated_flag(self):
        save_query("test rep flag", "fast", self._make_result(), reputation_updated=1)

        history = get_query_history()
        assert len(history) == 1
        assert history[0]["query"] == "test rep flag"


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self):
        a = np.array([1.0, 0.1, 0.0])
        b = np.array([1.0, 0.2, 0.0])
        sim = _cosine_similarity(a, b)
        assert sim > 0.99


class TestNormalizeQuery:
    def test_strips_and_lowercases(self):
        assert _normalize_query("  Hello World!  ") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_query("  is  the  earth  flat  ") == "is the earth flat"

    def test_strips_punctuation(self):
        assert _normalize_query("Is the Earth flat?") == "is the earth flat"


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


class TestFindSimilarQueries:
    def _make_result(self, verdict="REAL"):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.verdict.value = verdict
        result.model_dump.return_value = {"verdict": verdict, "confidence": 0.9}
        return result

    def _make_embedding(self, seed: float = 1.0, dims: int = 8) -> list[float]:
        """Create a simple deterministic embedding for testing."""
        return [seed * (i + 1) / dims for i in range(dims)]

    def test_no_history_returns_empty(self):
        emb = self._make_embedding()
        results = find_similar_queries(emb, threshold=0.5)
        assert results == []

    def test_finds_identical_query(self):
        emb = self._make_embedding(seed=1.0)
        save_query("Is the earth round?", "precise", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, threshold=0.5)
        assert len(results) == 1
        assert results[0]["query"] == "Is the earth round?"
        assert results[0]["similarity"] == pytest.approx(1.0, abs=0.001)

    def test_similar_found_above_threshold(self):
        emb1 = self._make_embedding(seed=1.0)
        emb2 = self._make_embedding(seed=1.01)  # very similar
        save_query("Is the earth round?", "precise", self._make_result(), embedding=emb1)

        results = find_similar_queries(emb2, threshold=0.99)
        assert len(results) == 1

    def test_dissimilar_filtered_by_threshold(self):
        emb1 = self._make_embedding(seed=1.0)
        emb2 = [(-1) ** i * v for i, v in enumerate(emb1)]  # very different
        save_query("Is the earth round?", "precise", self._make_result(), embedding=emb1)

        results = find_similar_queries(emb2, threshold=0.95)
        assert len(results) == 0

    def test_mode_fast_returns_precise_and_fast(self):
        emb = self._make_embedding()
        save_query("claim A", "precise", self._make_result(), embedding=emb)
        save_query("claim B", "fast", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, mode="fast", threshold=0.5)
        modes = {r["mode"] for r in results}
        assert "precise" in modes
        assert "fast" in modes

    def test_mode_precise_excludes_fast(self):
        emb = self._make_embedding()
        save_query("claim A", "fast", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, mode="precise", threshold=0.5)
        assert len(results) == 0

    def test_mode_precise_includes_precise(self):
        emb = self._make_embedding()
        save_query("claim A", "precise", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, mode="precise", threshold=0.5)
        assert len(results) == 1

    def test_top_k_limits_results(self):
        emb = self._make_embedding()
        for i in range(5):
            save_query(f"claim {i}", "precise", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, top_k=2, threshold=0.5)
        assert len(results) == 2

    def test_deduplicates_by_normalized_query(self):
        emb = self._make_embedding()
        save_query("Is the earth round?", "precise", self._make_result("REAL"), embedding=emb)
        save_query("is the earth round", "precise", self._make_result("FAKE"), embedding=emb)

        results = find_similar_queries(emb, threshold=0.5)
        # Should return only the latest (FAKE) — dedup by normalized text
        assert len(results) == 1
        assert results[0]["result"]["verdict"] == "FAKE"

    def test_queries_without_embedding_ignored(self):
        emb = self._make_embedding()
        save_query("no embedding query", "precise", self._make_result())  # no embedding
        save_query("has embedding", "precise", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, threshold=0.5)
        assert len(results) == 1
        assert results[0]["query"] == "has embedding"

    def test_prefers_precise_over_fast_at_equal_similarity(self):
        emb = self._make_embedding()
        save_query("claim fast", "fast", self._make_result(), embedding=emb)
        save_query("claim precise", "precise", self._make_result(), embedding=emb)

        results = find_similar_queries(emb, mode="fast", top_k=2, threshold=0.5)
        assert results[0]["mode"] == "precise"

    def test_result_contains_verdict_and_confidence(self):
        """Returned result dict must contain the original FactCheckResult fields."""
        emb = self._make_embedding()
        save_query("climate change is real", "precise", self._make_result("REAL"), embedding=emb)

        results = find_similar_queries(emb, threshold=0.5)
        assert len(results) == 1
        r = results[0]["result"]
        assert r["verdict"] == "REAL"
        assert r["confidence"] == 0.9
        assert "similarity" in results[0]
        assert "created_at" in results[0]
        assert "id" in results[0]

    def test_results_sorted_by_similarity_descending(self):
        """When multiple matches exist, highest similarity must come first."""
        base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        close = [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # small angle from base
        farther = [0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 45-degree angle

        save_query("close claim", "precise", self._make_result(), embedding=close)
        save_query("farther claim", "precise", self._make_result(), embedding=farther)

        results = find_similar_queries(base, top_k=5, threshold=0.5)
        assert len(results) == 2
        assert results[0]["similarity"] >= results[1]["similarity"]
        assert results[0]["query"] == "close claim"

    def test_same_query_different_modes_dedup_keeps_latest(self):
        """Same normalized query in fast and precise — should dedup to the latest."""
        emb = self._make_embedding()
        save_query("earth is round", "fast", self._make_result("REAL"), embedding=emb)
        save_query("Earth is round!", "precise", self._make_result("FAKE"), embedding=emb)

        results = find_similar_queries(emb, mode="fast", threshold=0.5)
        # Dedup by normalized text → only latest (FAKE, precise) kept
        assert len(results) == 1
        assert results[0]["result"]["verdict"] == "FAKE"

    def test_embedding_stored_and_retrievable(self):
        """save_query stores embedding and it can be read back."""
        emb = [0.1, 0.2, 0.3, 0.4]
        save_query("test embed storage", "fast", self._make_result(), embedding=emb)

        import json as json_mod

        from verifyn.agent.db import QueryHistory, get_session

        with get_session() as session:
            record = session.query(QueryHistory).filter_by(query="test embed storage").first()
            assert record is not None
            assert record.embedding is not None
            stored = json_mod.loads(record.embedding)
            assert stored == emb

    def test_save_query_without_embedding(self):
        """save_query with no embedding stores None — not found by similarity search."""
        save_query("no embedding", "fast", self._make_result())

        from verifyn.agent.db import QueryHistory, get_session

        with get_session() as session:
            record = session.query(QueryHistory).filter_by(query="no embedding").first()
            assert record.embedding is None

    def test_mode_fast_finds_precise_only_when_no_fast_exists(self):
        """Fast mode should still return results even if only precise entries exist."""
        emb = self._make_embedding()
        save_query("only precise result", "precise", self._make_result("REAL"), embedding=emb)

        results = find_similar_queries(emb, mode="fast", threshold=0.5)
        assert len(results) == 1
        assert results[0]["mode"] == "precise"

    def test_default_threshold_used_when_not_specified(self):
        """When threshold is not passed, SIMILARITY_THRESHOLD env var is used."""
        emb = self._make_embedding(seed=1.0)
        save_query("test threshold", "precise", self._make_result(), embedding=emb)

        # Identical embedding → similarity=1.0, always above any threshold
        results = find_similar_queries(emb)
        assert len(results) == 1
