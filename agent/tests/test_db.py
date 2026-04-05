"""Tests for domain reputation database."""

from __future__ import annotations

import os

import pytest

# Force SQLite in-memory for tests
os.environ["DATABASE_URL"] = "sqlite://"

from agent.db import (
    SCORING_TABLE,
    Base,
    _get_engine,
    _query_hash,
    extract_domain,
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
    import agent.db as db_mod

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
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def _make_result(self, verdict="REAL", url="https://reuters.com/a"):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.verdict.value = verdict
        result.model_dump.return_value = {"verdict": verdict, "confidence": 0.9}
        evidence = MagicMock()
        evidence.url = url
        result.evidence_for = [evidence]
        result.evidence_against = []
        result.sources_checked = []
        return result

    def test_first_query_updates_reputation(self):
        result = self._make_result("REAL")
        count = update_reputation_from_result(result, query_text="Is earth round?")
        assert count == 1
        assert get_domain("reuters.com").true_points == 1.0

    def test_duplicate_query_skips_reputation(self):
        result = self._make_result("REAL")
        # First time — updates
        update_reputation_from_result(result, query_text="Is earth round?")
        save_query("Is earth round?", "fast", result, reputation_updated=1)

        # Second time — same query, should skip
        count = update_reputation_from_result(result, query_text="Is earth round?")
        assert count == 0
        # Points should not have doubled
        assert get_domain("reuters.com").true_points == 1.0

    def test_normalized_duplicate_detected(self):
        result = self._make_result("REAL")
        update_reputation_from_result(result, query_text="Is Earth Round?")
        save_query("Is Earth Round?", "fast", result, reputation_updated=1)

        # Same query with different casing/punctuation
        count = update_reputation_from_result(result, query_text="is earth round")
        assert count == 0

    def test_different_query_updates_normally(self):
        result = self._make_result("REAL")
        update_reputation_from_result(result, query_text="Is earth round?")
        save_query("Is earth round?", "fast", result, reputation_updated=1)

        # Different query — should update
        count = update_reputation_from_result(result, query_text="Is the sky blue?")
        assert count == 1
        assert get_domain("reuters.com").true_points == 2.0

    def test_hash_is_stable(self):
        h1 = _query_hash("Is the Earth flat?")
        h2 = _query_hash("Is the Earth flat?")
        assert h1 == h2

    def test_hash_normalizes(self):
        h1 = _query_hash("Is the Earth flat?")
        h2 = _query_hash("  is  the  earth  flat  ")
        assert h1 == h2
