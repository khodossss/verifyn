"""E2E verdict tests — real LLM, real search, real DB.

Verifies the agent produces correct verdicts for known claims.
Each test checks: verdict, confidence range, evidence presence, and key fields.

Run:
    pytest agent/tests/e2e/test_verdicts.py --run-e2e -v

Requires: OPENAI_API_KEY (or other LLM_PROVIDER keys in .env)
Cost: ~$0.05-0.15 per test
"""

from __future__ import annotations

import os
import tempfile

import pytest
from dotenv import load_dotenv

load_dotenv()

# Use a file-based temp DB for e2e tests (isolated from production)
_E2E_DB_FILE = os.path.join(tempfile.gettempdir(), "verifyn_e2e_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_E2E_DB_FILE}"

from agent.db import Base, _get_engine
from agent.models import FactCheckResult, Verdict

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def e2e_db():
    """Create a fresh DB for the entire e2e test module."""
    import agent.db as db_mod

    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None
    os.environ["DATABASE_URL"] = f"sqlite:///{_E2E_DB_FILE}"
    db_mod.DATABASE_URL = f"sqlite:///{_E2E_DB_FILE}"

    if os.path.exists(_E2E_DB_FILE):
        os.remove(_E2E_DB_FILE)
    engine = _get_engine()
    Base.metadata.create_all(engine)
    yield
    if db_mod._engine is not None:
        db_mod._engine.dispose()
    db_mod._engine = None
    db_mod._SessionFactory = None
    if os.path.exists(_E2E_DB_FILE):
        try:
            os.remove(_E2E_DB_FILE)
        except OSError:
            pass


def _check_result_structure(result: FactCheckResult, *, allow_empty_claims: bool = False):
    """Common assertions for any FactCheckResult."""
    assert isinstance(result, FactCheckResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_level is not None
    assert result.reasoning, "reasoning must not be empty"
    # NO_CLAIMS verdict may have empty summary/claims — that's valid
    if result.verdict != Verdict.NO_CLAIMS:
        assert result.summary, "summary must not be empty"
        if not allow_empty_claims:
            assert result.main_claims, "main_claims must not be empty"


# ---------------------------------------------------------------------------
# Verdict correctness
# ---------------------------------------------------------------------------


class TestRealNews:
    """Claims that are factually true — agent should return REAL."""

    def test_established_fact(self):
        """Well-known, easily verifiable fact."""
        from agent import analyze_news

        result = analyze_news(
            "The Earth orbits the Sun and completes one revolution approximately every 365.25 days.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict == Verdict.REAL
        assert result.confidence >= 0.7
        assert len(result.evidence_for) >= 1

    def test_recent_verifiable_event(self):
        """A widely reported real event."""
        from agent import analyze_news

        result = analyze_news(
            "The 2024 Summer Olympics were held in Paris, France.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict == Verdict.REAL
        assert result.confidence >= 0.7


class TestFakeNews:
    """Claims that are fabricated — agent should return FAKE."""

    def test_classic_debunked_claim(self):
        """Well-known debunked conspiracy theory."""
        from agent import analyze_news

        result = analyze_news(
            "NASA confirmed that drinking bleach cures COVID-19 and recommends it as official treatment.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict in (Verdict.FAKE, Verdict.MISLEADING)
        assert result.confidence >= 0.7
        assert len(result.evidence_against) >= 1

    def test_fabricated_statistic(self):
        """Completely made up statistic from a real-sounding source."""
        from agent import analyze_news

        result = analyze_news(
            "According to WHO's 2025 annual report, 78% of all cancer cases worldwide are caused by 5G radiation.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict == Verdict.FAKE
        assert result.confidence >= 0.6


class TestMisleadingNews:
    """Claims based on real facts but distorted — agent should return MISLEADING."""

    def test_out_of_context_stat(self):
        """Real data presented without crucial context."""
        from agent import analyze_news

        result = analyze_news(
            "EXPOSED: Government data shows vaccinated people are dying at HIGHER rates than unvaccinated! "
            "Official UK statistics prove vaccines are killing people!",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict in (Verdict.MISLEADING, Verdict.FAKE, Verdict.PARTIALLY_FAKE)
        assert result.confidence >= 0.5


class TestSatire:
    """Satirical content — agent should return SATIRE."""

    def test_obvious_satire(self):
        from agent import analyze_news

        result = analyze_news(
            "The Onion reports: Scientists discover that the Moon is actually made of cheese. "
            "NASA plans to send astronauts to collect samples for a giant fondue.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict in (Verdict.SATIRE, Verdict.FAKE)


class TestNoClaims:
    """Inputs without verifiable claims — agent should return NO_CLAIMS."""

    def test_gibberish(self):
        from agent import analyze_news

        result = analyze_news(
            "asdfghjkl qwerty zxcvbnm 12345 random words without meaning",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict == Verdict.NO_CLAIMS

    def test_opinion_only(self):
        from agent import analyze_news

        result = analyze_news(
            "I think chocolate ice cream is the best flavor in the world and nobody can change my mind.",
            reasoning_effort="low",
        )
        _check_result_structure(result)
        assert result.verdict in (Verdict.NO_CLAIMS, Verdict.UNVERIFIABLE)
