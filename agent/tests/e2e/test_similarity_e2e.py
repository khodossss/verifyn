"""E2E similarity search tests — real embeddings, real DB.

Tests the full similarity pipeline:
1. First query → agent does full research → saves with embedding
2. Identical query → agent finds it in history
3. Paraphrased query → agent finds similar in history
4. Unrelated query → no match in history

These tests MUST run in order (each builds on previous state).
Use pytest-ordering or run the file directly.

Run:
    pytest agent/tests/e2e/test_similarity_e2e.py --run-e2e -v

Requires: OPENAI_API_KEY
Cost: ~$0.30-0.50 total (4 agent runs + embeddings)
"""

from __future__ import annotations

import os
import tempfile

import pytest
from dotenv import load_dotenv

load_dotenv()

_E2E_DB_FILE = os.path.join(tempfile.gettempdir(), "verifyn_e2e_similarity.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_E2E_DB_FILE}"

from agent.db import Base, _get_engine, find_similar_queries, get_query_history
from agent.models import FactCheckResult

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Module-scoped DB — persists across all tests in this file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def e2e_db():
    """Single DB for the entire module — tests build on each other."""
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


# ---------------------------------------------------------------------------
# Shared state across ordered tests
# ---------------------------------------------------------------------------

_first_result: FactCheckResult | None = None
_first_embedding: list[float] | None = None

ORIGINAL_CLAIM = "NASA confirmed that the James Webb Space Telescope discovered signs of life on an exoplanet in 2024."


# ---------------------------------------------------------------------------
# Tests (ordered — each depends on previous)
# ---------------------------------------------------------------------------


class TestSimilarityPipeline:
    """Ordered tests that verify the full similarity search lifecycle."""

    def test_01_first_query_saves_with_embedding(self):
        """First time checking a claim — full agent run, result saved with embedding."""
        global _first_result

        from agent import analyze_news

        result = analyze_news(ORIGINAL_CLAIM, reasoning_effort="low")

        assert isinstance(result, FactCheckResult)
        assert result.verdict is not None
        assert result.summary

        _first_result = result

        # Verify saved to DB with embedding
        history = get_query_history(limit=5)
        assert len(history) >= 1
        assert history[0]["query"] == ORIGINAL_CLAIM

        # Verify embedding exists
        from agent.db import QueryHistory, get_session

        with get_session() as session:
            record = session.query(QueryHistory).order_by(QueryHistory.id.desc()).first()
            assert record is not None
            assert record.embedding is not None, "Embedding should be saved with query"

    def test_02_identical_query_found_in_similarity_search(self):
        """Same exact text → embedding search should find the previous result."""
        from agent.db import compute_embedding

        embedding = compute_embedding(ORIGINAL_CLAIM)
        matches = find_similar_queries(embedding, mode="fast", threshold=0.85)

        assert len(matches) >= 1, "Identical query should be found in history"
        assert matches[0]["similarity"] >= 0.95, (
            f"Identical query similarity should be >= 0.95, got {matches[0]['similarity']}"
        )
        assert matches[0]["query"] == ORIGINAL_CLAIM

    def test_03_paraphrased_query_found_in_similarity_search(self):
        """Semantically similar but differently worded → should still match."""
        from agent.db import compute_embedding

        paraphrased = (
            "Has NASA's James Webb telescope found evidence of extraterrestrial life "
            "on a planet outside our solar system?"
        )
        embedding = compute_embedding(paraphrased)
        matches = find_similar_queries(embedding, mode="fast", threshold=0.65)

        assert len(matches) >= 1, (
            f"Paraphrased query should find similar result. Checked: '{paraphrased}' vs '{ORIGINAL_CLAIM}'"
        )
        assert matches[0]["similarity"] >= 0.65

    def test_04_unrelated_query_not_found(self):
        """Completely different topic → should NOT match."""
        from agent.db import compute_embedding

        unrelated = "What is the current price of Bitcoin in US dollars today?"
        embedding = compute_embedding(unrelated)
        matches = find_similar_queries(embedding, mode="fast", threshold=0.85)

        assert len(matches) == 0, f"Unrelated query should not match. Got: {[m['query'] for m in matches]}"

    def test_05_second_query_saved_increases_history(self):
        """Running a second different claim should add to history."""
        from agent import analyze_news

        result = analyze_news(
            "The World Health Organization declared the COVID-19 pandemic officially over in May 2023.",
            reasoning_effort="low",
        )
        assert isinstance(result, FactCheckResult)

        history = get_query_history(limit=10)
        assert len(history) >= 2, "Should have at least 2 queries in history now"

    def test_06_mode_filtering_works(self):
        """Fast-mode results should be findable in fast mode but not in precise mode
        (unless precise results also exist)."""
        from agent.db import compute_embedding

        embedding = compute_embedding(ORIGINAL_CLAIM)

        # Fast mode should find results (we ran with reasoning_effort=low → fast mode)
        fast_matches = find_similar_queries(embedding, mode="fast", threshold=0.85)
        assert len(fast_matches) >= 1

        # Precise mode should NOT find fast-mode results
        precise_matches = find_similar_queries(embedding, mode="precise", threshold=0.85)
        # All our queries were in fast mode, so precise should find nothing
        fast_only = [m for m in precise_matches if m["mode"] == "fast"]
        assert len(fast_only) == 0, "Precise mode should not return fast-mode results"
