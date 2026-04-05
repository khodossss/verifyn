"""Unit tests for agent models and business logic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.models import (
    ConfidenceLevel,
    EvidenceItem,
    FactCheckResult,
    ManipulationType,
    Verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> FactCheckResult:
    defaults = dict(
        verdict=Verdict.FAKE,
        confidence=0.85,
        confidence_level=ConfidenceLevel.HIGH,
        manipulation_type=ManipulationType.FABRICATED,
        main_claims=["Claim A"],
        reasoning="Step 1 through 8 reasoning.",
        summary="This is fake.",
    )
    defaults.update(overrides)
    return FactCheckResult(**defaults)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_real(self):
        assert Verdict.REAL == "REAL"

    def test_fake(self):
        assert Verdict.FAKE == "FAKE"

    def test_partially_fake(self):
        assert Verdict.PARTIALLY_FAKE == "PARTIALLY_FAKE"

    def test_misleading(self):
        assert Verdict.MISLEADING == "MISLEADING"

    def test_unverifiable(self):
        assert Verdict.UNVERIFIABLE == "UNVERIFIABLE"

    def test_satire(self):
        assert Verdict.SATIRE == "SATIRE"

    def test_all_six_verdicts_exist(self):
        assert len(Verdict) == 6


# ---------------------------------------------------------------------------
# ManipulationType enum
# ---------------------------------------------------------------------------


class TestManipulationType:
    def test_none(self):
        assert ManipulationType.NONE == "NONE"

    def test_fabricated(self):
        assert ManipulationType.FABRICATED == "FABRICATED"

    def test_context_manipulation(self):
        assert ManipulationType.CONTEXT_MANIPULATION == "CONTEXT_MANIPULATION"

    def test_old_content_recycled(self):
        assert ManipulationType.OLD_CONTENT_RECYCLED == "OLD_CONTENT_RECYCLED"

    def test_misleading_headline(self):
        assert ManipulationType.MISLEADING_HEADLINE == "MISLEADING_HEADLINE"

    def test_partial_truth(self):
        assert ManipulationType.PARTIAL_TRUTH == "PARTIAL_TRUTH"

    def test_satire_misrepresented(self):
        assert ManipulationType.SATIRE_MISREPRESENTED == "SATIRE_MISREPRESENTED"

    def test_coordinated_disinfo(self):
        assert ManipulationType.COORDINATED_DISINFO == "COORDINATED_DISINFO"

    def test_impersonation(self):
        assert ManipulationType.IMPERSONATION == "IMPERSONATION"

    def test_all_nine_types_exist(self):
        assert len(ManipulationType) == 9


# ---------------------------------------------------------------------------
# ConfidenceLevel enum
# ---------------------------------------------------------------------------


class TestConfidenceLevel:
    def test_high(self):
        assert ConfidenceLevel.HIGH == "HIGH"

    def test_medium(self):
        assert ConfidenceLevel.MEDIUM == "MEDIUM"

    def test_low(self):
        assert ConfidenceLevel.LOW == "LOW"


# ---------------------------------------------------------------------------
# EvidenceItem model
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_supports_claim_true(self):
        item = EvidenceItem(source="reuters.com", summary="Confirms the event.", supports_claim=True)
        assert item.supports_claim is True

    def test_supports_claim_false(self):
        item = EvidenceItem(source="snopes.com", summary="Debunks the claim.", supports_claim=False)
        assert item.supports_claim is False

    def test_url_is_optional(self):
        item = EvidenceItem(source="source", summary="text", supports_claim=True)
        assert item.url is None

    def test_url_accepted(self):
        item = EvidenceItem(
            source="reuters.com",
            url="https://reuters.com/article",
            summary="text",
            supports_claim=True,
        )
        assert item.url == "https://reuters.com/article"

    def test_credibility_defaults_to_empty_string(self):
        item = EvidenceItem(source="s", summary="t", supports_claim=True)
        assert item.credibility == ""


# ---------------------------------------------------------------------------
# FactCheckResult model
# ---------------------------------------------------------------------------


class TestFactCheckResult:
    def test_valid_result_creates_successfully(self):
        result = _make_result()
        assert result.verdict == Verdict.FAKE

    def test_confidence_boundary_zero(self):
        result = _make_result(confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self):
        result = _make_result(confidence=1.0)
        assert result.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            _make_result(confidence=1.01)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            _make_result(confidence=-0.01)

    def test_main_claims_defaults_to_empty_list(self):
        result = _make_result(main_claims=[])
        assert result.main_claims == []

    def test_evidence_for_defaults_to_empty_list(self):
        result = _make_result()
        assert result.evidence_for == []

    def test_evidence_against_defaults_to_empty_list(self):
        result = _make_result()
        assert result.evidence_against == []

    def test_manipulation_type_defaults_to_none(self):
        result = FactCheckResult(
            verdict=Verdict.REAL,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            reasoning="all good",
            summary="looks real",
        )
        assert result.manipulation_type == ManipulationType.NONE

    def test_primary_source_is_optional(self):
        result = _make_result()
        assert result.primary_source is None

    def test_date_context_is_optional(self):
        result = _make_result()
        assert result.date_context is None

    def test_sources_checked_defaults_to_empty_list(self):
        result = _make_result()
        assert result.sources_checked == []


# ---------------------------------------------------------------------------
# confidence_level auto-derived from score (model_validator)
# ---------------------------------------------------------------------------


class TestConfidenceLevelFromScore:
    """The model_validator _enforce_confidence_level always overrides confidence_level."""

    def test_score_085_is_high(self):
        result = _make_result(confidence=0.85)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_score_09_is_high(self):
        result = _make_result(confidence=0.9)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_score_1_is_high(self):
        result = _make_result(confidence=1.0)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_score_05_is_medium(self):
        result = _make_result(confidence=0.5)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_score_084_is_medium(self):
        result = _make_result(confidence=0.84)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_score_049_is_low(self):
        result = _make_result(confidence=0.49)
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_score_0_is_low(self):
        result = _make_result(confidence=0.0)
        assert result.confidence_level == ConfidenceLevel.LOW
