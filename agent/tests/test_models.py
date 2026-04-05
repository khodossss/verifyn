"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from agent.models import (
    ConfidenceLevel,
    EvidenceItem,
    FactCheckResult,
    ManipulationType,
    Verdict,
)


def _minimal() -> dict:
    """Minimal valid FactCheckResult payload."""
    return {
        "verdict": "REAL",
        "confidence": 0.9,
        "confidence_level": "HIGH",
        "manipulation_type": "NONE",
        "main_claims": ["The sky is blue"],
        "reasoning": "Step-by-step analysis.",
        "summary": "The claim is accurate.",
    }


class TestVerdictEnum:
    def test_all_values_exist(self):
        assert {v.value for v in Verdict} == {"REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"}

    def test_invalid_verdict_raises(self):
        data = _minimal()
        data["verdict"] = "WRONG"
        with pytest.raises(ValidationError):
            FactCheckResult(**data)


class TestConfidenceValidation:
    def test_valid_boundaries(self):
        for val in (0.0, 0.5, 1.0):
            r = FactCheckResult(**{**_minimal(), "confidence": val})
            assert r.confidence == val

    def test_above_one_raises(self):
        with pytest.raises(ValidationError):
            FactCheckResult(**{**_minimal(), "confidence": 1.1})

    def test_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FactCheckResult(**{**_minimal(), "confidence": -0.1})


class TestConfidenceLevelEnum:
    def test_high_medium_low_exist(self):
        assert {c.value for c in ConfidenceLevel} == {"HIGH", "MEDIUM", "LOW"}

    def test_invalid_raises(self):
        with pytest.raises(ValidationError):
            FactCheckResult(**{**_minimal(), "confidence_level": "VERY_HIGH"})


class TestManipulationType:
    def test_none_is_default(self):
        data = _minimal()
        data.pop("manipulation_type", None)
        r = FactCheckResult(**data)
        assert r.manipulation_type == ManipulationType.NONE

    def test_all_types_valid(self):
        for mt in ManipulationType:
            r = FactCheckResult(**{**_minimal(), "manipulation_type": mt.value})
            assert r.manipulation_type == mt


class TestEvidenceItem:
    def test_minimal_evidence(self):
        ev = EvidenceItem(source="BBC", summary="Confirmed event", supports_claim=True)
        assert ev.url is None
        assert ev.credibility == ""

    def test_full_evidence(self):
        ev = EvidenceItem(
            source="Reuters",
            url="https://reuters.com/article",
            summary="Detailed summary",
            supports_claim=False,
            credibility="established media",
        )
        assert ev.supports_claim is False
        assert "reuters" in ev.url


class TestFactCheckResult:
    def test_minimal_valid(self):
        r = FactCheckResult(**_minimal())
        assert r.verdict == Verdict.REAL
        assert r.evidence_for == []
        assert r.evidence_against == []
        assert r.sources_checked == []

    def test_full_result(self):
        ev = EvidenceItem(source="AP", summary="AP confirms", supports_claim=True)
        r = FactCheckResult(
            **{
                **_minimal(),
                "evidence_for": [ev],
                "evidence_against": [],
                "fact_checker_results": ["Snopes: True"],
                "sources_checked": ["https://ap.org"],
                "primary_source": "https://ap.org",
                "date_context": "Published 2024-01-01",
            }
        )
        assert len(r.evidence_for) == 1
        assert r.sources_checked == ["https://ap.org"]

    def test_confidence_level_enforced_from_score(self):
        r = FactCheckResult(**{**_minimal(), "confidence": 0.85})
        assert r.confidence_level == ConfidenceLevel.HIGH

        r = FactCheckResult(**{**_minimal(), "confidence": 0.6})
        assert r.confidence_level == ConfidenceLevel.MEDIUM

        r = FactCheckResult(**{**_minimal(), "confidence": 0.3})
        assert r.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_overrides_wrong_input(self):
        r = FactCheckResult(**{**_minimal(), "confidence": 0.3, "confidence_level": "HIGH"})
        assert r.confidence_level == ConfidenceLevel.LOW

    def test_serialisation_round_trip(self):
        r = FactCheckResult(**_minimal())
        dumped = r.model_dump(mode="json")
        restored = FactCheckResult.model_validate(dumped)
        assert restored.verdict == r.verdict
        assert restored.confidence == r.confidence
