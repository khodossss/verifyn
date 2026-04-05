from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Verdict(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    PARTIALLY_FAKE = "PARTIALLY_FAKE"
    MISLEADING = "MISLEADING"
    UNVERIFIABLE = "UNVERIFIABLE"
    SATIRE = "SATIRE"
    NO_CLAIMS = "NO_CLAIMS"


class ManipulationType(str, Enum):
    NONE = "NONE"
    FABRICATED = "FABRICATED"  # completely made up
    CONTEXT_MANIPULATION = "CONTEXT_MANIPULATION"  # real fact, wrong context
    OLD_CONTENT_RECYCLED = "OLD_CONTENT_RECYCLED"  # old news as new
    MISLEADING_HEADLINE = "MISLEADING_HEADLINE"  # headline ≠ body
    PARTIAL_TRUTH = "PARTIAL_TRUTH"  # key facts omitted
    SATIRE_MISREPRESENTED = "SATIRE_MISREPRESENTED"  # satire taken as fact
    COORDINATED_DISINFO = "COORDINATED_DISINFO"  # organised campaign
    IMPERSONATION = "IMPERSONATION"  # fake outlet / account


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"  # 0.8 – 1.0
    MEDIUM = "MEDIUM"  # 0.5 – 0.79
    LOW = "LOW"  # 0.0 – 0.49


class EvidenceItem(BaseModel):
    source: str
    url: Optional[str] = None
    summary: str
    supports_claim: bool  # True = supports original claim, False = refutes it
    credibility: str = ""  # e.g. "established media", "fact-checker", "social media"


class FactCheckResult(BaseModel):
    """Structured verdict returned by the fact-check agent."""

    verdict: Verdict = Field(description="Overall verdict for the news item")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict, 0.0–1.0")
    confidence_level: Optional[ConfidenceLevel] = Field(default=None, description="Human-readable confidence level")
    manipulation_type: ManipulationType = Field(
        default=ManipulationType.NONE, description="Type of manipulation if fake/misleading"
    )

    # Claim analysis
    main_claims: list[str] = Field(default_factory=list, description="Key verifiable claims extracted from the text")
    primary_source: Optional[str] = Field(None, description="Original primary source found (URL or description)")
    date_context: Optional[str] = Field(None, description="Notes on publication date and temporal context")

    # Evidence
    evidence_for: list[EvidenceItem] = Field(default_factory=list, description="Evidence supporting the claims")
    evidence_against: list[EvidenceItem] = Field(default_factory=list, description="Evidence refuting the claims")
    fact_checker_results: list[str] = Field(
        default_factory=list, description="What professional fact-checkers have found"
    )
    sources_checked: list[str] = Field(
        default_factory=list, description="All URLs/sources consulted during verification"
    )

    # Reasoning
    reasoning: str = Field(default="", description="Step-by-step reasoning following the 8-step fact-check methodology")
    summary: str = Field(default="", description="Short human-readable summary of the verdict (2–4 sentences)")

    @model_validator(mode="after")
    def _enforce_confidence_level(self) -> "FactCheckResult":
        """Always derive confidence_level from the numeric score."""
        if self.confidence >= 0.85:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW
        return self
