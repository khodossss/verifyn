from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from verifyn.agent.constants import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD


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
    FABRICATED = "FABRICATED"
    CONTEXT_MANIPULATION = "CONTEXT_MANIPULATION"
    OLD_CONTENT_RECYCLED = "OLD_CONTENT_RECYCLED"
    MISLEADING_HEADLINE = "MISLEADING_HEADLINE"
    PARTIAL_TRUTH = "PARTIAL_TRUTH"
    SATIRE_MISREPRESENTED = "SATIRE_MISREPRESENTED"
    COORDINATED_DISINFO = "COORDINATED_DISINFO"
    IMPERSONATION = "IMPERSONATION"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EvidenceItem(BaseModel):
    source: str
    url: str | None = None
    summary: str
    supports_claim: bool
    credibility: str = ""


class FactCheckResult(BaseModel):
    verdict: Verdict = Field(description="Overall verdict for the news item")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict, 0.0-1.0")
    confidence_level: ConfidenceLevel | None = Field(default=None, description="Derived from confidence score")
    manipulation_type: ManipulationType = Field(
        default=ManipulationType.NONE, description="Type of manipulation if fake/misleading"
    )

    main_claims: list[str] = Field(default_factory=list, description="Verifiable claims extracted from the text")
    primary_source: str | None = Field(None, description="Original primary source (URL or description)")
    date_context: str | None = Field(None, description="Notes on publication date and temporal context")

    evidence_for: list[EvidenceItem] = Field(default_factory=list, description="Evidence supporting the claims")
    evidence_against: list[EvidenceItem] = Field(default_factory=list, description="Evidence refuting the claims")
    fact_checker_results: list[str] = Field(
        default_factory=list, description="Findings from professional fact-checkers"
    )
    sources_checked: list[str] = Field(default_factory=list, description="All URLs consulted during verification")

    reasoning: str = Field(default="", description="Step-by-step reasoning following the verification methodology")
    summary: str = Field(default="", description="Short plain-language verdict (2-4 sentences)")

    @model_validator(mode="after")
    def _enforce_confidence_level(self) -> "FactCheckResult":
        if self.confidence >= CONFIDENCE_HIGH_THRESHOLD:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW
        return self
