"""Tunable constants for the fact-check agent."""

from __future__ import annotations

import os

# Agent budget and control flow
MAX_SEARCH_BUDGET: int = int(os.environ.get("MAX_SEARCH_BUDGET", "5"))
MIN_CREDIBLE_SOURCES_FOR_AGREEMENT: int = 3
AGENT_RECURSION_LIMIT: int = 14
MAX_EXTRACT_ATTEMPTS: int = 3

# Confidence score -> ConfidenceLevel buckets
CONFIDENCE_HIGH_THRESHOLD: float = 0.85
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.50

# LLM defaults
LLM_DEFAULT_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 8192

# Maps the public API mode to the LLM reasoning_effort parameter
MODE_TO_REASONING_EFFORT: dict[str, str] = {
    "fast": "low",
    "precise": "medium",
}

# Claim detector pre-filter
CLAIM_SCORE_THRESHOLD: float = float(os.environ.get("CLAIM_SCORE_THRESHOLD", "0.0005"))

CLAIM_DETECTOR_ENABLED: bool = os.environ.get("CLAIM_DETECTOR_ENABLED", "true").lower() in ("1", "true", "yes")
