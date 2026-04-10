"""Shared fixtures for backend tests."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verifyn.agent.models import (
    ConfidenceLevel,
    EvidenceItem,
    FactCheckResult,
    ManipulationType,
    Verdict,
)


@pytest.fixture(scope="session")
def mock_result() -> FactCheckResult:
    return FactCheckResult(
        verdict=Verdict.FAKE,
        confidence=0.92,
        confidence_level=ConfidenceLevel.HIGH,
        manipulation_type=ManipulationType.FABRICATED,
        main_claims=["Claim A is false", "Statistic B is fabricated"],
        primary_source="https://reuters.com/original",
        date_context="Claimed today; original event from 2021",
        evidence_for=[
            EvidenceItem(
                source="example-blog.com",
                url="https://example-blog.com/story",
                summary="Repeats the claim without evidence.",
                supports_claim=True,
                credibility="low-credibility blog",
            )
        ],
        evidence_against=[
            EvidenceItem(
                source="reuters.com",
                url="https://reuters.com/fact-check",
                summary="Reuters found no evidence for the claim.",
                supports_claim=False,
                credibility="established media",
            )
        ],
        fact_checker_results=["Snopes: FALSE", "Reuters Fact Check: NOT VERIFIED"],
        sources_checked=["https://reuters.com/fact-check", "https://snopes.com/claim"],
        reasoning="Step 1: Extracted claims A and B.\nStep 2: Primary source not found.\n...",
        summary="This article is fabricated. No credible sources support the claims.",
    )


@pytest.fixture
def client(mock_result: FactCheckResult):
    with patch("backend.main.analyze_news", return_value=mock_result):
        # Import app lazily to ensure patches are in place before FastAPI init
        from backend.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_with_error():
    with patch("backend.main.analyze_news", side_effect=RuntimeError("LLM unavailable")):
        from backend.main import app

        with TestClient(app) as c:
            yield c
