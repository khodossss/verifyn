"""Tests for the FastAPI backend endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

VALID_TEXT = "Breaking: Scientists discover that eating chocolate cures all diseases according to a new study."

# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_root_returns_service_info(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "Fake News Detection API"
    assert "endpoints" in data
    assert "POST /analyze" in data["endpoints"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /analyze — success paths
# ---------------------------------------------------------------------------


def test_analyze_returns_200_with_valid_text(client: TestClient):
    resp = client.post("/analyze", json={"text": VALID_TEXT})
    assert resp.status_code == 200


def test_analyze_response_contains_verdict(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert data["verdict"] == "FAKE"


def test_analyze_response_contains_confidence(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert data["confidence"] == 0.92
    assert data["confidence_level"] == "HIGH"


def test_analyze_response_contains_manipulation_type(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert data["manipulation_type"] == "FABRICATED"


def test_analyze_response_has_all_required_fields(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    required_fields = [
        "verdict",
        "confidence",
        "confidence_level",
        "manipulation_type",
        "main_claims",
        "evidence_for",
        "evidence_against",
        "fact_checker_results",
        "sources_checked",
        "reasoning",
        "summary",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_analyze_evidence_for_structure(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert len(data["evidence_for"]) == 1
    item = data["evidence_for"][0]
    assert "source" in item
    assert "summary" in item
    assert "supports_claim" in item
    assert item["supports_claim"] is True


def test_analyze_evidence_against_structure(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert len(data["evidence_against"]) == 1
    item = data["evidence_against"][0]
    assert item["supports_claim"] is False
    assert item["credibility"] == "established media"


def test_analyze_sources_checked_is_list(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert isinstance(data["sources_checked"], list)
    assert len(data["sources_checked"]) == 2


def test_analyze_fact_checker_results_is_list(client: TestClient):
    data = client.post("/analyze", json={"text": VALID_TEXT}).json()
    assert isinstance(data["fact_checker_results"], list)


def test_analyze_verbose_flag_accepted(client: TestClient):
    resp = client.post("/analyze", json={"text": VALID_TEXT, "verbose": True})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /analyze — validation errors
# ---------------------------------------------------------------------------


def test_analyze_rejects_missing_text(client: TestClient):
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422


def test_analyze_rejects_text_too_short(client: TestClient):
    resp = client.post("/analyze", json={"text": "short"})
    assert resp.status_code == 422


def test_analyze_rejects_whitespace_only_text(client: TestClient):
    resp = client.post("/analyze", json={"text": "          "})
    assert resp.status_code == 422


def test_analyze_rejects_empty_string(client: TestClient):
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422


def test_analyze_rejects_non_string_text(client: TestClient):
    resp = client.post("/analyze", json={"text": 12345})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze — error handling
# ---------------------------------------------------------------------------


def test_analyze_returns_500_when_agent_raises(client_with_error: TestClient):
    resp = client_with_error.post("/analyze", json={"text": VALID_TEXT})
    assert resp.status_code == 500


def test_analyze_500_includes_error_detail(client_with_error: TestClient):
    resp = client_with_error.post("/analyze", json={"text": VALID_TEXT})
    data = resp.json()
    assert "detail" in data
    assert "LLM unavailable" in data["detail"]
