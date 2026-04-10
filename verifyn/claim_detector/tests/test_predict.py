"""Tests for the claim-detector ONNX inference module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from verifyn.claim_detector import predict as predict_mod
from verifyn.claim_detector import predict_claim, reset_model, score_claim


@pytest.fixture(autouse=True)
def _clean_model_cache():
    reset_model()
    yield
    reset_model()


def _install_fake_session(claim_score: float) -> None:
    """Inject a fake ONNX session that returns a fixed score."""

    class FakeEncoding:
        def __init__(self):
            self.ids = [1, 2, 3]
            self.attention_mask = [1, 1, 1]

    class FakeTokenizer:
        def encode(self, text):
            return FakeEncoding()

        def enable_truncation(self, **kwargs):
            pass

        def enable_padding(self, **kwargs):
            pass

    logit_0 = 0.0
    logit_1 = float(np.log(claim_score / (1 - claim_score + 1e-12)))

    class FakeSession:
        def run(self, output_names, inputs):
            return [np.array([[logit_0, logit_1]], dtype=np.float32)]

    predict_mod._session = FakeSession()
    predict_mod._tokenizer = FakeTokenizer()
    predict_mod._load_failed = False


class TestScoreClaim:
    def test_returns_zero_on_empty_text(self):
        assert score_claim("") == 0.0
        assert score_claim("    ") == 0.0

    def test_returns_high_score_for_claim(self):
        _install_fake_session(claim_score=0.95)
        result = score_claim("NASA confirmed water on Mars in 2025")
        assert result is not None
        assert 0.93 <= result <= 0.97

    def test_returns_low_score_for_non_claim(self):
        _install_fake_session(claim_score=0.05)
        result = score_claim("hello, how are you doing?")
        assert result is not None
        assert 0.03 <= result <= 0.07

    def test_returns_none_when_session_fails_to_load(self):
        with patch.object(predict_mod, "_load_model", return_value=False):
            assert score_claim("NASA confirmed water on Mars") is None

    def test_caches_session_after_first_call(self):
        _install_fake_session(claim_score=0.7)
        score_claim("first call")
        score_claim("second call")
        assert predict_mod._session is not None


class TestPredictClaim:
    def test_returns_label_above_threshold(self):
        _install_fake_session(claim_score=0.92)
        out = predict_claim("Russia invaded Ukraine in 2022", threshold=0.5)
        assert out is not None
        assert out["label"] == "CLAIM"
        assert out["score"] > 0.9

    def test_returns_label_below_threshold(self):
        _install_fake_session(claim_score=0.10)
        out = predict_claim("good morning everyone", threshold=0.5)
        assert out is not None
        assert out["label"] == "NOT_CLAIM"

    def test_custom_threshold(self):
        _install_fake_session(claim_score=0.45)
        out = predict_claim("borderline text", threshold=0.7)
        assert out is not None
        assert out["label"] == "NOT_CLAIM"

    def test_returns_none_when_unavailable(self):
        with patch.object(predict_mod, "_load_model", return_value=False):
            assert predict_claim("any text") is None


class TestResetModel:
    def test_clears_global_state(self):
        _install_fake_session(claim_score=0.5)
        assert predict_mod._session is not None
        reset_model()
        assert predict_mod._session is None
        assert predict_mod._tokenizer is None
        assert predict_mod._load_failed is False
