"""Claim detection inference via ONNX Runtime.

Public API:
    score_claim(text)             -> float in [0.0, 1.0]
    predict_claim(text)           -> dict with label + score
    is_claim_detector_available() -> bool
    reset_model()                 -> drop cached session (for tests)

Uses ONNX Runtime (~50 MB) instead of PyTorch (~2 GB). The ONNX model is
exported from the fine-tuned DeBERTa-v3-base checkpoint via export_onnx.ipynb.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("verifyn.claim_detector")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "claim_detector_onnx"
MAX_LENGTH = 256

_session = None
_tokenizer = None
_load_lock = threading.Lock()
_load_failed = False


def _load_model(model_dir: str | Path | None = None) -> bool:
    """Load ONNX session + tokenizer. Returns True on success."""
    global _session, _tokenizer, _load_failed

    if _session is not None:
        return True
    if _load_failed:
        return False

    with _load_lock:
        if _session is not None:
            return True
        if _load_failed:
            return False

        path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        onnx_path = path / "model.onnx"
        tokenizer_path = path / "tokenizer.json"

        if not onnx_path.exists():
            logger.warning("ONNX model not found at %s", onnx_path)
            _load_failed = True
            return False

        if not tokenizer_path.exists():
            logger.warning("Tokenizer not found at %s", tokenizer_path)
            _load_failed = True
            return False

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            logger.warning("Claim detector unavailable: %s", exc)
            _load_failed = True
            return False

        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if os.environ.get("CLAIM_DETECTOR_DEVICE") == "cuda"
                else ["CPUExecutionProvider"]
            )
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            tok = Tokenizer.from_file(str(tokenizer_path))
            tok.enable_truncation(max_length=MAX_LENGTH)
            tok.enable_padding(length=None)
        except Exception as exc:
            logger.warning("Failed to load claim detector from %s: %s", path, exc)
            _load_failed = True
            return False

        _session = sess
        _tokenizer = tok
        logger.info("Loaded ONNX claim detector from %s", path)
        return True


def is_claim_detector_available() -> bool:
    return _load_model()


def score_claim(text: str, *, model_dir: str | Path | None = None) -> float | None:
    """Return P(CLAIM) for text, or None if unavailable."""
    if not text or not text.strip():
        return 0.0

    if not _load_model(model_dir):
        return None

    encoded = _tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    logits = _session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    # Numerically stable softmax
    exp = np.exp(logits[0] - logits[0].max())
    probs = exp / exp.sum()
    return float(probs[1])


def predict_claim(
    text: str,
    *,
    threshold: float = 0.5,
    model_dir: str | Path | None = None,
) -> dict | None:
    score = score_claim(text, model_dir=model_dir)
    if score is None:
        return None
    return {
        "text": text,
        "score": round(score, 4),
        "label": "CLAIM" if score >= threshold else "NOT_CLAIM",
        "threshold": threshold,
    }


def reset_model() -> None:
    global _session, _tokenizer, _load_failed
    _session = None
    _tokenizer = None
    _load_failed = False
