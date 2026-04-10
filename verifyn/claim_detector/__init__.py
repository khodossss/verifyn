"""Fine-tuned DeBERTa classifier used as a cheap pre-filter for the agent."""

from .predict import (
    DEFAULT_MODEL_DIR,
    is_claim_detector_available,
    predict_claim,
    reset_model,
    score_claim,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "is_claim_detector_available",
    "predict_claim",
    "reset_model",
    "score_claim",
]
