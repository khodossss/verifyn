# Claim Detector Pre-filter

## Overview

Before the ReAct agent runs (LLM calls, web searches, embedding lookups), a
fine-tuned **DeBERTa-v3-base** classifier checks whether the input contains a
verifiable factual claim. If the classifier score falls below a calibrated
threshold, the system returns `verdict=NO_CLAIMS` immediately, skipping the
entire agent pipeline.

```
Input text
    |
    v
 [Claim Detector]  ONNX Runtime, ~10ms, CPU
    |
    +-- score < 0.0005 --> return NO_CLAIMS (skip agent)
    |
    +-- score >= 0.0005 --> continue to 9-step ReAct agent
    |
    +-- classifier unavailable --> continue (graceful fallback)
```

Saves an LLM call per non-claim input.

## Model

| Property | Value |
|----------|-------|
| Architecture | DeBERTa-v3-base (184M params) |
| Inference format | ONNX (via `onnxruntime`, ~350 MB model file) |
| Runtime deps | `onnxruntime` (~50 MB) + `tokenizers` (~5 MB) |
| Labels | `0: NOT_CLAIM`, `1: CLAIM` |
| Training set | [Nithiwat/claim-detection](https://huggingface.co/datasets/Nithiwat/claim-detection) (23K train) |
| Test accuracy | 0.921 |
| Test F1 | 0.921 (macro) |
| Inference latency | ~10ms (ONNX/CPU), ~50ms (PyTorch/CPU) |


## Fine-tuning

Notebooks:

- `verifyn/claim_detector/notebooks/train.ipynb` (training)
- `verifyn/claim_detector/notebooks/threshold_tuning.ipynb` (threshold calibration)
- `verifyn/claim_detector/notebooks/export_onnx.ipynb` (ONNX export)

### Training setup

- Pure PyTorch training loop (no HuggingFace Trainer)
- `dtype=torch.float32` + `ignore_mismatched_sizes=True` (required for DeBERTa-v3 fp16 weights)
- AdamW optimizer, lr=2e-5, weight_decay=0.01
- Linear warmup (10% of total steps) + linear decay
- Gradient clipping (max_norm=1.0)
- Early stopping on validation F1 (patience=2)

### Dataset: Nithiwat/claim-detection

36K political debate sentences labeled for check-worthiness.

| Split | Total | CLAIM | NOT_CLAIM | Ratio |
|-------|------:|------:|----------:|------:|
| train | 23,276 | 11,872 | 11,404 | 51% |
| valid | 5,819 | 2,921 | 2,898 | 50% |
| test | 7,274 | 3,691 | 3,583 | 51% |

### Training results

```
Epoch 1/5 train_loss=0.2971 val_f1=0.9188 val_acc=0.9215 ** saved
Epoch 2/5 train_loss=0.1633 val_f1=0.9272 val_acc=0.9266 ** saved
Epoch 3/5 train_loss=0.0960 val_f1=0.9242 val_acc=0.9220 (patience 1/2)
Epoch 4/5 train_loss=0.0523 val_f1=0.9306 val_acc=0.9309 ** saved
Epoch 5/5 train_loss=0.0237 val_f1=0.9274 val_acc=0.9270 (patience 2/2)
Early stopping at epoch 5
```

Best model saved at epoch 4: val_f1=0.9306.

### Test set results

| Metric | Value |
|--------|------:|
| Accuracy | 0.921 |
| F1 | 0.921 |
| Precision | 0.939 |
| Recall | 0.903 |

### ONNX export verification

PyTorch vs ONNX Runtime: **4.0x speedup** (165ms vs 41ms per inference).

Scores match exactly (delta = 0.00000):

```
Text                                                PyTorch    ONNX
NASA confirmed water exists on Mars                  1.0000  1.0000
The unemployment rate dropped to 3.4%                1.0000  1.0000
I think this movie was really great                  0.0002  0.0002
Good morning everyone!                               0.0002  0.0002
asdfghjkl random words here                          0.0006  0.0006
Russia invaded Ukraine on February 24, 2022          0.9999  0.9999
```

## Threshold Tuning

The threshold determines the operating point of the pre-filter. Tuning
notebook: `verifyn/claim_detector/notebooks/threshold_tuning.ipynb`.

### Cost model

| Outcome | Cost |
|---------|------|
| Block a real claim (false negative) | **HIGH** -- silently dropped |
| Pass a non-claim to agent (false positive) | LOW -- agent wastes an LLM call |
| Block a non-claim (true negative) | Saves an LLM call |
| Pass a real claim (true positive) | Required |

### Chosen threshold

`CLAIM_SCORE_THRESHOLD = 0.0005`

The ONNX model produces near-binary scores (0.0002-0.0006 for non-claims,
0.9999-1.0000 for claims). Threshold candidates from the PR sweep:

| Target P | Threshold | Precision | Recall | F1 |
|----------|----------:|----------:|-------:|------:|
| 0.75 | 0.000549 | 0.750 | 0.984 | 0.851 |
| 0.78 | 0.000785 | 0.780 | 0.976 | 0.867 |
| 0.80 | 0.001007 | 0.800 | 0.972 | 0.878 |

At the chosen threshold 0.0005:

- Non-claims (score ~0.0002) are blocked
- Claims (score ~0.9999) all pass through
- CLAIM recall = 0.984 (only 1.6% of real claims silently dropped)

Configurable via `CLAIM_SCORE_THRESHOLD` in `.env`.

## Integration

The pre-filter runs in both `analyze_news()` and `analyze_news_stream()`
in `verifyn/agent/agent.py`, before any LLM or embedding call.

### Graceful fallback

If onnxruntime or model weights are missing, `_claim_prefilter` returns
`None` and the agent proceeds normally:

- Set `CLAIM_DETECTOR_ENABLED=false` in `.env` to disable
- Remove `models/claim_detector_onnx/` and the system works as before

### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `CLAIM_DETECTOR_ENABLED` | `true` | Enable/disable the pre-filter |
| `CLAIM_SCORE_THRESHOLD` | `0.0005` | Score below which input is NO_CLAIMS |
| `CLAIM_DETECTOR_DEVICE` | `cpu` | `cpu` or `cuda` (for onnxruntime-gpu) |

### Model weights

ONNX model lives in `models/claim_detector_onnx/` (gitignored, ~350 MB).
In Docker, mounted read-only: `./models/claim_detector_onnx:/app/models/claim_detector_onnx:ro`.

To train and export your own:
1. Run `verifyn/claim_detector/notebooks/all_colab.ipynb` in Google Colab
2. Download `claim_detector_onnx.zip` and extract to `models/claim_detector_onnx/`
3. Update `CLAIM_SCORE_THRESHOLD` in `.env` from `threshold_summary.json`

## Test Coverage

| Test file | Tests | What it covers |
|-----------|------:|----------------|
| `verifyn/claim_detector/tests/test_predict.py` | 10 | score_claim, predict_claim, ONNX session caching, soft fallback, reset |
| `verifyn/agent/tests/unit/test_agent_cycle.py::TestClaimPrefilter` | 3 | short-circuit in analyze_news, pass-through, streaming events |
