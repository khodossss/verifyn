# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fakenewsnet_politifact (soft scoring) |
| Total samples | 19 |
| Accuracy | 84.2% |
| Macro F1 | 0.837 |
| Weighted F1 | 0.846 |
| Original total | 32 |
| Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR) | 13 |
| Scored items | 19 |
| Original accuracy | 40.6% |
| Soft scoring rules | FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.75 | 0.86 | 0.80 | 7 |
| FAKE | 0.91 | 0.83 | 0.87 | 12 |
| **macro avg** | 0.83 | 0.85 | 0.84 | 19 |
| **weighted avg** | 0.85 | 0.84 | 0.85 | 19 |
| **accuracy** | | | 0.84 | 19 |

## Confusion Matrix

| True \ Pred | REAL | FAKE |
|---|---|---|
| **REAL** | **6** | 1 |
| **FAKE** | 2 | **10** |

## Error Analysis

Misclassified 3 out of 19 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| politifact15327 | Delta targeted in online free airline ticket scam | FAKE | REAL | 92% |
| politifact11115 | Inquiry Sought in Hillary Clinton’s Use of Email | REAL | FAKE | 55% |
| politifact14233 | President Trump Underscores US-Jamaica Relations | FAKE | REAL | 90% |
