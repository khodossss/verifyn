# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fever (soft scoring) |
| Total samples | 25 |
| Accuracy | 92.0% |
| Macro F1 | 0.923 |
| Weighted F1 | 0.925 |
| Original total | 32 |
| Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR) | 7 |
| Scored items | 25 |
| Original accuracy | 68.8% |
| Soft scoring rules | FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 1.00 | 0.82 | 0.90 | 11 |
| FAKE | 0.88 | 1.00 | 0.93 | 14 |
| **macro avg** | 0.94 | 0.91 | 0.92 | 25 |
| **weighted avg** | 0.93 | 0.92 | 0.92 | 25 |
| **accuracy** | | | 0.92 | 25 |

## Confusion Matrix

| True \ Pred | REAL | FAKE |
|---|---|---|
| **REAL** | **9** | 2 |
| **FAKE** | 0 | **14** |

## Error Analysis

Misclassified 2 out of 25 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| 157640 | Justin Chatwin starred in Doctor Who. | REAL | FAKE | 92% |
| 180726 | Victoria (Dance Exponents song) reached number 8 on the New Zealand singles char… | REAL | FAKE | 92% |
