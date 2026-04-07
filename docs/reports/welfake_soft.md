# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | welfake (soft scoring) |
| Total samples | 18 |
| Accuracy | 88.9% |
| Macro F1 | 0.862 |
| Weighted F1 | 0.889 |
| Original total | 32 |
| Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR) | 14 |
| Scored items | 18 |
| Original accuracy | 37.5% |
| Soft scoring rules | FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.92 | 0.92 | 0.92 | 13 |
| FAKE | 0.80 | 0.80 | 0.80 | 5 |
| **macro avg** | 0.86 | 0.86 | 0.86 | 18 |
| **weighted avg** | 0.89 | 0.89 | 0.89 | 18 |
| **accuracy** | | | 0.89 | 18 |

## Confusion Matrix

| True \ Pred | REAL | FAKE |
|---|---|---|
| **REAL** | **12** | 1 |
| **FAKE** | 1 | **4** |

## Error Analysis

Misclassified 2 out of 18 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| Fake_17856 | CHICAGO DELI OWNER Says He Was Relieved To Hear Las Vegas Massacre Happened At C… | FAKE | REAL | 85% |
| True_30644 | Claims of votes by the dead, felons cloud North Carolina governor race | REAL | FAKE | 78% |
