# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | liar (soft scoring) |
| Total samples | 18 |
| Accuracy | 77.8% |
| Macro F1 | 0.811 |
| Weighted F1 | 0.811 |
| Original total | 32 |
| Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR) | 14 |
| Scored items | 18 |
| Original accuracy | 28.1% |
| Soft scoring rules | FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 1.00 | 0.56 | 0.71 | 9 |
| FAKE | 0.69 | 1.00 | 0.82 | 9 |
| **macro avg** | 0.85 | 0.78 | 0.81 | 18 |
| **weighted avg** | 0.85 | 0.78 | 0.81 | 18 |
| **accuracy** | | | 0.78 | 18 |

## Confusion Matrix

| True \ Pred | REAL | FAKE |
|---|---|---|
| **REAL** | **5** | 4 |
| **FAKE** | 0 | **9** |

## Error Analysis

Misclassified 4 out of 18 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| 11777.json | Today, if you were raised poor, youre just as likely to stay poor as you were 50… | REAL | FAKE | 55% |
| 8294.json | In Florida we have 75,000 on (a) waiting list for child care and 23,000 on waiti… | REAL | FAKE | 45% |
| 12627.json | Says Donald Trumps proposed tax treatment of hedge fund managers makes the curre… | REAL | FAKE | 92% |
| 9405.json | Charlie Crist allowed college tuition to increase up to 15 percent every year. | REAL | FAKE | 84% |
