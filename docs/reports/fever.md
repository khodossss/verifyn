# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fever |
| Total samples | 32 |
| Accuracy | 68.8% |
| Macro F1 | 0.803 |
| Weighted F1 | 0.803 |
| Timestamp | 2026-04-07 14:53 |
| Avg time/item | 129s |
| Wall time | 321s |
| Concurrency | 20 |
| Errors | 0 |
| Sample size | 32 |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 1.00 | 0.56 | 0.72 | 16 |
| FAKE | 0.93 | 0.81 | 0.87 | 16 |
| **macro avg** | 0.96 | 0.69 | 0.80 | 32 |
| **weighted avg** | 0.96 | 0.69 | 0.80 | 32 |
| **accuracy** | | | 0.69 | 32 |

## Confusion Matrix

| True \ Pred | REAL | FAKE | MISLEADING | PARTIALLY_FAKE | UNVERIFIABLE |
|---|---|---|---|---|---|
| **REAL** | **9** | 1 | 1 | 0 | 5 |
| **FAKE** | 0 | **13** | 0 | 1 | 2 |
| **MISLEADING** | 0 | 0 | **0** | 0 | 0 |
| **PARTIALLY_FAKE** | 0 | 0 | 0 | **0** | 0 |
| **UNVERIFIABLE** | 0 | 0 | 0 | 0 | **0** |

## Error Analysis

Misclassified 10 out of 32 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| 157640 | Justin Chatwin starred in Doctor Who. | REAL | MISLEADING | 92% |
| 128723 | Underdog features the voice talents of Amy Adams as a lead role. | REAL | UNVERIFIABLE | 10% |
| 226860 | Jenna Jameson worked as a glamor model. | REAL | UNVERIFIABLE | 10% |
| 180726 | Victoria (Dance Exponents song) reached number 8 on the New Zealand singles char… | REAL | FAKE | 92% |
| 216363 | The Chagatai language was a Turkic language spoken in India. | REAL | UNVERIFIABLE | 10% |
| 171631 | Dave Gibbons was born on April 12. | FAKE | UNVERIFIABLE | 10% |
| 203995 | Glee.com was released by Community Connect Inc.. | REAL | UNVERIFIABLE | 10% |
| 21187 | The Battle of France happened during World War II. | REAL | UNVERIFIABLE | 10% |
| 102445 | Aleister Crowley refused to be an occultist. | FAKE | UNVERIFIABLE | 10% |
| 51328 | Billboard Dad was directed by Alan Metter in 2001. | FAKE | PARTIALLY_FAKE | 72% |
