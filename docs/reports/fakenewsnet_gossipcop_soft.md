# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fakenewsnet_gossipcop (soft scoring) |
| Total samples | 16 |
| Accuracy | 68.8% |
| Macro F1 | 0.580 |
| Weighted F1 | 0.667 |
| Original total | 32 |
| Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR) | 16 |
| Scored items | 16 |
| Original accuracy | 34.4% |
| Soft scoring rules | FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.71 | 0.91 | 0.80 | 11 |
| FAKE | 0.50 | 0.20 | 0.29 | 5 |
| **macro avg** | 0.61 | 0.55 | 0.58 | 16 |
| **weighted avg** | 0.65 | 0.69 | 0.67 | 16 |
| **accuracy** | | | 0.69 | 16 |

## Confusion Matrix

| True \ Pred | REAL | FAKE |
|---|---|---|
| **REAL** | **10** | 1 |
| **FAKE** | 4 | **1** |

## Error Analysis

Misclassified 5 out of 16 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| gossipcop-4867710974 | Maks Chmerkovskiy Returns to Dancing with the Stars After Skipping Last Week Due… | FAKE | REAL | 92% |
| gossipcop-4294323822 | How the Weeknd Helped Selena Gomez Recover from Her Kidney Transplant: ‘One of H… | FAKE | REAL | 92% |
| gossipcop-7533724553 | Kim Kardashian Gets Her Stretch Marks Removed: 'It Didn't Hurt That Badly' | FAKE | REAL | 92% |
| gossipcop-1659503489 | Ciara and Russell Wilson Welcome Sienna Princess | FAKE | REAL | 92% |
| gossipcop-946100 | Bootleg Quaaludes, Crabs & Shooting Ashes From a Cannon: 7 Bizarre Revelations F… | REAL | FAKE | 62% |
