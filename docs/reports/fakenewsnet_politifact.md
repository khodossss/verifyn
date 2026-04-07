# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fakenewsnet_politifact |
| Total samples | 32 |
| Accuracy | 40.6% |
| Macro F1 | 0.555 |
| Weighted F1 | 0.555 |
| Timestamp | 2026-04-07 15:01 |
| Avg time/item | 164s |
| Wall time | 501s |
| Concurrency | 20 |
| Errors | 0 |
| Sample size | 32 |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.75 | 0.38 | 0.50 | 16 |
| FAKE | 1.00 | 0.44 | 0.61 | 16 |
| **macro avg** | 0.88 | 0.41 | 0.55 | 32 |
| **weighted avg** | 0.88 | 0.41 | 0.55 | 32 |
| **accuracy** | | | 0.41 | 32 |

## Confusion Matrix

| True \ Pred | REAL | FAKE | MISLEADING | PARTIALLY_FAKE | UNVERIFIABLE | NO_CLAIMS |
|---|---|---|---|---|---|---|
| **REAL** | **6** | 0 | 1 | 0 | 3 | 6 |
| **FAKE** | 2 | **7** | 2 | 1 | 1 | 3 |
| **MISLEADING** | 0 | 0 | **0** | 0 | 0 | 0 |
| **PARTIALLY_FAKE** | 0 | 0 | 0 | **0** | 0 | 0 |
| **UNVERIFIABLE** | 0 | 0 | 0 | 0 | **0** | 0 |
| **NO_CLAIMS** | 0 | 0 | 0 | 0 | 0 | **0** |

## Error Analysis

Misclassified 19 out of 32 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| politifact390 | The Democratic Debate in Cleveland | REAL | NO_CLAIMS | 0% |
| politifact13816 | Information for the Nation | FAKE | NO_CLAIMS | 0% |
| politifact15327 | Delta targeted in online free airline ticket scam | FAKE | REAL | 92% |
| politifact73 | Care Without Coverage: Too Little, Too Late | REAL | NO_CLAIMS | 0% |
| politifact14855 | Alabama Secretary of State | FAKE | NO_CLAIMS | 0% |
| politifact14601 | The Kardashians donate whopping sum to Las Vegas mass shooting victims | FAKE | UNVERIFIABLE | 10% |
| politifact15031 | Food Stamp Enrollment Drops by Four Million in One Month | FAKE | MISLEADING | 92% |
| politifact11115 | Inquiry Sought in Hillary Clinton’s Use of Email | REAL | MISLEADING | 55% |
| politifact14595 | Las Vegas Shooting Witnesses Report Multiple Gunmen Dressed as Security Guards O… | FAKE | MISLEADING | 92% |
| politifact3221 | Romney CPAC Speech: It's All About the Jobs | REAL | UNVERIFIABLE | 10% |
| politifact13823 | Man pardoned by Obama ‘executed’ by masked men at halfway house | FAKE | PARTIALLY_FAKE | 65% |
| politifact13260 | PolitiFact’s annotated transcript of the second presidential debate | REAL | UNVERIFIABLE | 10% |
| politifact14233 | President Trump Underscores US-Jamaica Relations | FAKE | REAL | 90% |
| politifact667 | U.S. Senate: U.S. Senate Roll Call Votes 110th Congress | REAL | NO_CLAIMS | 0% |
| politifact15301 | Account Suspended | FAKE | NO_CLAIMS | 0% |
| politifact3539 | U.S. Senator Debbie Stabenow of Michigan | REAL | UNVERIFIABLE | 10% |
| politifact3113 | National Income and Product Accounts Table 175 Relation of Gross Domestic Produc… | REAL | NO_CLAIMS | 0% |
| politifact20 | Rising to a New Generation of Global Challenges | REAL | NO_CLAIMS | 0% |
| politifact195 | Uniform Crime Reports | REAL | NO_CLAIMS | 0% |
