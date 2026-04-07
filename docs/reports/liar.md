# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | liar |
| Total samples | 31 |
| Accuracy | 29.0% |
| Macro F1 | 0.452 |
| Weighted F1 | 0.450 |
| Timestamp | 2026-04-07 15:01 |
| Avg time/item | 147s |
| Wall time | 803s |
| Concurrency | 20 |
| Errors | 1 |
| Sample size | 32 |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 1.00 | 0.33 | 0.50 | 15 |
| FAKE | 1.00 | 0.25 | 0.40 | 16 |
| **macro avg** | 1.00 | 0.29 | 0.45 | 31 |
| **weighted avg** | 1.00 | 0.29 | 0.45 | 31 |
| **accuracy** | | | 0.29 | 31 |

## Confusion Matrix

| True \ Pred | REAL | FAKE | MISLEADING | UNVERIFIABLE | NO_CLAIMS |
|---|---|---|---|---|---|
| **REAL** | **5** | 0 | 4 | 6 | 0 |
| **FAKE** | 0 | **4** | 5 | 6 | 1 |
| **MISLEADING** | 0 | 0 | **0** | 0 | 0 |
| **UNVERIFIABLE** | 0 | 0 | 0 | **0** | 0 |
| **NO_CLAIMS** | 0 | 0 | 0 | 0 | **0** |

## Error Analysis

Misclassified 22 out of 32 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| 1120.json | The Democrat-backed health care reform plan "will require (Americans) to subsidi… | FAKE | MISLEADING | 65% |
| 9507.json | NASA scientists fudged the numbers to make 1998 the hottest year to overstate th… | FAKE | MISLEADING | 86% |
| 13527.json | Theres been no conclusive or specific report to say Russia was trying to muddy t… | FAKE | UNVERIFIABLE | 10% |
| 5785.json | Says there is an upcoming vote to preserve benefits of Texas homestead exemption… | FAKE | MISLEADING | 88% |
| 2477.json | Over the past year ... our 16 counties have hemorrhaged more than 6,000 jobs wit… | FAKE | UNVERIFIABLE | 10% |
| 7744.json | Says states mandated tests come from an English company. | REAL | UNVERIFIABLE | 10% |
| 4714.json | I made a bunch of these promises during the campaign. ... Weve got about 60 perc… | FAKE | UNVERIFIABLE | 10% |
| 9610.json | David Perdue has never voted in a Republican primary until his name was on the b… | FAKE | UNVERIFIABLE | 10% |
| 9683.json | Im not a conspiracy theorist and I never allow conspiracy theorists on my progra… | FAKE | NO_CLAIMS | 0% |
| 11777.json | Today, if you were raised poor, youre just as likely to stay poor as you were 50… | REAL | MISLEADING | 55% |
| 2092.json | Sixty percent of the Hispanics support the Arizona immigration law | FAKE | UNVERIFIABLE | 10% |
| 11375.json | We have 650 people who move to Texas every day. | REAL | UNVERIFIABLE | 0% |
| 5848.json | Says the oil industry subsidies that President Barack Obama is attacking dont ex… | FAKE | MISLEADING | 92% |
| 8294.json | In Florida we have 75,000 on (a) waiting list for child care and 23,000 on waiti… | REAL | MISLEADING | 45% |
| 10055.json | In Massachusetts, Scott Brown pushed for a law to force women considering aborti… | REAL | UNVERIFIABLE | 10% |
| 12195.json | More people are struck by lightning than commit in-person voter fraud by imperso… | REAL | UNVERIFIABLE | 10% |
| 3158.json | Says his state budget will provide an increase in state funding for the 2011-12 … | REAL | UNVERIFIABLE | 10% |
| 6206.json | If you look at most of the polls, this is a margin-of-error race on Fourth of Ju… | REAL | UNVERIFIABLE | 10% |
| 12627.json | Says Donald Trumps proposed tax treatment of hedge fund managers makes the curre… | REAL | MISLEADING | 92% |
| 9578.json | When undocumented children are picked up at the border and told to appear later … | FAKE | MISLEADING | 65% |
| 9405.json | Charlie Crist allowed college tuition to increase up to 15 percent every year. | REAL | MISLEADING | 84% |
| 4250.json | Says state Rep. Sandy Pasch, her recall opponent, voted to allow public school e… | FAKE | UNVERIFIABLE | 10% |
