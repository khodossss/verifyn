# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | welfake |
| Total samples | 32 |
| Accuracy | 37.5% |
| Macro F1 | 0.414 |
| Weighted F1 | 0.414 |
| Timestamp | 2026-04-07 15:02 |
| Avg time/item | 162s |
| Wall time | 861s |
| Concurrency | 20 |
| Errors | 0 |
| Sample size | 32 |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.92 | 0.75 | 0.83 | 16 |
| FAKE | 0.00 | 0.00 | 0.00 | 16 |
| **macro avg** | 0.46 | 0.38 | 0.41 | 32 |
| **weighted avg** | 0.46 | 0.38 | 0.41 | 32 |
| **accuracy** | | | 0.38 | 32 |

## Confusion Matrix

| True \ Pred | REAL | FAKE | MISLEADING | UNVERIFIABLE | NO_CLAIMS |
|---|---|---|---|---|---|
| **REAL** | **12** | 0 | 1 | 2 | 1 |
| **FAKE** | 1 | **0** | 4 | 10 | 1 |
| **MISLEADING** | 0 | 0 | **0** | 0 | 0 |
| **UNVERIFIABLE** | 0 | 0 | 0 | **0** | 0 |
| **NO_CLAIMS** | 0 | 0 | 0 | 0 | **0** |

## Error Analysis

Misclassified 20 out of 32 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| Fake_869 | Apparently Donald Trump Has An Imaginary Xenophobic Friend Named Jim | FAKE | UNVERIFIABLE | 10% |
| Fake_6515 | WATCH: Fox News Host Flies Off The Handle At Republicans For Not Supporting Trum… | FAKE | UNVERIFIABLE | 10% |
| Fake_19309 | MEDIA ATTACKS TAYLOR SWIFT Over Refusal To Criticize President-Elect Trump | FAKE | UNVERIFIABLE | 10% |
| Fake_21295 | LIBERAL LUNACY: A Real Tom Turkey You’ll Get A Kick Out Of! | FAKE | NO_CLAIMS | 0% |
| Fake_13746 | TRUMP IS RIGHT! FOUR REASONS WHY The Judge In The Trump Univ. Case Should Recuse… | FAKE | MISLEADING | 60% |
| Fake_18390 | BRAINWASHED CHILDREN MOCK President Trump in Disturbing Washington Post VIDEO | FAKE | UNVERIFIABLE | 10% |
| Fake_17856 | CHICAGO DELI OWNER Says He Was Relieved To Hear Las Vegas Massacre Happened At C… | FAKE | REAL | 85% |
| True_30644 | Claims of votes by the dead, felons cloud North Carolina governor race | REAL | MISLEADING | 78% |
| Fake_7623 | Anti-Vaxxer Parents Let Toddler Die Of Meningitis After Maple Syrup Fails To Cur… | FAKE | UNVERIFIABLE | 10% |
| Fake_16559 | AMERICAN WORKERS IGNORE GAG ORDER: Speak Out On Being Replaced By Foreign Worker… | FAKE | MISLEADING | 55% |
| Fake_7223 | DOJ Tells Cops To Go Ahead And Start Stealing Money And TVs From The Public Agai… | FAKE | UNVERIFIABLE | 35% |
| Fake_19726 | LOL! HOW TO TRIGGER A LIBERAL On Halloween: “Make Your Costume The Most Tasteles… | FAKE | UNVERIFIABLE | 10% |
| Fake_9115 | “ENTITLED” DEM REP. SHEILA JACKSON LEE Has Been Taking Advantage Of Her “Public … | FAKE | UNVERIFIABLE | 45% |
| True_24521 | Highlights of Reuters interview with House Speaker Ryan | REAL | NO_CLAIMS | 0% |
| Fake_14719 | ANDREW BREITBART Warned Us The Occupy Wall Street Movement Would Morph Into An O… | FAKE | MISLEADING | 62% |
| Fake_22981 | MOCKINGBIRD MIRROR: Declassified Docs Depict Deeper Link Between the CIA and Ame… | FAKE | MISLEADING | 55% |
| True_41350 | Israel says Hezbollah runs Lebanese army, signaling both are foes | REAL | UNVERIFIABLE | 10% |
| Fake_212 | Trump Finally Delivers On Promised Phone Call To Soldier’s Wife, Says The Most D… | FAKE | UNVERIFIABLE | 10% |
| Fake_23462 | Eyewitness Says Feds Ambushed Bundys, 100 Shots Fired at Passengers, Lavoy Finic… | FAKE | UNVERIFIABLE | 10% |
| True_42829 | U.S. takes North Korea threat of H-bomb test seriously, Trump official says | REAL | UNVERIFIABLE | 10% |
