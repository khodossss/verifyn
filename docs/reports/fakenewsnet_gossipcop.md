# Verifyn Evaluation Report

## Overview

| Metric | Value |
|--------|-------|
| Dataset | fakenewsnet_gossipcop |
| Total samples | 32 |
| Accuracy | 34.4% |
| Macro F1 | 0.491 |
| Weighted F1 | 0.491 |
| Timestamp | 2026-04-07 14:59 |
| Avg time/item | 151s |
| Wall time | 380s |
| Concurrency | 20 |
| Errors | 0 |
| Sample size | 32 |

## Classification Report

| Verdict | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| REAL | 0.71 | 0.62 | 0.67 | 16 |
| FAKE | 1.00 | 0.06 | 0.12 | 16 |
| **macro avg** | 0.86 | 0.34 | 0.49 | 32 |
| **weighted avg** | 0.86 | 0.34 | 0.49 | 32 |
| **accuracy** | | | 0.34 | 32 |

## Confusion Matrix

| True \ Pred | REAL | FAKE | MISLEADING | UNVERIFIABLE | NO_CLAIMS |
|---|---|---|---|---|---|
| **REAL** | **10** | 0 | 1 | 3 | 2 |
| **FAKE** | 4 | **1** | 0 | 8 | 3 |
| **MISLEADING** | 0 | 0 | **0** | 0 | 0 |
| **UNVERIFIABLE** | 0 | 0 | 0 | **0** | 0 |
| **NO_CLAIMS** | 0 | 0 | 0 | 0 | **0** |

## Error Analysis

Misclassified 21 out of 32 items.

| ID | Text | Expected | Predicted | Confidence |
|----|------|----------|-----------|------------|
| gossipcop-4492802133 | Did Miley Cyrus get mad at Liam Hemsworth for refusing to wear his promise ring? | FAKE | NO_CLAIMS | 0% |
| gossipcop-4367523155 | Outlet Flip Flops On What Camilla Parker Bowles Thinks Of Meghan Markle | FAKE | UNVERIFIABLE | 25% |
| gossipcop-4867710974 | Maks Chmerkovskiy Returns to Dancing with the Stars After Skipping Last Week Due… | FAKE | REAL | 92% |
| gossipcop-872847 | Define Dog days at Dictionary.com | REAL | UNVERIFIABLE | 10% |
| gossipcop-2333766473 | Is Ben Affleck's Girlfriend Lindsay Shookus Pregnant?! | FAKE | UNVERIFIABLE | 0% |
| gossipcop-800399273 | List of songs recorded by Adele | FAKE | NO_CLAIMS | 0% |
| gossipcop-927093 | 32 Pics That Prove Gwen & Blake's Romance Is the Real Thing | REAL | UNVERIFIABLE | 10% |
| gossipcop-859153 | Tim McGraw | REAL | NO_CLAIMS | 0% |
| gossipcop-4294323822 | How the Weeknd Helped Selena Gomez Recover from Her Kidney Transplant: ‘One of H… | FAKE | REAL | 92% |
| gossipcop-3713172855 | Jennifer Aniston Should Just Follow 'Friends' To Get Over Her Justin Theroux Spl… | FAKE | UNVERIFIABLE | 10% |
| gossipcop-2630132619 | Expert: Meghan Markle and Duchess Kate's handwriting reveals major difference | FAKE | UNVERIFIABLE | 10% |
| gossipcop-4765521585 | Gwen Stefani “Ditching Plastic Surgery Thanks To Blake Shelton” Is Made-Up Story | FAKE | UNVERIFIABLE | 10% |
| gossipcop-2493253446 | Joaquin Phoenix Wants Rooney Mara’s Family To Stop NFL From Using Leather Footba… | FAKE | UNVERIFIABLE | 10% |
| gossipcop-7533724553 | Kim Kardashian Gets Her Stretch Marks Removed: 'It Didn't Hurt That Badly' | FAKE | REAL | 92% |
| gossipcop-138407761 | New Intimate Details of Tristan Thompson's Affair Prove It's Worse Than Khloé Ka… | FAKE | UNVERIFIABLE | 10% |
| gossipcop-8791564318 | Ariel Winter Levi Meaden Derailed | FAKE | NO_CLAIMS | 0% |
| gossipcop-853000 | Taylor Swift and Joe Alwyn | REAL | NO_CLAIMS | 0% |
| gossipcop-875174 | Big shoes to fill! AnnaSophia Robb swaps her high heels for comfy UGG boots on s… | REAL | UNVERIFIABLE | 10% |
| gossipcop-2123292210 | Feuding Over Friends! Jen ‘Secretly Despises’ Justin’s Creepy Pals | FAKE | UNVERIFIABLE | 10% |
| gossipcop-1659503489 | Ciara and Russell Wilson Welcome Sienna Princess | FAKE | REAL | 92% |
| gossipcop-946100 | Bootleg Quaaludes, Crabs & Shooting Ashes From a Cannon: 7 Bizarre Revelations F… | REAL | MISLEADING | 62% |
