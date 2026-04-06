# Evaluation

## Overview

Verifyn includes an evaluation framework for benchmarking agent accuracy against labeled datasets. The framework supports multiple dataset formats and generates classification reports with confusion matrices.

## Supported Datasets

| Dataset | Format | Labels | Adapter |
|---------|--------|--------|---------|
| **Custom** | JSON (`eval/dataset.json`) | All 7 Verifyn verdicts | `load_custom()` |
| **LIAR** | TSV | 6 labels → mapped to Verifyn verdicts | `load_liar()` |
| **FEVER** | JSONL | 3 labels → mapped to Verifyn verdicts | `load_fever()` |
| **FakeNewsNet** | CSV directory | 2 labels → REAL/FAKE | `load_fakenewsnet()` |

## Label Mappings

### LIAR → Verifyn

| LIAR Label | Verifyn Verdict |
|-----------|----------------|
| true, mostly-true | REAL |
| barely-true, false, pants-fire | FAKE |
| half-true | PARTIALLY_FAKE |

### FEVER → Verifyn

| FEVER Label | Verifyn Verdict |
|------------|----------------|
| SUPPORTS | REAL |
| REFUTES | FAKE |
| NOT ENOUGH INFO | UNVERIFIABLE |

## Metrics

The evaluation produces:

- **Confusion matrix** — actual vs predicted across all verdict classes
- **Per-class precision, recall, F1** — via sklearn-style classification report
- **Overall accuracy** — weighted by class frequency
- **Error analysis** — markdown table of misclassified examples with reasoning

## Custom Dataset

`agent/eval/dataset.json` contains 20 hand-labeled test cases:

- 3 REAL — documented events with official sources
- 7 FAKE — fabricated claims (bleach cures, moon hoax, microchips)
- 5 MISLEADING — out-of-context quotes, sensationalism
- 1 PARTIALLY_FAKE — mix of true and false
- 1 UNVERIFIABLE — private meeting, no primary source
- 1 SATIRE — The Onion-style absurdity
- 4 multilingual — Russian-language claims about Ukraine, vaccines

## Running Evaluation

```bash
# Run on custom dataset
python -m agent.evaluate agent/eval/dataset.json

# Run on LIAR dataset (TSV)
python -m agent.evaluate path/to/liar/test.tsv --type liar --sample 100

# Run on FEVER dataset (JSONL)
python -m agent.evaluate path/to/fever/paper_test.jsonl --type fever --sample 100
```

## Cost Considerations

Each evaluation example runs the full agent pipeline (LLM calls + web searches).

Use `--sample N` to limit evaluation size during development.

## Future Improvements

- Publish benchmark results in README with comparison to baselines
- Add cost/latency tracking per evaluation run
- Confidence calibration plots
- Automated regression testing on dataset subset in CI
