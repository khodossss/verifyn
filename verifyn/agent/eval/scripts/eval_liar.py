"""Run agent evaluation on LIAR test set (10 REAL + 10 FAKE balanced)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from verifyn.agent.eval.adapters import balanced_sample, load_liar
from verifyn.agent.eval.scripts._runner import run_balanced_eval

DATASET_PATH = Path(__file__).resolve().parents[4] / "data" / "datasets" / "liar" / "test.tsv"
REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "reports"


def main() -> None:
    items = load_liar(DATASET_PATH)
    items = balanced_sample(items, per_class=16, classes=("REAL", "FAKE"), seed=42)
    run_balanced_eval(dataset_name="liar", items=items, report_dir=REPORT_DIR)


if __name__ == "__main__":
    main()
