"""Run agent evaluation on WELFake (Kaggle, 16 REAL + 16 FAKE balanced)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from verifyn.agent.eval.adapters import balanced_sample, load_welfake
from verifyn.agent.eval.scripts._runner import run_balanced_eval

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "welfake"
REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "reports"


def main() -> None:
    items = load_welfake(DATASET_DIR)
    items = balanced_sample(items, per_class=16, classes=("REAL", "FAKE"), seed=42)
    run_balanced_eval(dataset_name="welfake", items=items, report_dir=REPORT_DIR)


if __name__ == "__main__":
    main()
