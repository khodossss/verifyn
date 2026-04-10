"""Run agent evaluation on FakeNewsNet GossipCop subset (16 REAL + 16 FAKE)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from verifyn.agent.eval.adapters import balanced_sample, load_fakenewsnet
from verifyn.agent.eval.scripts._runner import run_balanced_eval

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "fakenewsnet"
REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "reports"


def main() -> None:
    files = [DATASET_DIR / "gossipcop_fake.csv", DATASET_DIR / "gossipcop_real.csv"]
    items: list[dict] = []
    for f in files:
        items.extend(load_fakenewsnet(f))

    items = balanced_sample(items, per_class=16, classes=("REAL", "FAKE"), seed=42)
    run_balanced_eval(dataset_name="fakenewsnet_gossipcop", items=items, report_dir=REPORT_DIR)


if __name__ == "__main__":
    main()
