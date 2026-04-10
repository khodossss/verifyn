"""Re-score existing eval JSON reports with a soft binary scheme.

Rules
-----
- Items with actual verdict UNVERIFIABLE or NO_CLAIMS are excluded entirely
  (the agent admitted it could not decide; not counted as a miss).
- FAKE, MISLEADING, PARTIALLY_FAKE, SATIRE are collapsed into "FAKE".
- REAL stays REAL.
- Items where expected verdict is something other than REAL/FAKE are skipped
  (the source datasets are binary, so this almost never fires).

Generates ``<dataset>_soft.md`` next to each existing ``<dataset>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from verifyn.agent.eval.metrics import compute_confusion_matrix, export_markdown

REPORT_DIR = Path(__file__).resolve().parents[3] / "docs" / "reports"

EXCLUDE_ACTUAL = {"UNVERIFIABLE", "NO_CLAIMS", "ERROR"}
FAKE_LEANING = {"FAKE", "MISLEADING", "PARTIALLY_FAKE", "SATIRE"}


def _collapse(verdict: str) -> str | None:
    if verdict in EXCLUDE_ACTUAL:
        return None
    if verdict in FAKE_LEANING:
        return "FAKE"
    if verdict == "REAL":
        return "REAL"
    return None


def rescore_one(json_path: Path) -> dict | None:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    items = raw.get("results", [])
    if not items:
        return None

    soft_results: list[dict] = []
    excluded = 0
    for r in items:
        actual = _collapse(r["actual_verdict"])
        if actual is None:
            excluded += 1
            continue
        expected = _collapse(r["expected_verdict"])
        if expected is None:
            continue
        new_r = dict(r)
        new_r["actual_verdict"] = actual
        new_r["expected_verdict"] = expected
        new_r["verdict_match"] = actual == expected
        soft_results.append(new_r)

    if not soft_results:
        return None

    y_true = [r["expected_verdict"] for r in soft_results]
    y_pred = [r["actual_verdict"] for r in soft_results]
    cm = compute_confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])

    md = export_markdown(
        cm,
        soft_results,
        dataset_name=f"{raw['dataset']} (soft scoring)",
        extra_meta={
            "Original total": raw["total"],
            "Excluded (UNVERIFIABLE/NO_CLAIMS/ERROR)": excluded,
            "Scored items": len(soft_results),
            "Original accuracy": f"{raw['verdict_accuracy']:.1%}",
            "Soft scoring rules": "FAKE+MISLEADING+PARTIALLY_FAKE+SATIRE -> FAKE; UNVERIFIABLE/NO_CLAIMS dropped",
        },
    )

    out_path = REPORT_DIR / f"{raw['dataset']}_soft.md"
    out_path.write_text(md, encoding="utf-8")

    return {
        "dataset": raw["dataset"],
        "original_total": raw["total"],
        "scored": len(soft_results),
        "excluded": excluded,
        "soft_accuracy": cm["accuracy"],
        "soft_macro_f1": cm["macro_f1"],
        "soft_real_f1": cm["per_class"].get("REAL", {}).get("f1", 0.0),
        "soft_fake_f1": cm["per_class"].get("FAKE", {}).get("f1", 0.0),
        "report_path": str(out_path),
    }


def main() -> None:
    json_files = sorted(REPORT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON reports found in {REPORT_DIR}")
        return

    print(f"Re-scoring {len(json_files)} reports with soft binary rules...\n")
    print(f"{'Dataset':<28} {'Total':>6} {'Scored':>7} {'Excl':>5} {'Acc':>6} {'F1':>6}")
    print("-" * 64)

    summary = []
    for jf in json_files:
        if jf.stem.endswith("_soft"):
            continue
        result = rescore_one(jf)
        if result is None:
            print(f"{jf.stem:<28} (no data)")
            continue
        summary.append(result)
        print(
            f"{result['dataset']:<28} {result['original_total']:>6} {result['scored']:>7} "
            f"{result['excluded']:>5} {result['soft_accuracy']:>5.0%} {result['soft_macro_f1']:>5.2f}"
        )

    print(f"\nSoft reports written to {REPORT_DIR}/<dataset>_soft.md")


if __name__ == "__main__":
    main()
