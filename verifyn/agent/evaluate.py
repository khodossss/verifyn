#!/usr/bin/env python3
"""
Evaluation script — runs the fact-check agent against labeled datasets
and reports precision, recall, F1, confusion matrix, and classification report.

Supports three dataset formats:
- **Custom JSON** (``agent/eval/dataset.json``) — project's own smoke-test set
- **LIAR**  (PolitiFact TSV) — download from https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **FEVER** (JSONL)          — download from https://fever.ai/resources.html

Usage
-----
    # Smoke test (built-in 20 items)
    python -m agent.evaluate

    # LIAR benchmark — sample 50 items from test.tsv
    python -m agent.evaluate -d agent/eval/liar/test.tsv --sample 50

    # FEVER benchmark — sample 50 items
    python -m agent.evaluate -d agent/eval/fever/paper_dev.jsonl --sample 50

    # Export markdown report
    python -m agent.evaluate -d agent/eval/liar/test.tsv --sample 50 --report eval_report.md

    # Run specific IDs from custom dataset
    python -m agent.evaluate --id 4 17

    # Save raw results as JSON
    python -m agent.evaluate --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()

# Ensure UTF-8 output on Windows
import io as _io
import os as _os

if _os.name == "nt":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from verifyn.agent.eval.adapters import detect_and_load  # noqa: E402
from verifyn.agent.eval.metrics import (  # noqa: E402
    compute_confusion_matrix,
    export_markdown,
    print_classification_report,
    print_confusion_matrix,
)

DATASET_PATH = Path(__file__).parent / "eval" / "dataset.json"

VERDICT_STYLE: dict[str, str] = {
    "REAL": "bold green",
    "FAKE": "bold red",
    "MISLEADING": "bold yellow",
    "PARTIALLY_FAKE": "bold orange3",
    "UNVERIFIABLE": "bold blue",
    "SATIRE": "bold magenta",
    "ERROR": "dim red",
}

VERDICT_ICON: dict[str, str] = {
    "REAL": "+",
    "FAKE": "X",
    "MISLEADING": "!",
    "PARTIALLY_FAKE": "~",
    "UNVERIFIABLE": "?",
    "SATIRE": "S",
    "ERROR": "E",
}

# Avoid Rich's LegacyWindowsTerm which can't handle all Unicode on cp1252 consoles
console = Console(force_terminal=True)


def _verdict_text(verdict: str, *, dim: bool = False) -> Text:
    style = VERDICT_STYLE.get(verdict, "white")
    if dim:
        style = "dim"
    icon = VERDICT_ICON.get(verdict, "?")
    return Text(f"{icon} {verdict}", style=style)


def _match_icon(expected: str, actual: str) -> Text:
    if actual == "ERROR":
        return Text("[ERROR]", style="dim red")
    if expected == actual:
        return Text("[PASS]", style="bold green")
    return Text("[FAIL]", style="bold red")


def run_item(item: dict) -> dict:
    from verifyn.agent import analyze_news

    start = time.time()
    error: str | None = None
    actual_verdict: str = "ERROR"
    actual_manipulation: str = "NONE"
    actual_confidence: float = 0.0
    summary: str = ""

    try:
        result = analyze_news(item["text"])
        actual_verdict = result.verdict.value
        actual_manipulation = result.manipulation_type.value
        actual_confidence = result.confidence
        summary = result.summary
    except Exception as exc:
        error = str(exc)

    elapsed = time.time() - start

    return {
        "id": item["id"],
        "text_preview": item["text"][:80] + ("…" if len(item["text"]) > 80 else ""),
        "expected_verdict": item["expected_verdict"],
        "expected_manipulation": item.get("expected_manipulation", "NONE"),
        "actual_verdict": actual_verdict,
        "actual_manipulation": actual_manipulation,
        "confidence": actual_confidence,
        "verdict_match": actual_verdict == item["expected_verdict"],
        "manipulation_match": actual_manipulation == item.get("expected_manipulation", "NONE"),
        "summary": summary,
        "elapsed": round(elapsed, 1),
        "error": error,
        "notes": item.get("notes", ""),
        "source_dataset": item.get("source_dataset", "custom"),
        "original_label": item.get("original_label", ""),
    }


def print_item_result(r: dict, index: int, total: int) -> None:
    console.print()
    console.print(
        Rule(
            f"[dim]Item {index}/{total}[/dim]  [bold]#{r['id']}[/bold]",
            style="bright_blue",
        )
    )

    # Text preview
    console.print(f"[dim]Text:[/dim] {r['text_preview']}")
    if r["notes"]:
        console.print(f"[dim]Notes:[/dim] {r['notes']}")
    console.print()

    # Verdict comparison
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim")
    tbl.add_column("", width=14)
    tbl.add_column("Verdict", width=16)
    tbl.add_column("Manipulation", width=22)
    tbl.add_column("Confidence", width=10)

    tbl.add_row(
        "Expected",
        _verdict_text(r["expected_verdict"]),
        r["expected_manipulation"],
        "—",
    )
    tbl.add_row(
        "Actual",
        _verdict_text(r["actual_verdict"]),
        r["actual_manipulation"],
        f"{r['confidence']:.0%}" if r["actual_verdict"] != "ERROR" else "—",
    )

    verdict_ok = r["verdict_match"]
    manip_ok = r["manipulation_match"]
    tbl.add_row(
        "Match",
        Text("[PASS]", style="bold green") if verdict_ok else Text("[FAIL]", style="bold red"),
        Text("[OK]", style="green") if manip_ok else Text("[NO]", style="red"),
        f"{r['elapsed']}s",
    )
    console.print(tbl)

    if r["error"]:
        console.print(f"[bold red]Error:[/bold red] {r['error']}")
    elif r["summary"]:
        console.print(Panel(r["summary"], title="Agent Summary", border_style="dim", padding=(0, 1)))


def print_summary(results: list[dict]) -> None:
    """Print legacy summary table (overall stats + per-verdict breakdown)."""
    console.print()
    console.print(Rule("[bold]Evaluation Summary[/bold]", style="bright_blue"))
    console.print()

    total = len(results)
    errors = sum(1 for r in results if r["error"])
    verdict_pass = sum(1 for r in results if r["verdict_match"])
    manip_pass = sum(1 for r in results if r["manipulation_match"] and not r["error"])

    stats = Table(box=box.SIMPLE_HEAD, show_header=False)
    stats.add_column("Metric", style="bold")
    stats.add_column("Value")

    stats.add_row("Total items", str(total))
    stats.add_row(
        "Verdict accuracy",
        Text(
            f"{verdict_pass}/{total}  ({verdict_pass / total:.0%})",
            style="bold green"
            if verdict_pass / total >= 0.8
            else "bold yellow"
            if verdict_pass / total >= 0.6
            else "bold red",
        ),
    )
    stats.add_row(
        "Manipulation accuracy",
        f"{manip_pass}/{total - errors}  ({manip_pass / max(total - errors, 1):.0%})",
    )
    stats.add_row("Errors", str(errors))
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0
    stats.add_row("Avg time/item", f"{avg_time:.0f}s")

    console.print(stats)


def print_metrics(results: list[dict]) -> dict:
    """Compute and print confusion matrix + classification report. Returns cm dict."""
    # Filter out ERROR results for metrics
    valid = [r for r in results if r["actual_verdict"] != "ERROR"]
    if not valid:
        console.print("[red]No valid results to compute metrics.[/red]")
        return {}

    y_true = [r["expected_verdict"] for r in valid]
    y_pred = [r["actual_verdict"] for r in valid]

    cm = compute_confusion_matrix(y_true, y_pred)

    console.print()
    console.print(Rule("[bold]Metrics[/bold]", style="bright_blue"))
    console.print()

    print_confusion_matrix(cm, console)
    console.print()
    print_classification_report(cm, console)

    return cm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the fact-check agent against labeled datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit", "-n", type=int, help="Run only the first N items")
    parser.add_argument("--id", type=int, nargs="+", metavar="ID", dest="ids", help="Run specific item IDs only")
    parser.add_argument("--out", "-o", type=Path, help="Save full results to a JSON file")
    parser.add_argument("--dataset", "-d", type=Path, default=DATASET_PATH, help="Path to dataset (JSON/TSV/JSONL)")
    parser.add_argument(
        "--dataset-type",
        choices=["custom", "liar", "fever", "fakenewsnet"],
        help="Force dataset type (auto-detected from extension)",
    )
    parser.add_argument("--sample", "-s", type=int, help="Randomly sample N items from dataset (for large benchmarks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--report", "-r", type=Path, help="Export evaluation report as Markdown")
    args = parser.parse_args()

    # Load dataset
    try:
        dataset_type, dataset = detect_and_load(
            args.dataset,
            dataset_type=args.dataset_type,
            sample=args.sample,
            seed=args.seed,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load dataset: {exc}[/red]")
        return 1

    if args.ids:
        # IDs can be int or string depending on dataset
        str_ids = {str(i) for i in args.ids}
        dataset = [item for item in dataset if str(item["id"]) in str_ids or item["id"] in args.ids]
    if args.limit:
        dataset = dataset[: args.limit]

    if not dataset:
        console.print("[red]No items to evaluate.[/red]")
        return 1

    # Label distribution
    from collections import Counter

    dist = Counter(item["expected_verdict"] for item in dataset)
    dist_str = ", ".join(f"{v}: {c}" for v, c in sorted(dist.items()))

    console.print()
    console.print(
        Panel(
            f"[bold white]VERIFYN EVALUATION[/bold white]\n"
            f"[dim]Dataset: {args.dataset} ({dataset_type})  ·  Items: {len(dataset)}[/dim]\n"
            f"[dim]Distribution: {dist_str}[/dim]",
            border_style="bright_blue",
            padding=(0, 2),
        )
    )

    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task("Running agent…", total=len(dataset))

        for i, item in enumerate(dataset, 1):
            preview = item["text"][:50] + ("…" if len(item["text"]) > 50 else "")
            prog.update(task, description=f"[cyan]#{item['id']}[/cyan]  {preview}")

            result = run_item(item)
            results.append(result)

            icon = "+" if result["verdict_match"] else ("!!" if result["error"] else "X")
            style = "green" if result["verdict_match"] else ("dim red" if result["error"] else "red")
            prog.console.print(
                f"  [{style}]{icon}[/{style}]  "
                f"[bold]#{item['id']}[/bold]  "
                f"expected [bold]{item['expected_verdict']:<14}[/bold]  "
                f"got {_verdict_text(result['actual_verdict'])}  "
                f"[dim]{result['elapsed']}s[/dim]"
            )
            prog.advance(task)

    # Per-item details
    for i, r in enumerate(results, 1):
        print_item_result(r, i, len(results))

    # Summary + metrics
    print_summary(results)
    cm = print_metrics(results)

    # JSON output
    if args.out:
        output = {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(args.dataset),
            "dataset_type": dataset_type,
            "total": len(results),
            "verdict_accuracy": sum(1 for r in results if r["verdict_match"]) / len(results),
            "metrics": {
                "accuracy": cm.get("accuracy"),
                "macro_f1": cm.get("macro_f1"),
                "weighted_f1": cm.get("weighted_f1"),
                "per_class": cm.get("per_class"),
            }
            if cm
            else None,
            "results": results,
        }
        args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[dim]Results saved to {args.out}[/dim]")

    # Markdown report
    if args.report and cm:
        avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
        md = export_markdown(
            cm,
            results,
            dataset_name=f"{dataset_type} ({args.dataset.name})",
            extra_meta={
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Avg time/item": f"{avg_time:.0f}s",
                "Errors": sum(1 for r in results if r.get("error")),
                "Sample size": args.sample or len(dataset),
                "Seed": args.seed,
            },
        )
        args.report.write_text(md, encoding="utf-8")
        console.print(f"[dim]Report saved to {args.report}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
