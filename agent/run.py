#!/usr/bin/env python3
"""CLI entry point for the fake-news detection agent.

Usage
-----
# Analyse text passed as a string argument:
    python main.py --text "Breaking: Scientists discover cure for cancer using tap water"

# Analyse a text file:
    python main.py --file article.txt

# Read from stdin:
    echo "Some news text" | python main.py

# Verbose mode (stream agent thinking):
    python main.py --verbose --text "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

VERDICT_COLORS = {
    "REAL": "bright_green",
    "FAKE": "bright_red",
    "MISLEADING": "yellow",
    "PARTIALLY_FAKE": "orange3",
    "UNVERIFIABLE": "blue",
    "SATIRE": "magenta",
}

CONFIDENCE_COLORS = {
    "HIGH": "bright_green",
    "MEDIUM": "yellow",
    "LOW": "red",
}


def print_result(result) -> None:
    verdict_color = VERDICT_COLORS.get(result.verdict.value, "white")
    confidence_color = CONFIDENCE_COLORS.get(result.confidence_level.value, "white")

    # Header panel
    header = Text()
    header.append("VERDICT: ", style="bold white")
    header.append(result.verdict.value, style=f"bold {verdict_color}")
    header.append("   |   CONFIDENCE: ", style="bold white")
    header.append(f"{result.confidence:.0%}", style=f"bold {confidence_color}")
    header.append(f" ({result.confidence_level.value})", style=confidence_color)
    if result.manipulation_type.value != "NONE":
        header.append(f"\nManipulation: {result.manipulation_type.value}", style="bold yellow")

    console.print(Panel(header, title="[bold]Fact-Check Result[/bold]", border_style=verdict_color))

    # Summary
    console.print(Panel(result.summary, title="Summary", border_style="dim"))

    # Claims
    if result.main_claims:
        tbl = Table("#", "Claim", box=box.SIMPLE_HEAD, show_header=True)
        for i, claim in enumerate(result.main_claims, 1):
            tbl.add_row(str(i), claim)
        console.print(Panel(tbl, title="Main Claims Identified"))

    # Evidence tables
    if result.evidence_for:
        tbl = Table("Source", "Summary", "Credibility", box=box.SIMPLE_HEAD)
        for ev in result.evidence_for:
            tbl.add_row(ev.source, ev.summary[:120], ev.credibility)
        console.print(Panel(tbl, title="[green]Evidence Supporting Claims[/green]"))

    if result.evidence_against:
        tbl = Table("Source", "Summary", "Credibility", box=box.SIMPLE_HEAD)
        for ev in result.evidence_against:
            tbl.add_row(ev.source, ev.summary[:120], ev.credibility)
        console.print(Panel(tbl, title="[red]Evidence Refuting Claims[/red]"))

    # Fact-checker results
    if result.fact_checker_results:
        text = "\n".join(f"• {r}" for r in result.fact_checker_results)
        console.print(Panel(text, title="[cyan]Fact-Checker Findings[/cyan]"))

    # Primary source & date context
    details: list[str] = []
    if result.primary_source:
        details.append(f"**Primary source:** {result.primary_source}")
    if result.date_context:
        details.append(f"**Date/context:** {result.date_context}")
    if details:
        console.print(Panel("\n".join(details), title="Source Details"))

    # Sources checked
    if result.sources_checked:
        sources_text = "\n".join(f"• {s}" for s in result.sources_checked)
        console.print(Panel(sources_text, title=f"Sources Checked ({len(result.sources_checked)})"))

    # Full reasoning (collapsible in terminal)
    console.print(Panel(result.reasoning, title="Step-by-step Reasoning", border_style="dim"))


def print_json(result) -> None:
    print(json.dumps(result.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fake-news detection agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", "-t", help="News text to analyse")
    input_group.add_argument("--file", "-f", type=Path, help="Path to a text file containing the news")

    parser.add_argument(
        "--mode",
        "-m",
        choices=["fast", "precise"],
        default="fast",
        help="Inference mode: fast (low reasoning) or precise (medium reasoning)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream agent tool calls and reasoning")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of pretty print")

    args = parser.parse_args()

    # Determine input
    if args.text:
        news_text = args.text
    elif args.file:
        news_text = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        news_text = sys.stdin.read()
    else:
        parser.print_help()
        return 1

    news_text = news_text.strip()
    if not news_text:
        console.print("[red]Error: empty news text[/red]")
        return 1

    console.print(f"\n[bold]Analysing news text[/bold] ({len(news_text)} chars)...\n")

    try:
        from agent import analyze_news

        effort_map = {"fast": "low", "precise": "medium"}
        result = analyze_news(news_text, verbose=args.verbose, reasoning_effort=effort_map[args.mode])
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Agent error: {e}[/red]")
        raise

    if args.json:
        print_json(result)
    else:
        print_result(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
