"""Shared runner for parallel per-dataset evaluation scripts.

Sets VERIFYN_EVAL_MODE=1 so the agent skips DB reads/writes and similarity
search, then runs all items concurrently via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Disable DB writes/similarity search before agent imports propagate
os.environ.setdefault("VERIFYN_EVAL_MODE", "1")

from agent.eval.metrics import export_markdown
from agent.evaluate import console, print_metrics, print_summary, run_item


async def _run_item_async(item: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        return await asyncio.to_thread(run_item, item)


async def _run_all(items: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [_run_item_async(item, sem) for item in items]
    return await asyncio.gather(*tasks)


def run_balanced_eval(
    *,
    dataset_name: str,
    items: list[dict],
    report_dir: Path,
    concurrency: int = 20,
) -> dict:
    """Run the agent against a pre-balanced item list and persist results."""
    report_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]Evaluating: {dataset_name}[/bold]")
    console.print(f"Items: {len(items)} | Concurrency: {concurrency}")

    t0 = time.time()
    results = asyncio.run(_run_all(items, concurrency))
    wall = time.time() - t0
    console.print(f"\n[dim]Total wall time: {wall:.0f}s[/dim]")

    print_summary(results)
    cm = print_metrics(results)

    out = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "total": len(results),
        "verdict_accuracy": sum(1 for r in results if r["verdict_match"]) / len(results),
        "errors": sum(1 for r in results if r["error"]),
        "wall_time_seconds": round(wall, 1),
        "concurrency": concurrency,
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
    (report_dir / f"{dataset_name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if cm:
        avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
        md = export_markdown(
            cm,
            results,
            dataset_name=dataset_name,
            extra_meta={
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Avg time/item": f"{avg_time:.0f}s",
                "Wall time": f"{wall:.0f}s",
                "Concurrency": concurrency,
                "Errors": sum(1 for r in results if r.get("error")),
                "Sample size": len(items),
            },
        )
        (report_dir / f"{dataset_name}.md").write_text(md, encoding="utf-8")

    console.print(f"\n[dim]Reports saved to {report_dir}/{dataset_name}.{{json,md}}[/dim]")
    return out
