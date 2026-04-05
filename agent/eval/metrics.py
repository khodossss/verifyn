"""Evaluation metrics — confusion matrix, precision/recall/F1, classification report.

Pure Python implementation (no sklearn dependency) for portability.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

VERDICTS = ["REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute confusion matrix and per-class metrics.

    Returns
    -------
    dict with keys:
        labels: list of label names
        matrix: list[list[int]] — matrix[true_idx][pred_idx]
        per_class: dict[label → {tp, fp, fn, precision, recall, f1, support}]
        macro_precision, macro_recall, macro_f1: macro-averaged metrics
        weighted_precision, weighted_recall, weighted_f1: weighted by support
        accuracy: overall accuracy
        total: total number of predictions
    """
    if labels is None:
        seen = set(y_true) | set(y_pred)
        labels = [v for v in VERDICTS if v in seen]
        # Add any labels not in VERDICTS (e.g. ERROR)
        for label in sorted(seen - set(labels)):
            labels.append(label)

    label_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    for true, pred in zip(y_true, y_pred):
        ti = label_idx.get(true)
        pi = label_idx.get(pred)
        if ti is not None and pi is not None:
            matrix[ti][pi] += 1

    # Per-class metrics
    per_class: dict[str, dict[str, Any]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        support = sum(matrix[i][j] for j in range(n))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Macro average (only over classes with support > 0)
    active = [m for m in per_class.values() if m["support"] > 0]
    macro_p = _safe_div(sum(m["precision"] for m in active), len(active))
    macro_r = _safe_div(sum(m["recall"] for m in active), len(active))
    macro_f1 = _safe_div(2 * macro_p * macro_r, macro_p + macro_r)

    # Weighted average
    total_support = sum(m["support"] for m in active)
    weighted_p = _safe_div(sum(m["precision"] * m["support"] for m in active), total_support)
    weighted_r = _safe_div(sum(m["recall"] * m["support"] for m in active), total_support)
    weighted_f1 = _safe_div(2 * weighted_p * weighted_r, weighted_p + weighted_r)

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    return {
        "labels": labels,
        "matrix": matrix,
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "accuracy": _safe_div(correct, total),
        "total": total,
    }


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------

VERDICT_STYLE: dict[str, str] = {
    "REAL": "green",
    "FAKE": "red",
    "MISLEADING": "yellow",
    "PARTIALLY_FAKE": "orange3",
    "UNVERIFIABLE": "blue",
    "SATIRE": "magenta",
    "ERROR": "dim red",
}


def print_confusion_matrix(cm: dict[str, Any], console: Console | None = None) -> None:
    """Render confusion matrix as a Rich table."""
    console = console or Console()
    labels = cm["labels"]
    matrix = cm["matrix"]

    table = Table(
        title="Confusion Matrix (rows=true, cols=predicted)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("True \\ Pred", style="bold", width=16)
    for label in labels:
        table.add_column(label, justify="center", width=max(6, len(label) + 1))

    for i, label in enumerate(labels):
        row_style = VERDICT_STYLE.get(label, "white")
        cells: list[str | Text] = [Text(label, style=row_style)]
        for j in range(len(labels)):
            val = matrix[i][j]
            if i == j and val > 0:
                cells.append(Text(str(val), style="bold green"))
            elif val > 0:
                cells.append(Text(str(val), style="bold red"))
            else:
                cells.append(Text(".", style="dim"))
        table.add_row(*cells)

    console.print(table)


def print_classification_report(cm: dict[str, Any], console: Console | None = None) -> None:
    """Render per-class precision/recall/F1 as a Rich table."""
    console = console or Console()
    labels = cm["labels"]
    per_class = cm["per_class"]

    table = Table(
        title="Classification Report",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("Verdict", width=16)
    table.add_column("Precision", justify="center", width=10)
    table.add_column("Recall", justify="center", width=10)
    table.add_column("F1-Score", justify="center", width=10)
    table.add_column("Support", justify="center", width=8)

    def _fmt(val: float) -> Text:
        s = f"{val:.2f}"
        if val >= 0.8:
            return Text(s, style="green")
        elif val >= 0.5:
            return Text(s, style="yellow")
        else:
            return Text(s, style="red")

    for label in labels:
        m = per_class[label]
        if m["support"] == 0:
            continue
        style = VERDICT_STYLE.get(label, "white")
        table.add_row(
            Text(label, style=style),
            _fmt(m["precision"]),
            _fmt(m["recall"]),
            _fmt(m["f1"]),
            str(m["support"]),
        )

    # Separator + aggregates
    table.add_section()
    table.add_row(
        Text("macro avg", style="bold"),
        _fmt(cm["macro_precision"]),
        _fmt(cm["macro_recall"]),
        _fmt(cm["macro_f1"]),
        str(cm["total"]),
    )
    table.add_row(
        Text("weighted avg", style="bold"),
        _fmt(cm["weighted_precision"]),
        _fmt(cm["weighted_recall"]),
        _fmt(cm["weighted_f1"]),
        str(cm["total"]),
    )
    table.add_section()
    table.add_row(
        Text("accuracy", style="bold"),
        "",
        "",
        _fmt(cm["accuracy"]),
        str(cm["total"]),
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def export_markdown(
    cm: dict[str, Any],
    results: list[dict],
    *,
    dataset_name: str = "custom",
    extra_meta: dict | None = None,
) -> str:
    """Generate a markdown evaluation report."""
    lines: list[str] = []
    lines.append("# Verifyn Evaluation Report\n")

    # Metadata
    lines.append("## Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Dataset | {dataset_name} |")
    lines.append(f"| Total samples | {cm['total']} |")
    lines.append(f"| Accuracy | {cm['accuracy']:.1%} |")
    lines.append(f"| Macro F1 | {cm['macro_f1']:.3f} |")
    lines.append(f"| Weighted F1 | {cm['weighted_f1']:.3f} |")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # Classification report
    lines.append("## Classification Report\n")
    lines.append("| Verdict | Precision | Recall | F1-Score | Support |")
    lines.append("|---------|-----------|--------|----------|---------|")
    for label in cm["labels"]:
        m = cm["per_class"][label]
        if m["support"] == 0:
            continue
        lines.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    lines.append(
        f"| **macro avg** | {cm['macro_precision']:.2f} | {cm['macro_recall']:.2f} | {cm['macro_f1']:.2f} | {cm['total']} |"
    )
    lines.append(
        f"| **weighted avg** | {cm['weighted_precision']:.2f} | {cm['weighted_recall']:.2f} | {cm['weighted_f1']:.2f} | {cm['total']} |"
    )
    lines.append(f"| **accuracy** | | | {cm['accuracy']:.2f} | {cm['total']} |")
    lines.append("")

    # Confusion matrix
    lines.append("## Confusion Matrix\n")
    labels = cm["labels"]
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, label in enumerate(labels):
        row = (
            f"| **{label}** | "
            + " | ".join(f"**{cm['matrix'][i][j]}**" if i == j else str(cm["matrix"][i][j]) for j in range(len(labels)))
            + " |"
        )
        lines.append(row)
    lines.append("")

    # Error analysis
    errors = [r for r in results if not r["verdict_match"] and not r.get("error")]
    if errors:
        lines.append("## Error Analysis\n")
        lines.append(f"Misclassified {len(errors)} out of {len(results)} items.\n")
        lines.append("| ID | Text | Expected | Predicted | Confidence |")
        lines.append("|----|------|----------|-----------|------------|")
        for r in errors[:30]:  # Cap at 30 to keep report readable
            text = r["text_preview"].replace("|", "\\|")
            lines.append(
                f"| {r['id']} | {text} | {r['expected_verdict']} | {r['actual_verdict']} | {r['confidence']:.0%} |"
            )
        if len(errors) > 30:
            lines.append(f"\n*...and {len(errors) - 30} more misclassifications.*\n")
    lines.append("")

    return "\n".join(lines)
