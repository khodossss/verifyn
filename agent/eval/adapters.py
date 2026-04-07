"""Dataset adapters: load LIAR, FEVER, FakeNewsNet, WELFake, or custom JSON.

Each adapter returns a list of dicts:
    {"id": ..., "text": ..., "expected_verdict": ..., "expected_manipulation": ..., "notes": ...}

Download instructions
---------------------
LIAR
    https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    Unzip and place train.tsv / test.tsv in agent/eval/datasets/liar/

FEVER
    https://fever.ai/resources.html
    Download paper_dev.jsonl and place it in agent/eval/datasets/fever/

FakeNewsNet
    https://github.com/KaiDMML/FakeNewsNet
    Download the CSVs and place politifact_fake.csv, politifact_real.csv,
    gossipcop_fake.csv, gossipcop_real.csv in agent/eval/datasets/fakenewsnet/

WELFake
    https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
    Download the archive, extract Fake.csv and True.csv into
    agent/eval/datasets/welfake/

Custom (project smoke-test set)
    agent/eval/dataset.json (already in the repo)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# LIAR 6-class → Verifyn verdicts
LIAR_LABEL_MAP: dict[str, str] = {
    "true": "REAL",
    "mostly-true": "REAL",
    "half-true": "MISLEADING",
    "barely-true": "MISLEADING",
    "false": "FAKE",
    "pants-fire": "FAKE",
}

# FEVER 3-class → Verifyn verdicts
FEVER_LABEL_MAP: dict[str, str] = {
    "SUPPORTS": "REAL",
    "REFUTES": "FAKE",
    "NOT ENOUGH INFO": "UNVERIFIABLE",
}

# Reverse mapping for reporting
LIAR_REVERSE_MAP: dict[str, list[str]] = {}
for _src, _dst in LIAR_LABEL_MAP.items():
    LIAR_REVERSE_MAP.setdefault(_dst, []).append(_src)

FEVER_REVERSE_MAP: dict[str, list[str]] = {}
for _src, _dst in FEVER_LABEL_MAP.items():
    FEVER_REVERSE_MAP.setdefault(_dst, []).append(_src)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def load_custom(path: Path) -> list[dict[str, Any]]:
    """Load the project's own JSON dataset (``dataset.json``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        item.setdefault("expected_manipulation", "NONE")
        item.setdefault("notes", "")
    return data


def load_liar(
    path: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load a LIAR TSV file and map labels to Verifyn verdicts.

    Parameters
    ----------
    path:
        Path to a LIAR ``.tsv`` file (e.g. ``test.tsv``).
    sample:
        If given, randomly sample this many items (stratified attempt).
    seed:
        Random seed for reproducibility.
    """
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            # TSV columns: id, label, statement, subject, speaker, job, state, party, ...
            raw_label = row[1].strip().lower()
            verdict = LIAR_LABEL_MAP.get(raw_label)
            if verdict is None:
                continue
            items.append(
                {
                    "id": row[0].strip(),
                    "text": row[2].strip(),
                    "expected_verdict": verdict,
                    "expected_manipulation": "NONE",
                    "notes": f"LIAR original label: {raw_label}",
                    "source_dataset": "liar",
                    "original_label": raw_label,
                }
            )

    if sample and sample < len(items):
        rng = random.Random(seed)
        # Stratified sample: proportional to class distribution
        by_verdict: dict[str, list[dict]] = {}
        for item in items:
            by_verdict.setdefault(item["expected_verdict"], []).append(item)
        sampled: list[dict] = []
        for verdict, group in by_verdict.items():
            n = max(1, round(sample * len(group) / len(items)))
            sampled.extend(rng.sample(group, min(n, len(group))))
        rng.shuffle(sampled)
        items = sampled[:sample]

    return items


def load_fever(
    path: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load a FEVER JSONL file and map labels to Verifyn verdicts.

    Parameters
    ----------
    path:
        Path to a FEVER ``.jsonl`` file (e.g. ``paper_dev.jsonl``).
    sample:
        If given, randomly sample this many items.
    seed:
        Random seed for reproducibility.
    """
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_label = record.get("label", "").strip()
            verdict = FEVER_LABEL_MAP.get(raw_label)
            if verdict is None:
                continue
            claim = record.get("claim", "").strip()
            if not claim:
                continue
            items.append(
                {
                    "id": record.get("id", len(items)),
                    "text": claim,
                    "expected_verdict": verdict,
                    "expected_manipulation": "NONE",
                    "notes": f"FEVER original label: {raw_label}",
                    "source_dataset": "fever",
                    "original_label": raw_label,
                }
            )

    if sample and sample < len(items):
        rng = random.Random(seed)
        by_verdict: dict[str, list[dict]] = {}
        for item in items:
            by_verdict.setdefault(item["expected_verdict"], []).append(item)
        sampled: list[dict] = []
        for verdict, group in by_verdict.items():
            n = max(1, round(sample * len(group) / len(items)))
            sampled.extend(rng.sample(group, min(n, len(group))))
        rng.shuffle(sampled)
        items = sampled[:sample]

    return items


def load_fakenewsnet(
    path: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load FakeNewsNet from a directory containing the four CSV files.

    Parameters
    ----------
    path:
        Path to the directory containing ``politifact_fake.csv``,
        ``politifact_real.csv``, etc. OR path to a single CSV file.
    sample:
        If given, randomly sample this many items (stratified).
    seed:
        Random seed for reproducibility.
    """
    if path.is_file():
        # Single CSV — infer label from filename
        files = [(path, "fake" if "fake" in path.stem.lower() else "real")]
    else:
        # Directory — load all four CSVs
        files = []
        for name in ["politifact_fake.csv", "politifact_real.csv", "gossipcop_fake.csv", "gossipcop_real.csv"]:
            p = path / name
            if p.exists():
                label = "fake" if "fake" in name else "real"
                files.append((p, label))

    if not files:
        raise FileNotFoundError(
            f"No FakeNewsNet CSV files found in {path}. "
            "Expected: politifact_fake.csv, politifact_real.csv, "
            "gossipcop_fake.csv, gossipcop_real.csv"
        )

    # FakeNewsNet CSVs have a huge tweet_ids column — increase field limit
    csv.field_size_limit(10_000_000)

    items: list[dict[str, Any]] = []
    for filepath, label in files:
        source = filepath.stem  # e.g. "politifact_fake"
        verdict = "FAKE" if label == "fake" else "REAL"
        with filepath.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                if not title or len(title) < 10:
                    continue
                items.append(
                    {
                        "id": row.get("id", f"{source}_{len(items)}"),
                        "text": title,
                        "expected_verdict": verdict,
                        "expected_manipulation": "FABRICATED" if verdict == "FAKE" else "NONE",
                        "notes": f"FakeNewsNet source: {source}",
                        "source_dataset": "fakenewsnet",
                        "original_label": label,
                    }
                )

    if sample and sample < len(items):
        rng = random.Random(seed)
        by_verdict: dict[str, list[dict]] = {}
        for item in items:
            by_verdict.setdefault(item["expected_verdict"], []).append(item)
        sampled: list[dict] = []
        for verdict, group in by_verdict.items():
            n = max(1, round(sample * len(group) / len(items)))
            sampled.extend(rng.sample(group, min(n, len(group))))
        rng.shuffle(sampled)
        items = sampled[:sample]

    return items


def load_welfake(
    path: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load WELFake / ISOT-style fake-news dataset.

    Expects a directory with ``Fake.csv`` and ``True.csv`` (Kaggle format),
    each with at least ``title`` and ``text`` columns.
    """
    if path.is_file():
        files = [(path, "fake" if "fake" in path.stem.lower() else "real")]
    else:
        files = []
        for name, label in [("Fake.csv", "fake"), ("True.csv", "real")]:
            p = path / name
            if p.exists():
                files.append((p, label))

    if not files:
        raise FileNotFoundError(f"No WELFake CSV files found in {path}. Expected Fake.csv and True.csv")

    csv.field_size_limit(10_000_000)

    items: list[dict[str, Any]] = []
    for filepath, label in files:
        source = filepath.stem
        verdict = "FAKE" if label == "fake" else "REAL"
        with filepath.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                if not title or len(title) < 10:
                    continue
                items.append(
                    {
                        "id": f"{source}_{len(items)}",
                        "text": title,
                        "expected_verdict": verdict,
                        "expected_manipulation": "FABRICATED" if verdict == "FAKE" else "NONE",
                        "notes": f"WELFake source: {source}",
                        "source_dataset": "welfake",
                        "original_label": label,
                    }
                )

    if sample and sample < len(items):
        rng = random.Random(seed)
        by_verdict: dict[str, list[dict]] = {}
        for item in items:
            by_verdict.setdefault(item["expected_verdict"], []).append(item)
        sampled: list[dict] = []
        for verdict, group in by_verdict.items():
            n = max(1, round(sample * len(group) / len(items)))
            sampled.extend(rng.sample(group, min(n, len(group))))
        rng.shuffle(sampled)
        items = sampled[:sample]

    return items


def balanced_sample(
    items: list[dict[str, Any]],
    *,
    per_class: int,
    classes: tuple[str, ...] = ("REAL", "FAKE"),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return *per_class* items per verdict class. Skips classes with no items."""
    rng = random.Random(seed)
    by_verdict: dict[str, list[dict]] = {}
    for item in items:
        by_verdict.setdefault(item["expected_verdict"], []).append(item)

    out: list[dict] = []
    for cls in classes:
        group = by_verdict.get(cls, [])
        if not group:
            continue
        n = min(per_class, len(group))
        out.extend(rng.sample(group, n))

    rng.shuffle(out)
    return out


_SAMPABLE_LOADERS = {"liar", "fever", "fakenewsnet"}

# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "custom": load_custom,
    "liar": load_liar,
    "fever": load_fever,
    "fakenewsnet": load_fakenewsnet,
}


def detect_and_load(
    path: Path,
    *,
    dataset_type: str | None = None,
    sample: int | None = None,
    seed: int = 42,
) -> tuple[str, list[dict[str, Any]]]:
    """Auto-detect dataset type from file extension/content and load it.

    Returns (dataset_type, items).
    """
    if dataset_type:
        loader = DATASET_LOADERS[dataset_type]
        if dataset_type in _SAMPABLE_LOADERS:
            return dataset_type, loader(path, sample=sample, seed=seed)
        return dataset_type, loader(path)

    suffix = path.suffix.lower()

    # TSV → LIAR
    if suffix == ".tsv":
        return "liar", load_liar(path, sample=sample, seed=seed)

    # JSONL → FEVER
    if suffix == ".jsonl":
        return "fever", load_fever(path, sample=sample, seed=seed)

    # JSON → custom
    if suffix == ".json":
        return "custom", load_custom(path)

    # Directory → try FakeNewsNet
    if path.is_dir():
        return "fakenewsnet", load_fakenewsnet(path, sample=sample, seed=seed)

    # Single CSV with "fake" or "real" in name → FakeNewsNet
    if suffix == ".csv":
        return "fakenewsnet", load_fakenewsnet(path, sample=sample, seed=seed)

    raise ValueError(f"Cannot auto-detect dataset type for {path}. Use --dataset-type.")
