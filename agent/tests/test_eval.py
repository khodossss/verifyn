"""Tests for evaluation infrastructure — adapters and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.eval.adapters import (
    FEVER_LABEL_MAP,
    LIAR_LABEL_MAP,
    detect_and_load,
    load_custom,
    load_fakenewsnet,
    load_fever,
    load_liar,
)
from agent.eval.metrics import compute_confusion_matrix, export_markdown

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_dataset(tmp_path: Path) -> Path:
    data = [
        {"id": 1, "text": "Test claim one", "expected_verdict": "REAL", "expected_manipulation": "NONE", "notes": ""},
        {
            "id": 2,
            "text": "Test claim two",
            "expected_verdict": "FAKE",
            "expected_manipulation": "FABRICATED",
            "notes": "",
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def liar_tsv(tmp_path: Path) -> Path:
    # LIAR TSV format: id \t label \t statement \t subject \t speaker \t job \t state \t party \t ...
    rows = [
        "1001.json\ttrue\tThe economy grew by 3% last quarter.\teconomy\tJohn Doe\tsenator\tCA\trepublican\t5\t3\t2\t1\t0",
        "1002.json\tfalse\tMars has been colonized by humans since 2020.\tspace\tJane Smith\tanalyst\tNY\tdemocrat\t0\t1\t2\t3\t4",
        "1003.json\thalf-true\tThe unemployment rate is at a record low.\teconomy\tBob\tgovernor\tTX\trepublican\t2\t2\t2\t2\t2",
        "1004.json\tpants-fire\tAliens built the pyramids.\thistory\tAnon\tblogger\tFL\tnone\t0\t0\t0\t0\t5",
        "1005.json\tbarely-true\tCrime has doubled in major cities.\tcrime\tPol\tmayor\tIL\tdemocrat\t1\t3\t1\t0\t0",
        "1006.json\tmostly-true\tThe US has the largest GDP in the world.\teconomy\tEcon\tprofessor\tMA\tindependent\t4\t1\t0\t0\t0",
    ]
    p = tmp_path / "test.tsv"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture
def fever_jsonl(tmp_path: Path) -> Path:
    records = [
        {"id": 1, "label": "SUPPORTS", "claim": "Water boils at 100°C at sea level."},
        {"id": 2, "label": "REFUTES", "claim": "The sun revolves around the Earth."},
        {"id": 3, "label": "NOT ENOUGH INFO", "claim": "The president had a secret meeting last Thursday."},
        {"id": 4, "label": "SUPPORTS", "claim": "Python is a programming language."},
    ]
    p = tmp_path / "dev.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestCustomAdapter:
    def test_load_custom(self, custom_dataset: Path):
        items = load_custom(custom_dataset)
        assert len(items) == 2
        assert items[0]["expected_verdict"] == "REAL"
        assert items[1]["expected_verdict"] == "FAKE"

    def test_defaults_manipulation(self, tmp_path: Path):
        data = [{"id": 1, "text": "Test", "expected_verdict": "REAL"}]
        p = tmp_path / "ds.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        items = load_custom(p)
        assert items[0]["expected_manipulation"] == "NONE"


class TestLiarAdapter:
    def test_load_all(self, liar_tsv: Path):
        items = load_liar(liar_tsv)
        assert len(items) == 6

    def test_label_mapping(self, liar_tsv: Path):
        items = load_liar(liar_tsv)
        verdicts = {item["original_label"]: item["expected_verdict"] for item in items}
        assert verdicts["true"] == "REAL"
        assert verdicts["false"] == "FAKE"
        assert verdicts["half-true"] == "MISLEADING"
        assert verdicts["pants-fire"] == "FAKE"
        assert verdicts["barely-true"] == "MISLEADING"
        assert verdicts["mostly-true"] == "REAL"

    def test_sample(self, liar_tsv: Path):
        items = load_liar(liar_tsv, sample=3)
        assert len(items) == 3

    def test_source_dataset_tag(self, liar_tsv: Path):
        items = load_liar(liar_tsv)
        assert all(item["source_dataset"] == "liar" for item in items)

    def test_skips_bad_rows(self, tmp_path: Path):
        p = tmp_path / "bad.tsv"
        p.write_text("short\n", encoding="utf-8")
        items = load_liar(p)
        assert items == []


class TestFeverAdapter:
    def test_load_all(self, fever_jsonl: Path):
        items = load_fever(fever_jsonl)
        assert len(items) == 4

    def test_label_mapping(self, fever_jsonl: Path):
        items = load_fever(fever_jsonl)
        id_to_verdict = {item["id"]: item["expected_verdict"] for item in items}
        assert id_to_verdict[1] == "REAL"
        assert id_to_verdict[2] == "FAKE"
        assert id_to_verdict[3] == "UNVERIFIABLE"

    def test_sample(self, fever_jsonl: Path):
        items = load_fever(fever_jsonl, sample=2)
        assert len(items) == 2

    def test_source_dataset_tag(self, fever_jsonl: Path):
        items = load_fever(fever_jsonl)
        assert all(item["source_dataset"] == "fever" for item in items)


@pytest.fixture
def fakenewsnet_dir(tmp_path: Path) -> Path:
    fake_csv = tmp_path / "politifact_fake.csv"
    fake_csv.write_text(
        "id,news_url,title,tweet_ids\n"
        "pf001,http://example.com/fake1,BREAKING NFL Team Goes Bankrupt Over Kneeling,123\n"
        "pf002,http://example.com/fake2,Obama Ordered to Pay 400 Million,456\n",
        encoding="utf-8",
    )
    real_csv = tmp_path / "politifact_real.csv"
    real_csv.write_text(
        "id,news_url,title,tweet_ids\n"
        "pf003,http://example.com/real1,Senate Passes New Infrastructure Bill,789\n"
        "pf004,http://example.com/real2,Fed Raises Interest Rates by Quarter Point,012\n",
        encoding="utf-8",
    )
    return tmp_path


class TestFakeNewsNetAdapter:
    def test_load_from_directory(self, fakenewsnet_dir: Path):
        items = load_fakenewsnet(fakenewsnet_dir)
        assert len(items) == 4

    def test_label_mapping(self, fakenewsnet_dir: Path):
        items = load_fakenewsnet(fakenewsnet_dir)
        verdicts = {item["id"]: item["expected_verdict"] for item in items}
        assert verdicts["pf001"] == "FAKE"
        assert verdicts["pf002"] == "FAKE"
        assert verdicts["pf003"] == "REAL"
        assert verdicts["pf004"] == "REAL"

    def test_source_dataset_tag(self, fakenewsnet_dir: Path):
        items = load_fakenewsnet(fakenewsnet_dir)
        assert all(item["source_dataset"] == "fakenewsnet" for item in items)

    def test_sample(self, fakenewsnet_dir: Path):
        items = load_fakenewsnet(fakenewsnet_dir, sample=2)
        assert len(items) == 2

    def test_single_csv(self, fakenewsnet_dir: Path):
        items = load_fakenewsnet(fakenewsnet_dir / "politifact_fake.csv")
        assert len(items) == 2
        assert all(i["expected_verdict"] == "FAKE" for i in items)

    def test_skips_short_titles(self, tmp_path: Path):
        csv_file = tmp_path / "politifact_fake.csv"
        csv_file.write_text(
            "id,news_url,title,tweet_ids\n"
            "pf1,http://x.com,Short,1\n"
            "pf2,http://x.com,This title is long enough to pass the filter,2\n",
            encoding="utf-8",
        )
        items = load_fakenewsnet(csv_file)
        assert len(items) == 1

    def test_empty_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_fakenewsnet(tmp_path)


class TestAutoDetect:
    def test_detect_json(self, custom_dataset: Path):
        dtype, items = detect_and_load(custom_dataset)
        assert dtype == "custom"
        assert len(items) == 2

    def test_detect_tsv(self, liar_tsv: Path):
        dtype, items = detect_and_load(liar_tsv)
        assert dtype == "liar"
        assert len(items) == 6

    def test_detect_jsonl(self, fever_jsonl: Path):
        dtype, items = detect_and_load(fever_jsonl)
        assert dtype == "fever"
        assert len(items) == 4

    def test_force_type(self, liar_tsv: Path):
        dtype, items = detect_and_load(liar_tsv, dataset_type="liar")
        assert dtype == "liar"

    def test_detect_dir_as_fakenewsnet(self, fakenewsnet_dir: Path):
        dtype, items = detect_and_load(fakenewsnet_dir)
        assert dtype == "fakenewsnet"
        assert len(items) == 4

    def test_detect_csv_as_fakenewsnet(self, fakenewsnet_dir: Path):
        dtype, items = detect_and_load(fakenewsnet_dir / "politifact_fake.csv")
        assert dtype == "fakenewsnet"
        assert len(items) == 2

    def test_unknown_extension(self, tmp_path: Path):
        p = tmp_path / "data.xyz"
        p.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_and_load(p)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y_true = ["REAL", "FAKE", "MISLEADING", "REAL", "FAKE"]
        y_pred = ["REAL", "FAKE", "MISLEADING", "REAL", "FAKE"]
        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm["accuracy"] == 1.0
        assert cm["macro_f1"] == 1.0
        assert cm["total"] == 5
        for label, m in cm["per_class"].items():
            assert m["precision"] == 1.0
            assert m["recall"] == 1.0
            assert m["f1"] == 1.0

    def test_all_wrong(self):
        y_true = ["REAL", "REAL", "REAL"]
        y_pred = ["FAKE", "FAKE", "FAKE"]
        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm["accuracy"] == 0.0
        assert cm["per_class"]["REAL"]["recall"] == 0.0
        assert cm["per_class"]["FAKE"]["precision"] == 0.0

    def test_mixed_results(self):
        y_true = ["REAL", "REAL", "FAKE", "FAKE", "MISLEADING"]
        y_pred = ["REAL", "FAKE", "FAKE", "MISLEADING", "MISLEADING"]
        cm = compute_confusion_matrix(y_true, y_pred)

        assert cm["accuracy"] == 3 / 5
        assert cm["per_class"]["REAL"]["tp"] == 1
        assert cm["per_class"]["REAL"]["fn"] == 1
        assert cm["per_class"]["FAKE"]["tp"] == 1
        assert cm["per_class"]["FAKE"]["fp"] == 1
        assert cm["per_class"]["MISLEADING"]["tp"] == 1
        assert cm["per_class"]["MISLEADING"]["fp"] == 1

    def test_matrix_shape(self):
        y_true = ["REAL", "FAKE"]
        y_pred = ["REAL", "REAL"]
        cm = compute_confusion_matrix(y_true, y_pred)

        n = len(cm["labels"])
        assert len(cm["matrix"]) == n
        assert all(len(row) == n for row in cm["matrix"])

    def test_custom_labels(self):
        y_true = ["A", "B", "C"]
        y_pred = ["A", "B", "A"]
        cm = compute_confusion_matrix(y_true, y_pred, labels=["A", "B", "C"])

        assert cm["labels"] == ["A", "B", "C"]
        assert cm["per_class"]["C"]["tp"] == 0

    def test_empty_class(self):
        y_true = ["REAL", "REAL"]
        y_pred = ["REAL", "REAL"]
        cm = compute_confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])

        assert cm["per_class"]["FAKE"]["support"] == 0
        assert cm["per_class"]["FAKE"]["precision"] == 0.0


class TestMarkdownExport:
    def test_generates_valid_markdown(self):
        y_true = ["REAL", "FAKE", "REAL"]
        y_pred = ["REAL", "FAKE", "FAKE"]
        cm = compute_confusion_matrix(y_true, y_pred)
        results = [
            {
                "id": 1,
                "text_preview": "claim 1",
                "expected_verdict": "REAL",
                "actual_verdict": "REAL",
                "verdict_match": True,
                "confidence": 0.9,
                "error": None,
            },
            {
                "id": 2,
                "text_preview": "claim 2",
                "expected_verdict": "FAKE",
                "actual_verdict": "FAKE",
                "verdict_match": True,
                "confidence": 0.8,
                "error": None,
            },
            {
                "id": 3,
                "text_preview": "claim 3",
                "expected_verdict": "REAL",
                "actual_verdict": "FAKE",
                "verdict_match": False,
                "confidence": 0.7,
                "error": None,
            },
        ]
        md = export_markdown(cm, results, dataset_name="test")

        assert "# Verifyn Evaluation Report" in md
        assert "Confusion Matrix" in md
        assert "Classification Report" in md
        assert "Error Analysis" in md
        assert "claim 3" in md  # misclassified item
        assert "66.7%" in md  # accuracy

    def test_no_error_analysis_when_all_correct(self):
        y_true = ["REAL", "FAKE"]
        y_pred = ["REAL", "FAKE"]
        cm = compute_confusion_matrix(y_true, y_pred)
        results = [
            {
                "id": 1,
                "text_preview": "c1",
                "expected_verdict": "REAL",
                "actual_verdict": "REAL",
                "verdict_match": True,
                "confidence": 0.9,
                "error": None,
            },
            {
                "id": 2,
                "text_preview": "c2",
                "expected_verdict": "FAKE",
                "actual_verdict": "FAKE",
                "verdict_match": True,
                "confidence": 0.8,
                "error": None,
            },
        ]
        md = export_markdown(cm, results)

        assert "Error Analysis" not in md


# ---------------------------------------------------------------------------
# Label mapping coverage
# ---------------------------------------------------------------------------


class TestLabelMappings:
    def test_liar_covers_all_labels(self):
        expected_labels = {"true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"}
        assert set(LIAR_LABEL_MAP.keys()) == expected_labels

    def test_fever_covers_all_labels(self):
        expected_labels = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
        assert set(FEVER_LABEL_MAP.keys()) == expected_labels

    def test_liar_maps_to_valid_verdicts(self):
        valid = {"REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"}
        for verdict in LIAR_LABEL_MAP.values():
            assert verdict in valid

    def test_fever_maps_to_valid_verdicts(self):
        valid = {"REAL", "FAKE", "MISLEADING", "PARTIALLY_FAKE", "UNVERIFIABLE", "SATIRE"}
        for verdict in FEVER_LABEL_MAP.values():
            assert verdict in valid
