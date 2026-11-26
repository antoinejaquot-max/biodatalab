
"""Tests for BioDataLab JSON export."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodatalab.export import export_report_json


def test_export_report_json(tmp_path):
    report = {
        "dataset": {"rows": 10, "columns": 3},
        "statistics": {"height": {"mean": 12.5}},
    }

    output = tmp_path / "report.json"

    result = export_report_json(report, output)

    assert result == output
    assert output.exists()

    with output.open(encoding="utf-8") as file:
        exported = json.load(file)

    assert exported == report


def test_export_creates_parent_directories(tmp_path):
    output = tmp_path / "nested" / "reports" / "report.json"

    export_report_json({"ok": True}, output)

    assert output.exists()


def test_export_numpy_types(tmp_path):
    report = {
        "integer": np.int64(42),
        "float": np.float64(3.14),
        "boolean": np.bool_(True),
        "array": np.array([1, 2, 3]),
    }

    output = tmp_path / "numpy.json"
    export_report_json(report, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["integer"] == 42
    assert result["float"] == pytest.approx(3.14)
    assert result["boolean"] is True
    assert result["array"] == [1, 2, 3]


def test_export_pandas_timestamp(tmp_path):
    report = {"timestamp": pd.Timestamp("2026-01-01")}

    output = tmp_path / "timestamp.json"
    export_report_json(report, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["timestamp"] == "2026-01-01T00:00:00"


def test_export_datetime(tmp_path):
    report = {"date": datetime(2026, 1, 1, 12, 30)}

    output = tmp_path / "datetime.json"
    export_report_json(report, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["date"] == "2026-01-01T12:30:00"


def test_export_path(tmp_path):
    report = {"file": Path("example.csv")}

    output = tmp_path / "path.json"
    export_report_json(report, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["file"] == "example.csv"


def test_export_rejects_invalid_input(tmp_path):
    with pytest.raises(TypeError):
        export_report_json([1, 2, 3], tmp_path / "report.json")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_export_rejects_nonfinite_numbers(tmp_path, value):
    with pytest.raises(ValueError):
        export_report_json(
            {"value": value},
            tmp_path / "report.json",
        )


def test_export_rejects_unsupported_object(tmp_path):
    with pytest.raises(ValueError):
        export_report_json(
            {"object": object()},
            tmp_path / "report.json",
        )


def test_export_does_not_corrupt_existing_file_on_normalization_failure(
    tmp_path,
):
    output = tmp_path / "report.json"
    output.write_text('{"original": true}', encoding="utf-8")

    with pytest.raises(ValueError):
        export_report_json({"bad": float("nan")}, output)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "original": True
    }


def test_export_nested_data(tmp_path):
    report = {
        "metadata": {"author": "BioDataLab"},
        "results": [
            {"mean": np.float64(4.2)},
            {"mean": np.float64(5.7)},
        ],
    }

    output = tmp_path / "nested.json"
    export_report_json(report, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert len(result["results"]) == 2
    assert result["results"][0]["mean"] == pytest.approx(4.2)
