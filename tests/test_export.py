
"""Tests for BioDataLab JSON export."""

import json
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


def test_export_report_rejects_invalid_input(tmp_path):
    with pytest.raises(TypeError):
        export_report_json([1, 2, 3], tmp_path / "report.json")


def test_export_report_rejects_invalid_json(tmp_path):
    report = {"value": float("nan")}

    with pytest.raises(ValueError):
        export_report_json(report, tmp_path / "report.json")
