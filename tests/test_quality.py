
import numpy as np
import pandas as pd
import pytest

from biodatalab.quality import assess_data_quality


def test_quality_summary():
    dataframe = pd.DataFrame(
        {
            "height": [10, 20, 30],
            "weight": [1, None, 3],
        }
    )

    report = assess_data_quality(dataframe)

    assert report["summary"]["rows"] == 3
    assert report["summary"]["columns"] == 2
    assert report["summary"]["missing_cells"] == 1
    assert report["summary"]["completeness_percentage"] == pytest.approx(
        83.333333
    )


def test_quality_detects_duplicate_rows():
    dataframe = pd.DataFrame(
        {"a": [1, 1, 2], "b": [3, 3, 4]}
    )

    report = assess_data_quality(dataframe)

    assert report["summary"]["duplicate_rows"] == 1
    assert "Dataset contains duplicate rows." in report["warnings"]


def test_quality_detects_empty_columns():
    dataframe = pd.DataFrame(
        {"a": [None, None], "b": [1, 2]}
    )

    report = assess_data_quality(dataframe)

    assert report["columns"]["a"]["empty"] is True
    assert "Dataset contains entirely empty columns." in report["warnings"]


def test_quality_detects_constant_columns():
    dataframe = pd.DataFrame(
        {"a": [5, 5, 5], "b": [1, 2, 3]}
    )

    report = assess_data_quality(dataframe)

    assert report["columns"]["a"]["constant"] is True
    assert report["columns"]["b"]["constant"] is False


def test_quality_detects_non_finite_values():
    dataframe = pd.DataFrame(
        {"a": [1.0, np.inf, -np.inf, 2.0]}
    )

    report = assess_data_quality(dataframe)

    assert report["non_finite_values"]["a"] == 2
    assert (
        "Dataset contains non-finite numeric values."
        in report["warnings"]
    )


def test_quality_handles_empty_dataframe():
    dataframe = pd.DataFrame()

    report = assess_data_quality(dataframe)

    assert report["summary"]["rows"] == 0
    assert report["summary"]["columns"] == 0
    assert report["summary"]["completeness_percentage"] == 100.0


def test_quality_handles_empty_rows():
    dataframe = pd.DataFrame(columns=["a", "b"])

    report = assess_data_quality(dataframe)

    assert report["summary"]["rows"] == 0
    assert report["summary"]["columns"] == 2
    assert report["columns"]["a"]["empty"] is True


def test_quality_rejects_non_dataframe():
    with pytest.raises(TypeError):
        assess_data_quality([1, 2, 3])


def test_quality_rejects_duplicate_columns():
    dataframe = pd.DataFrame(
        [[1, 2]],
        columns=["a", "a"],
    )

    with pytest.raises(ValueError):
        assess_data_quality(dataframe)


def test_quality_missing_percentage():
    dataframe = pd.DataFrame(
        {"a": [1, None, 3, None]}
    )

    report = assess_data_quality(dataframe)

    assert report["summary"]["missing_percentage"] == 50.0
    assert report["columns"]["a"]["missing_percentage"] == 50.0
