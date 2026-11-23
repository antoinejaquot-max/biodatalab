
"""Tests for automated BioDataLab reports."""

import pandas as pd
import pytest

from biodatalab.report import generate_analysis_report


def test_generate_analysis_report():
    dataframe = pd.DataFrame({
        "height": [10, 11, 12, 13, 100],
        "weight": [2, 3, 4, 5, 6],
        "species": ["Oak", "Oak", "Pine", "Pine", "Oak"],
    })

    report = generate_analysis_report(dataframe)

    assert report["dataset"]["rows"] == 5
    assert report["dataset"]["columns"] == 3
    assert "height" in report["statistics"]
    assert report["statistics"]["height"]["mean"] == pytest.approx(29.2)
    assert report["outliers"]["height"]["count"] == 1
    assert report["outlier_method"] == "iqr"


def test_generate_report_with_zscore():
    dataframe = pd.DataFrame({
        "height": [10, 11, 12, 13, 100],
        "weight": [2, 3, 4, 5, 6],
    })

    report = generate_analysis_report(dataframe, outlier_method="zscore")

    assert report["outlier_method"] == "zscore"


def test_report_empty_dataframe():
    with pytest.raises(ValueError):
        generate_analysis_report(pd.DataFrame())


def test_report_invalid_method():
    dataframe = pd.DataFrame({"height": [10, 20, 30]})

    with pytest.raises(ValueError):
        generate_analysis_report(dataframe, outlier_method="invalid")


def test_report_rejects_non_dataframe():
    with pytest.raises(TypeError):
        generate_analysis_report([1, 2, 3])
