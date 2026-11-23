
"""Tests for BioDataLab statistical functions."""

import pandas as pd
import pytest

from biodatalab.statistics import (
    analyze_column,
    analyze_dataset,
    calculate_iqr_outliers,
    calculate_correlation,
)


def test_analyze_column():
    series = pd.Series([1, 2, 3, 4, 5])

    result = analyze_column(series)

    assert result.count == 5
    assert result.missing == 0
    assert result.mean == 3.0
    assert result.median == 3.0
    assert result.minimum == 1.0
    assert result.maximum == 5.0
    assert result.q1 == 2.0
    assert result.q3 == 4.0


def test_analyze_column_with_missing_values():
    series = pd.Series([10, 20, None, 30])

    result = analyze_column(series)

    assert result.count == 3
    assert result.missing == 1
    assert result.mean == 20.0


def test_analyze_column_with_no_numeric_values():
    series = pd.Series(["plant", "flower", None])

    result = analyze_column(series)

    assert result.count == 0
    assert result.mean is None


def test_analyze_dataset():
    dataframe = pd.DataFrame({
        "height": [10, 20, 30],
        "weight": [2, 4, 6],
        "species": ["A", "B", "C"],
    })

    results = analyze_dataset(dataframe)

    assert "height" in results
    assert "weight" in results
    assert "species" not in results
    assert results["height"].mean == 20.0


def test_analyze_empty_dataset():
    dataframe = pd.DataFrame()

    with pytest.raises(ValueError):
        analyze_dataset(dataframe)


def test_iqr_outliers():
    series = pd.Series([10, 11, 12, 13, 100])

    result = calculate_iqr_outliers(series)

    assert result.iloc[-1]
    assert result.sum() == 1


def test_iqr_invalid_multiplier():
    series = pd.Series([10, 20, 30])

    with pytest.raises(ValueError):
        calculate_iqr_outliers(series, multiplier=0)


def test_correlation():
    dataframe = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
    })

    result = calculate_correlation(dataframe)

    assert result.loc["x", "y"] == pytest.approx(1.0)


def test_spearman_correlation():
    dataframe = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [4, 3, 2, 1],
    })

    result = calculate_correlation(
        dataframe,
        method="spearman",
    )

    assert result.loc["x", "y"] == pytest.approx(-1.0)


def test_invalid_correlation_method():
    dataframe = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [2, 4, 6],
    })

    with pytest.raises(ValueError):
        calculate_correlation(
            dataframe,
            method="invalid",
        )


def test_correlation_requires_two_columns():
    dataframe = pd.DataFrame({
        "height": [10, 20, 30],
    })

    with pytest.raises(ValueError):
        calculate_correlation(dataframe)
