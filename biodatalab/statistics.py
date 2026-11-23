
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
    assert result.mean == 3.0
    assert result.median == 3.0
    assert result.minimum == 1.0
    assert result.maximum == 5.0


def test_analyze_dataset():
    dataframe = pd.DataFrame({
        "height": [10, 20, 30],
        "weight": [2, 4, 6],
    })

    results = analyze_dataset(dataframe)

    assert "height" in results
    assert "weight" in results
    assert results["height"].mean == 20.0


def test_iqr_outliers():
    series = pd.Series([10, 11, 12, 13, 100])

    result = calculate_iqr_outliers(series)

    assert result.iloc[-1] == True


def test_correlation():
    dataframe = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
    })

    result = calculate_correlation(dataframe)

    assert result.loc["x", "y"] == pytest.approx(1.0)


def test_invalid_correlation_method():
    dataframe = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [2, 4, 6],
    })

    with pytest.raises(ValueError):
        calculate_correlation(dataframe, method="invalid")
