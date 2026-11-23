
"""Tests for BioDataLab outlier detection."""

import pandas as pd
import pytest

from biodatalab.outliers import (
    detect_iqr,
    detect_zscore,
    summarize_outliers,
    extract_outliers,
)


def test_detect_iqr():
    series = pd.Series([10, 11, 12, 13, 100])
    result = detect_iqr(series)

    assert result.iloc[-1]
    assert result.sum() == 1


def test_detect_zscore():
    series = pd.Series([10, 10, 10, 10, 100])
    result = detect_zscore(series, threshold=1.5)

    assert result.iloc[-1]


def test_detect_iqr_invalid_multiplier():
    with pytest.raises(ValueError):
        detect_iqr(pd.Series([1, 2, 3]), multiplier=0)


def test_detect_zscore_invalid_threshold():
    with pytest.raises(ValueError):
        detect_zscore(pd.Series([1, 2, 3]), threshold=0)


def test_summarize_outliers():
    dataframe = pd.DataFrame({
        "height": [10, 11, 12, 13, 100],
        "weight": [2, 3, 4, 5, 6],
    })

    result = summarize_outliers(dataframe)

    assert result["height"]["count"] == 1
    assert result["height"]["values"] == [100]


def test_summarize_invalid_method():
    dataframe = pd.DataFrame({"height": [1, 2, 3]})

    with pytest.raises(ValueError):
        summarize_outliers(dataframe, method="invalid")


def test_extract_outliers():
    dataframe = pd.DataFrame({
        "height": [10, 11, 12, 13, 100],
        "weight": [2, 3, 4, 5, 6],
    })

    result = extract_outliers(dataframe)

    assert len(result) == 1
    assert result.iloc[0]["height"] == 100


def test_extract_outliers_preserves_original():
    dataframe = pd.DataFrame({
        "height": [10, 11, 12, 13, 100],
    })

    original = dataframe.copy()
    extract_outliers(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_extract_outliers_invalid_method():
    dataframe = pd.DataFrame({"height": [1, 2, 3]})

    with pytest.raises(ValueError):
        extract_outliers(dataframe, method="invalid")
