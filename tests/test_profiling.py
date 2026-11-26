
"""Tests for BioDataLab profiling."""

import pandas as pd
import pytest

from biodatalab.profiling import profile_dataframe


def test_basic_profile():
    df = pd.DataFrame(
        {
            "height": [170, 180, 160],
            "species": ["A", "B", "A"],
        }
    )

    result = profile_dataframe(df)

    assert result["shape"] == (3, 2)
    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 0


def test_missing_values():
    df = pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": ["x", None, "y"],
        }
    )

    result = profile_dataframe(df)

    assert result["total_missing_values"] == 2
    assert result["columns"]["A"]["missing_count"] == 1
    assert result["columns"]["A"]["missing_percentage"] == pytest.approx(
        100 / 3
    )


def test_numeric_statistics():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

    result = profile_dataframe(df)

    info = result["columns"]["A"]

    assert info["mean"] == 3.0
    assert info["median"] == 3.0
    assert info["min"] == 1.0
    assert info["max"] == 5.0


def test_duplicate_rows():
    df = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": ["x", "x", "y"],
        }
    )

    result = profile_dataframe(df)

    assert result["duplicate_rows"] == 1


def test_empty_dataframe():
    df = pd.DataFrame()

    result = profile_dataframe(df)

    assert result["shape"] == (0, 0)
    assert result["total_missing_percentage"] == 0.0


def test_empty_numeric_column():
    df = pd.DataFrame({"A": [None, None]})

    result = profile_dataframe(df)

    assert result["columns"]["A"]["mean"] is None


def test_invalid_input():
    with pytest.raises(TypeError):
        profile_dataframe([1, 2, 3])


def test_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["A", "A"])

    with pytest.raises(ValueError):
        profile_dataframe(df)
