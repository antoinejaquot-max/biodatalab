
import pandas as pd
import pytest

from biodatalab.correlation import (
    correlation_matrix,
    correlation_pairs,
)


def test_correlation_matrix():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [2, 4, 6, 8],
            "C": [4, 3, 2, 1],
        }
    )

    result = correlation_matrix(df)

    assert result.shape == (3, 3)
    assert result.loc["A", "B"] == pytest.approx(1.0)
    assert result.loc["A", "C"] == pytest.approx(-1.0)


def test_spearman_method():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
        }
    )

    result = correlation_matrix(df, method="spearman")

    assert result.loc["A", "B"] == pytest.approx(1.0)


def test_kendall_method():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [4, 3, 2, 1],
        }
    )

    result = correlation_matrix(df, method="kendall")

    assert result.loc["A", "B"] == pytest.approx(-1.0)


def test_constant_columns_removed():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [5, 5, 5],
            "C": [3, 2, 1],
        }
    )

    result = correlation_matrix(df)

    assert "B" not in result.columns
    assert result.shape == (2, 2)


def test_missing_values():
    df = pd.DataFrame(
        {
            "A": [1, 2, None, 4],
            "B": [2, 4, 6, 8],
        }
    )

    result = correlation_matrix(df)

    assert pd.notna(result.loc["A", "B"])


def test_correlation_pairs_sorted():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [2, 4, 6, 8],
            "C": [4, 3, 2, 1],
        }
    )

    result = correlation_pairs(df)

    assert len(result) == 3
    assert result.iloc[0]["absolute_correlation"] == pytest.approx(1.0)
    assert "variable_1" in result.columns
    assert "variable_2" in result.columns


def test_invalid_method():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    with pytest.raises(ValueError):
        correlation_matrix(df, method="invalid")


def test_no_numeric_columns():
    df = pd.DataFrame({"A": ["a", "b", "c"]})

    with pytest.raises(ValueError):
        correlation_matrix(df)


def test_one_non_constant_column():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [5, 5, 5],
        }
    )

    with pytest.raises(ValueError):
        correlation_matrix(df)


def test_invalid_input():
    with pytest.raises(TypeError):
        correlation_matrix("invalid")


def test_duplicate_columns():
    df = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["A", "A"],
    )

    with pytest.raises(ValueError):
        correlation_matrix(df)
