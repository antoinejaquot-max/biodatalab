
import pandas as pd
import pytest

from biodatalab.cleaning import (
    remove_empty_columns,
    remove_empty_rows,
    fill_missing_numeric,
    remove_duplicate_rows,
    clean_dataset,
)


def test_remove_empty_columns():
    df = pd.DataFrame({
        "height": [10, 20, 30],
        "empty": [None, None, None],
    })

    result = remove_empty_columns(df)

    assert "height" in result.columns
    assert "empty" not in result.columns


def test_remove_empty_rows():
    df = pd.DataFrame({
        "height": [10, None, 30],
        "weight": [2, None, 6],
    })

    result = remove_empty_rows(df)

    assert len(result) == 2


def test_fill_missing_numeric():
    df = pd.DataFrame({
        "height": [10, None, 30],
    })

    result = fill_missing_numeric(df, strategy="mean")

    assert result["height"].isna().sum() == 0
    assert result["height"].iloc[1] == 20


def test_fill_missing_with_median():
    df = pd.DataFrame({
        "height": [10, None, 30, 100],
    })

    result = fill_missing_numeric(df, strategy="median")

    assert result["height"].iloc[1] == 30


def test_fill_missing_with_zero():
    df = pd.DataFrame({
        "height": [10, None, 30],
    })

    result = fill_missing_numeric(df, strategy="zero")

    assert result["height"].iloc[1] == 0


def test_invalid_strategy():
    df = pd.DataFrame({
        "height": [10, None, 30],
    })

    with pytest.raises(ValueError):
        fill_missing_numeric(df, strategy="invalid")


def test_remove_duplicate_rows():
    df = pd.DataFrame({
        "height": [10, 20, 20, 30],
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 3


def test_clean_dataset():
    df = pd.DataFrame({
        "height": [10, 20, 20, None],
        "empty": [None, None, None, None],
    })

    result = clean_dataset(df)

    assert "empty" not in result.columns
    assert len(result) == 2
    assert result["height"].isna().sum() == 0
