
"""Tests for dataframe validation."""

import pandas as pd
import pytest

from biodatalab.validation import validate_dataframe


def test_valid_dataframe():
    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "value": [1, 2],
        }
    )

    result = validate_dataframe(df)

    assert result is not None


def test_empty_dataframe():
    df = pd.DataFrame()

    result = validate_dataframe(df)

    assert result is not None


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_dataframe_with_missing_values():
    df = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": ["x", "y", "z"],
        }
    )

    result = validate_dataframe(df)

    assert result is not None


def test_dataframe_with_duplicate_rows():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": ["x", "x", "y"],
        }
    )

    result = validate_dataframe(df)

    assert result is not None


def test_dataframe_with_duplicate_columns():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["a", "a"],
    )

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_dataframe_with_mixed_types():
    df = pd.DataFrame(
        {
            "value": [1, "two", 3.0],
        }
    )

    result = validate_dataframe(df)

    assert result is not None


def test_dataframe_with_non_unique_index():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
        },
        index=[0, 0, 1],
    )

    result = validate_dataframe(df)

    assert result is not None
