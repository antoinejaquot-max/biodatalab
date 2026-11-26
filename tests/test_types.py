
"""Tests for column type detection."""

import pandas as pd
import pytest

from biodatalab.types import detect_column_types


def test_detect_numeric_column():
    df = pd.DataFrame(
        {"measurement": [1.5, 2.5, 3.5, 4.5]}
    )

    result = detect_column_types(df)

    assert result["measurement"]["inferred_type"] == "numeric"


def test_detect_categorical_numeric():
    df = pd.DataFrame(
        {"group": [1, 2, 1, 2, 1]}
    )

    result = detect_column_types(df)

    assert result["group"]["inferred_type"] == "categorical_numeric"


def test_detect_boolean():
    df = pd.DataFrame(
        {"active": [True, False, True]}
    )

    result = detect_column_types(df)

    assert result["active"]["inferred_type"] == "boolean"


def test_detect_text():
    df = pd.DataFrame(
        {"description": ["alpha", "beta", "gamma", "delta"]}
    )

    result = detect_column_types(df)

    assert result["description"]["inferred_type"] == "text"


def test_detect_categorical_text():
    df = pd.DataFrame(
        {"group": ["A", "B", "A", "B", "A"]}
    )

    result = detect_column_types(df)

    assert result["group"]["inferred_type"] == "categorical"


def test_detect_numeric_candidate():
    df = pd.DataFrame(
        {
            "measurement": [
                "100",
                "200",
                "300",
                "400",
                "500",
                "600",
                "700",
                "800",
                "900",
                "1000",
                "1100",
                "1200",
                "1300",
                "1400",
                "1500",
                "1600",
                "1700",
                "1800",
                "1900",
                "2000",
                "2100",
                "2200",
            ]
        }
    )

    result = detect_column_types(df)

    assert (
        result["measurement"]["inferred_type"]
        == "numeric_candidate"
    )


def test_detect_missing_values():
    df = pd.DataFrame(
        {"value": [1, None, 3, None]}
    )

    result = detect_column_types(df)

    assert result["value"]["missing_count"] == 2


def test_detect_unique_count():
    df = pd.DataFrame(
        {"value": ["A", "B", "A", None]}
    )

    result = detect_column_types(df)

    assert result["value"]["unique_count"] == 2


def test_empty_dataframe():
    df = pd.DataFrame()

    result = detect_column_types(df)

    assert result == {}


def test_non_dataframe():
    with pytest.raises(TypeError):
        detect_column_types([1, 2, 3])


def test_duplicate_columns():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["a", "a"],
    )

    with pytest.raises(ValueError):
        detect_column_types(df)
