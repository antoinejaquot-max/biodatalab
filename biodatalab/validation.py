
"""Tests for dataset validation."""

import pandas as pd
import pytest

from biodatalab.validation import validate_dataframe


def test_valid_dataframe():
    df = pd.DataFrame({
        "height": [10, 20, 30],
        "species": ["A", "B", "C"],
    })

    result = validate_dataframe(
        df,
        {
            "required_columns": ["height", "species"],
            "columns": {
                "height": {"type": "numeric", "min": 0},
                "species": {"type": "string"},
            },
        },
    )

    assert result["valid"] is True
    assert result["summary"]["errors"] == 0


def test_missing_required_column():
    df = pd.DataFrame({"height": [10, 20]})

    result = validate_dataframe(
        df,
        {"required_columns": ["height", "species"]},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "missing_column"
        for error in result["errors"]
    )


def test_numeric_type_validation():
    df = pd.DataFrame({"value": [1, "invalid", 3]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"type": "numeric"}}},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "invalid_type"
        for error in result["errors"]
    )


def test_integer_validation():
    df = pd.DataFrame({"value": [1, 2.5, 3]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"type": "integer"}}},
    )

    assert result["valid"] is False


def test_string_validation():
    df = pd.DataFrame({"value": ["a", 2, "c"]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"type": "string"}}},
    )

    assert result["valid"] is False


def test_boolean_validation():
    df = pd.DataFrame({"value": [True, False, 1]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"type": "boolean"}}},
    )

    assert result["valid"] is False


def test_minimum_value():
    df = pd.DataFrame({"value": [-1, 2, 3]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"min": 0}}},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "out_of_range"
        for error in result["errors"]
    )


def test_maximum_value():
    df = pd.DataFrame({"value": [1, 2, 101]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"max": 100}}},
    )

    assert result["valid"] is False


def test_allowed_values():
    df = pd.DataFrame({"species": ["A", "B", "X"]})

    result = validate_dataframe(
        df,
        {"columns": {"species": {"allowed": ["A", "B"]}}},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "invalid_value"
        for error in result["errors"]
    )


def test_unique_column():
    df = pd.DataFrame({"id": [1, 2, 2, 3]})

    result = validate_dataframe(
        df,
        {"columns": {"id": {"unique": True}}},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "duplicate_values"
        for error in result["errors"]
    )


def test_unique_rows():
    df = pd.DataFrame({
        "a": [1, 2, 2],
        "b": [3, 4, 4],
    })

    result = validate_dataframe(
        df,
        {"unique_rows": True},
    )

    assert result["valid"] is False


def test_non_nullable_column():
    df = pd.DataFrame({"value": [1, None, 3]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"nullable": False}}},
    )

    assert result["valid"] is False
    assert any(
        error["code"] == "null_values"
        for error in result["errors"]
    )


def test_nullable_column():
    df = pd.DataFrame({"value": [1, None, 3]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"nullable": True}}},
    )

    assert result["valid"] is True


def test_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])

    result = validate_dataframe(df)

    assert result["valid"] is True
    assert result["summary"]["total_rows"] == 0


def test_duplicate_column_names():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_non_dataframe_input():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_no_rules():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = validate_dataframe(df)

    assert result["valid"] is True
    assert result["errors"] == []


def test_invalid_range_with_non_numeric_values():
    df = pd.DataFrame({"value": ["a", "b"]})

    result = validate_dataframe(
        df,
        {"columns": {"value": {"min": 0}}},
    )

    assert result["valid"] is True
    assert len(result["warnings"]) == 1
