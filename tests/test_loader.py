
"""Tests for BioDataLab loading and dataset validation."""

import numpy as np
import pandas as pd
import pytest

from biodatalab.loader import load_dataset, validate_dataset


def test_load_csv(tmp_path):
    file = tmp_path / "plants.csv"
    file.write_text(
        "plant,height\nP01,12\nP02,15\n",
        encoding="utf-8",
    )

    result = load_dataset(file)

    assert result.shape == (2, 2)
    assert result["height"].tolist() == [12, 15]


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("missing.csv")


def test_unsupported_format(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(file)


def test_empty_dataset(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("name,height\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(file)


def test_validate_dataset():
    df = pd.DataFrame({
        "height": [10, 15, 15],
        "weight": [2, None, None],
    })

    result = validate_dataset(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_values"] == 2
    assert result["duplicate_rows"] == 1


def test_duplicate_csv_headers_rejected(tmp_path):
    file = tmp_path / "duplicates.csv"
    file.write_text("height,height\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(file)


def test_blank_csv_header_rejected(tmp_path):
    file = tmp_path / "blank.csv"
    file.write_text("height,,weight\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(file)


def test_whitespace_column_names_trimmed(tmp_path):
    file = tmp_path / "spaces.csv"
    file.write_text(" height , weight \n1,2\n", encoding="utf-8")

    result = load_dataset(file)

    assert list(result.columns) == ["height", "weight"]


def test_invalid_separator():
    with pytest.raises(ValueError):
        load_dataset("plants.csv", separator=";;")


def test_validate_column_details():
    df = pd.DataFrame({
        "height": [10, 15, 20],
        "species": ["A", "A", "B"],
    })

    result = validate_dataset(df)

    assert result["column_details"]["height"]["numeric_candidate"] is True
    assert result["column_details"]["species"]["numeric_candidate"] is False
    assert result["column_details"]["height"]["missing_values"] == 0


def test_empty_column_detection():
    df = pd.DataFrame({
        "height": [1, 2, 3],
        "empty": [None, None, None],
    })

    result = validate_dataset(df)

    assert "empty" in result["empty_columns"]


def test_constant_column_detection():
    df = pd.DataFrame({
        "constant": [5, 5, 5],
        "variable": [1, 2, 3],
    })

    result = validate_dataset(df)

    assert "constant" in result["constant_columns"]
    assert "variable" not in result["constant_columns"]


def test_nonfinite_numeric_detection():
    df = pd.DataFrame({
        "values": [1.0, np.inf, -np.inf, 3.0],
    })

    result = validate_dataset(df)

    assert result["column_details"]["values"]["non_finite_numeric_values"] == 2


def test_validate_dataset_wrong_type():
    with pytest.raises(TypeError):
        validate_dataset([1, 2, 3])


def test_validate_dataset_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["x", "x"])

    with pytest.raises(ValueError):
        validate_dataset(df)
