
import pandas as pd
import pytest

from biodatalab.loader import (
    load_dataset,
    validate_dataset,
)


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
