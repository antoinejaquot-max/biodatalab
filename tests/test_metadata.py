
"""Tests for BioDataLab metadata."""

import pandas as pd
import pytest

from biodatalab.metadata import generate_dataset_metadata


def test_metadata_structure():
    df = pd.DataFrame({"A": [1, 2, 3]})

    result = generate_dataset_metadata(df)

    assert "dataset_fingerprint" in result
    assert "python_version" in result
    assert "pandas_version" in result
    assert "biodatalab_version" in result


def test_fingerprint_is_deterministic():
    df = pd.DataFrame({"A": [1, 2, 3]})

    first = generate_dataset_metadata(df)
    second = generate_dataset_metadata(df)

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]


def test_fingerprint_changes_with_data():
    first_df = pd.DataFrame({"A": [1, 2, 3]})
    second_df = pd.DataFrame({"A": [1, 2, 4]})

    first = generate_dataset_metadata(first_df)
    second = generate_dataset_metadata(second_df)

    assert first["dataset_fingerprint"] != second["dataset_fingerprint"]


def test_fingerprint_changes_with_schema():
    first_df = pd.DataFrame({"A": [1, 2]})
    second_df = pd.DataFrame({"B": [1, 2]})

    first = generate_dataset_metadata(first_df)
    second = generate_dataset_metadata(second_df)

    assert first["dataset_fingerprint"] != second["dataset_fingerprint"]


def test_metadata_shape():
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": ["x", "y"],
        }
    )

    result = generate_dataset_metadata(df)

    assert result["dataset_shape"] == (2, 2)
    assert result["column_names"] == ["A", "B"]


def test_invalid_input():
    with pytest.raises(TypeError):
        generate_dataset_metadata([1, 2, 3])
