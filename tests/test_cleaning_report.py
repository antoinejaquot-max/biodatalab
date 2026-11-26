
import pandas as pd
import pytest

from biodatalab.cleaning_report import (
    cleaning_report_dataframe,
    generate_cleaning_report,
)


def test_report_detects_removed_rows():
    original = pd.DataFrame(
        {"value": [1, 2, 3]},
        index=[0, 1, 2],
    )
    cleaned = original.loc[[0, 2]]

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["rows_removed"] == 1
    assert report["changes"]["rows_added"] == 0


def test_report_detects_added_columns():
    original = pd.DataFrame({"value": [1, 2]})
    cleaned = original.assign(extra=[3, 4])

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["columns_added"] == ["extra"]
    assert report["changes"]["columns_removed"] == []


def test_report_detects_removed_columns():
    original = pd.DataFrame(
        {"value": [1, 2], "extra": [3, 4]}
    )
    cleaned = original[["value"]]

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["columns_removed"] == ["extra"]


def test_report_detects_changed_cells():
    original = pd.DataFrame({"value": [1, 2, 3]})
    cleaned = pd.DataFrame({"value": [1, 5, 3]})

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["cells_changed"] == 1
    assert report["changes"]["comparable_cells"] == 3


def test_report_handles_nan_equality():
    original = pd.DataFrame({"value": [1, None, 3]})
    cleaned = pd.DataFrame({"value": [1, None, 3]})

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["cells_changed"] == 0


def test_report_detects_missing_value_reduction():
    original = pd.DataFrame(
        {"a": [1, None, None], "b": [4, 5, None]}
    )
    cleaned = original.fillna(0)

    report = generate_cleaning_report(original, cleaned)

    assert report["missing_values"]["before"]["total"] == 3
    assert report["missing_values"]["after"]["total"] == 0
    assert report["missing_values"]["reduction"] == 3


def test_report_detects_duplicate_reduction():
    original = pd.DataFrame({"a": [1, 1, 2]})
    cleaned = original.drop_duplicates()

    report = generate_cleaning_report(original, cleaned)

    assert report["duplicates"]["before"] == 1
    assert report["duplicates"]["after"] == 0


def test_report_handles_empty_dataframes():
    original = pd.DataFrame()
    cleaned = pd.DataFrame()

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["cells_changed"] == 0
    assert report["summary"]["original_rows"] == 0
    assert report["summary"]["cleaned_rows"] == 0


def test_report_dataframe_conversion():
    original = pd.DataFrame({"a": [1, None, 3]})
    cleaned = original.fillna(0)

    report = generate_cleaning_report(original, cleaned)
    summary = cleaning_report_dataframe(report)

    assert isinstance(summary, pd.DataFrame)
    assert "Metric" in summary.columns
    assert "Before" in summary.columns
    assert "After" in summary.columns


def test_report_rejects_non_dataframe():
    with pytest.raises(TypeError):
        generate_cleaning_report([], pd.DataFrame())


def test_report_rejects_duplicate_columns():
    original = pd.DataFrame([[1, 2]], columns=["a", "a"])
    cleaned = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError):
        generate_cleaning_report(original, cleaned)


def test_report_handles_different_indexes():
    original = pd.DataFrame(
        {"a": [1, 2]},
        index=[10, 20],
    )
    cleaned = pd.DataFrame(
        {"a": [1, 2]},
        index=[30, 40],
    )

    report = generate_cleaning_report(original, cleaned)

    assert report["changes"]["cells_changed"] == 0
    assert report["changes"]["comparable_cells"] == 2
