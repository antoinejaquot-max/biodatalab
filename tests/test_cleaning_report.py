
"""Tests for cleaning audit utilities."""

import pandas as pd

from biodatalab.cleaning_report import (
    cleaning_report_dataframe,
    generate_cleaning_report,
)


def test_report_detects_removed_rows_and_columns():
    original = pd.DataFrame(
        {
            "height": [10, 20, 30],
            "unused": [None, None, None],
        },
        index=[0, 1, 2],
    )

    cleaned = pd.DataFrame(
        {"height": [10, 20]},
        index=[0, 1],
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["changes"]["rows_removed"] == 1
    assert report["changes"]["columns_removed"] == [
        "unused"
    ]


def test_report_detects_missing_values():
    original = pd.DataFrame(
        {"values": [1, None, 3]}
    )

    cleaned = pd.DataFrame(
        {"values": [1, 2, 3]}
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["missing_values"]["before"]["total"] == 1
    assert report["missing_values"]["after"]["total"] == 0
    assert report["missing_values"]["reduction"] == 1


def test_report_detects_changed_cells():
    original = pd.DataFrame(
        {"values": [1, 2, 3]}
    )

    cleaned = pd.DataFrame(
        {"values": [1, 5, 3]}
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["changes"]["cells_changed"] == 1


def test_report_detects_duplicate_rows():
    original = pd.DataFrame(
        {"values": [1, 1, 2]}
    )

    cleaned = pd.DataFrame(
        {"values": [1, 2]}
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["duplicates"]["before"] == 1
    assert report["duplicates"]["after"] == 0


def test_report_dataframe():
    original = pd.DataFrame(
        {"values": [1, None, 3]}
    )

    cleaned = pd.DataFrame(
        {"values": [1, 2, 3]}
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    summary = cleaning_report_dataframe(report)

    assert isinstance(summary, pd.DataFrame)
    assert "Metric" in summary.columns
    assert "Before" in summary.columns
    assert "After" in summary.columns


def test_report_handles_empty_dataframes():
    original = pd.DataFrame()
    cleaned = pd.DataFrame()

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["summary"]["original_rows"] == 0
    assert report["changes"]["cells_changed"] == 0


def test_report_handles_duplicate_indices():
    original = pd.DataFrame(
        {"values": [1, 2]},
        index=[0, 0],
    )

    cleaned = pd.DataFrame(
        {"values": [1, 3]},
        index=[0, 0],
    )

    report = generate_cleaning_report(
        original,
        cleaned,
    )

    assert report["changes"]["cell_comparison_method"] == (
        "position"
    )
    assert report["changes"]["cells_changed"] == 1
