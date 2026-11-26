
import pandas as pd
import pytest

from biodatalab.time_series import analyze_time_series


def test_time_series_analysis():
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "value": [30, 10, 20],
        }
    )

    result = analyze_time_series(df, "date", "value")

    assert list(result["value"]) == [10, 20, 30]
    assert list(result["change"].iloc[1:]) == [10, 10]
    assert list(result["cumulative_change"]) == [0, 10, 20]


def test_percentage_change():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "value": [10, 20],
        }
    )

    result = analyze_time_series(df, "date", "value")

    assert result["percentage_change"].iloc[1] == pytest.approx(100)


def test_missing_observations():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", None],
            "value": [10, None, 30],
        }
    )

    result = analyze_time_series(df, "date", "value")

    assert len(result) == 1


def test_invalid_date():
    df = pd.DataFrame(
        {
            "date": ["invalid"],
            "value": [10],
        }
    )

    with pytest.raises(ValueError):
        analyze_time_series(df, "date", "value")


def test_missing_column():
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1]})

    with pytest.raises(ValueError):
        analyze_time_series(df, "missing", "value")


def test_non_numeric_values():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "value": ["abc"],
        }
    )

    with pytest.raises(TypeError):
        analyze_time_series(df, "date", "value")


def test_empty_dataframe():
    df = pd.DataFrame(
        {
            "date": [],
            "value": [],
        }
    )

    with pytest.raises(TypeError):
        analyze_time_series(df, "date", "value")


def test_duplicate_columns():
    df = pd.DataFrame(
        [[1, 2]],
        columns=["date", "date"],
    )

    with pytest.raises(ValueError):
        analyze_time_series(df, "date", "date")


def test_same_column():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "value": [1],
        }
    )

    with pytest.raises(ValueError):
        analyze_time_series(df, "date", "date")


def test_cumulative_change():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [5, 8, 2],
        }
    )

    result = analyze_time_series(df, "date", "value")

    assert list(result["cumulative_change"]) == [0, 3, -3]
