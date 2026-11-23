
"""Automated analysis reports for BioDataLab."""

from dataclasses import asdict

import pandas as pd

from biodatalab.loader import validate_dataset
from biodatalab.statistics import analyze_dataset
from biodatalab.outliers import summarize_outliers


def generate_analysis_report(
    dataframe: pd.DataFrame,
    outlier_method: str = "iqr",
) -> dict:
    """Generate a comprehensive dataset analysis report."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    if outlier_method not in {"iqr", "zscore"}:
        raise ValueError("Method must be iqr or zscore.")

    dataset_info = validate_dataset(dataframe)

    statistics = analyze_dataset(dataframe)

    outliers = summarize_outliers(
        dataframe,
        method=outlier_method,
    )

    return {
        "dataset": dataset_info,
        "statistics": {
            column: asdict(values)
            for column, values in statistics.items()
        },
        "outliers": outliers,
        "outlier_method": outlier_method,
    }
