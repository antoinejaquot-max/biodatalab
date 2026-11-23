
"""Tests for BioDataLab scientific visualizations."""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from biodatalab.visualization import (
    plot_histogram,
    plot_scatter,
    plot_correlation_matrix,
)


def test_plot_histogram(tmp_path):
    dataframe = pd.DataFrame({"height": [10, 12, 14, 16, 18]})
    output = tmp_path / "histogram.png"

    result = plot_histogram(dataframe, "height", output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_scatter(tmp_path):
    dataframe = pd.DataFrame({
        "height": [10, 12, 14, 16],
        "weight": [20, 24, 28, 32],
    })
    output = tmp_path / "scatter.png"

    result = plot_scatter(dataframe, "height", "weight", output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_correlation_matrix(tmp_path):
    dataframe = pd.DataFrame({
        "height": [10, 12, 14, 16],
        "weight": [20, 24, 28, 32],
        "age": [2, 3, 4, 5],
    })
    output = tmp_path / "correlation.png"

    result = plot_correlation_matrix(dataframe, output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_histogram_invalid_column(tmp_path):
    dataframe = pd.DataFrame({"height": [10, 12, 14]})

    with pytest.raises(ValueError):
        plot_histogram(dataframe, "unknown", tmp_path / "test.png")


def test_scatter_invalid_column(tmp_path):
    dataframe = pd.DataFrame({"height": [10, 12, 14]})

    with pytest.raises(ValueError):
        plot_scatter(dataframe, "height", "unknown", tmp_path / "test.png")


def test_correlation_invalid_method(tmp_path):
    dataframe = pd.DataFrame({
        "height": [10, 12, 14],
        "weight": [20, 24, 28],
    })

    with pytest.raises(ValueError):
        plot_correlation_matrix(
            dataframe,
            tmp_path / "test.png",
            method="invalid",
        )
