
"""Scientific visualization utilities for BioDataLab."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(
    dataframe: pd.DataFrame,
    column: str,
    output_path: str | Path,
    bins: int = 20,
) -> Path:
    """Create and save a histogram for a numeric column."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if column not in dataframe.columns:
        raise ValueError(f"Unknown column: {column}")

    if bins <= 0:
        raise ValueError("Bins must be positive.")

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if values.empty:
        raise ValueError("The selected column has no numeric values.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def plot_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: str | Path,
) -> Path:
    """Create and save a scatter plot between two numeric columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    for column in (x_column, y_column):
        if column not in dataframe.columns:
            raise ValueError(f"Unknown column: {column}")

    x = pd.to_numeric(dataframe[x_column], errors="coerce")
    y = pd.to_numeric(dataframe[y_column], errors="coerce")
    valid = x.notna() & y.notna()

    if not valid.any():
        raise ValueError("No valid numeric pairs are available.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(x[valid], y[valid])
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def plot_correlation_matrix(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    method: str = "pearson",
) -> Path:
    """Create and save a numeric correlation matrix."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Unsupported correlation method.")

    numeric = dataframe.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    correlation = numeric.corr(method=method)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation.columns)
    ax.set_title(f"Correlation matrix ({method})")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path
