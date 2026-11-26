"""Principal component analysis utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def perform_pca(
    dataframe: pd.DataFrame,
    n_components: int | None = None,
    standardize: bool = True,
) -> dict[str, object]:
    """Perform PCA using singular value decomposition."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame must not be empty.")

    if dataframe.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in dataframe.dtypes
    ):
        raise TypeError("All columns must be numeric.")

    if dataframe.isna().any().any():
        raise ValueError("DataFrame must not contain missing values.")

    try:
        values = dataframe.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("DataFrame must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError("DataFrame must not contain infinite values.")

    n_samples, n_features = values.shape

    if n_samples < 2:
        raise ValueError("At least 2 observations are required.")

    if n_features < 1:
        raise ValueError("At least 1 feature is required.")

    if n_components is None:
        n_components = min(n_samples, n_features)

    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise TypeError("n_components must be an integer.")

    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            "n_components must be between 1 and "
            "min(number of observations, number of features)."
        )

    means = values.mean(axis=0)
    centered = values - means

    if standardize:
        stds = values.std(axis=0, ddof=1)

        if np.any(stds == 0):
            raise ValueError(
                "Cannot standardize columns with zero variance."
            )

        transformed = centered / stds
    else:
        transformed = centered

    _, singular_values, vt = np.linalg.svd(
        transformed,
        full_matrices=False,
    )

    components = vt[:n_components]

    scores = transformed @ components.T

    total_variance = np.sum(singular_values**2) / (n_samples - 1)

    if total_variance == 0:
        raise ValueError("Data must contain non-zero variance.")

    explained_variance = (
        singular_values[:n_components] ** 2 / (n_samples - 1)
    )

    explained_variance_ratio = (
        singular_values[:n_components] ** 2
        / np.sum(singular_values**2)
    )

    component_names = [
        f"PC{i + 1}" for i in range(n_components)
    ]

    scores_df = pd.DataFrame(
        scores,
        index=dataframe.index,
        columns=component_names,
    )

    loadings_df = pd.DataFrame(
        components.T,
        index=dataframe.columns,
        columns=component_names,
    )

    variance_df = pd.DataFrame(
        {
            "explained_variance": explained_variance,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": np.cumsum(
                explained_variance_ratio
            ),
        },
        index=component_names,
    )

    return {
        "scores": scores_df,
        "loadings": loadings_df,
        "explained_variance": variance_df,
        "n_components": n_components,
        "standardized": standardize,
    }
