"""Clustering utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kmeans_clustering(
    dataframe: pd.DataFrame,
    n_clusters: int = 3,
    max_iter: int = 300,
    random_state: int | None = 42,
    standardize: bool = True,
) -> dict[str, object]:
    """Perform K-means clustering without external dependencies."""

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

    values = dataframe.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("DataFrame must not contain infinite values.")

    if len(values) < 2:
        raise ValueError("At least 2 observations are required.")

    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise TypeError("n_clusters must be an integer.")

    if n_clusters < 1 or n_clusters > len(values):
        raise ValueError("Invalid number of clusters.")

    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an integer.")

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    if standardize:
        means = values.mean(axis=0)
        stds = values.std(axis=0)

        if np.any(stds == 0):
            raise ValueError("Cannot standardize constant columns.")

        values = (values - means) / stds

    rng = np.random.default_rng(random_state)

    centroids = values[
        rng.choice(len(values), size=n_clusters, replace=False)
    ].copy()

    labels = np.zeros(len(values), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(
            values[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        new_labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for cluster in range(n_clusters):
            members = values[new_labels == cluster]

            if len(members) > 0:
                new_centroids[cluster] = members.mean(axis=0)

        if np.array_equal(new_labels, labels) and np.allclose(
            new_centroids, centroids
        ):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    distances = np.linalg.norm(
        values - centroids[labels],
        axis=1,
    )

    inertia = float(np.sum(distances**2))

    return {
        "labels": pd.Series(
            labels,
            index=dataframe.index,
            name="cluster",
        ),
        "centroids": pd.DataFrame(
            centroids,
            columns=dataframe.columns,
        ),
        "inertia": inertia,
        "n_clusters": n_clusters,
        "iterations": _ + 1,
    }
