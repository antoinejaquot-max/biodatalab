
"""Sampling utilities for BioDataLab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_dataframe(
    dataframe: pd.DataFrame,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    """Randomly sample rows from a DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame must not be empty.")

    if n is not None and frac is not None:
        raise ValueError("Specify either n or frac, not both.")

    if n is None and frac is None:
        raise ValueError("Specify either n or frac.")

    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("n must be an integer.")

        if n < 1:
            raise ValueError("n must be at least 1.")

        if not replace and n > len(dataframe):
            raise ValueError(
                "n cannot exceed the number of rows without replacement."
            )

    if frac is not None:
        if not isinstance(frac, (int, float)):
            raise TypeError("frac must be numeric.")

        if not np.isfinite(frac) or frac <= 0:
            raise ValueError("frac must be finite and positive.")

        if not replace and frac > 1:
            raise ValueError("frac cannot exceed 1 without replacement.")

    if n is None and frac is not None:
        n = int(round(len(dataframe) * frac))

        if n < 1:
            raise ValueError("frac produces an empty sample.")

    return dataframe.sample(
        n=n,
        frac=None,
        replace=replace,
        random_state=random_state,
    )


def stratified_sample(
    dataframe: pd.DataFrame,
    stratify_by: str,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Draw a proportional stratified sample from a DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("DataFrame must not be empty.")

    if stratify_by not in dataframe.columns:
        raise ValueError(f"Column '{stratify_by}' not found.")

    if dataframe[stratify_by].isna().any():
        raise ValueError("Stratification column must not contain missing values.")

    if n is not None and frac is not None:
        raise ValueError("Specify either n or frac, not both.")

    if n is None and frac is None:
        raise ValueError("Specify either n or frac.")

    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("n must be an integer.")

        if n < 1 or n > len(dataframe):
            raise ValueError("n must be between 1 and the dataset size.")

    if frac is not None:
        if not isinstance(frac, (int, float)):
            raise TypeError("frac must be numeric.")

        if not np.isfinite(frac) or not 0 < frac <= 1:
            raise ValueError("frac must be finite and in (0, 1].")

    if frac is not None:
        n = int(round(len(dataframe) * frac))

    if n is None or n < 1:
        raise ValueError("Sample size must be at least 1.")

    if n == len(dataframe):
        return dataframe.copy()

    groups = dataframe.groupby(
        stratify_by,
        observed=True,
        sort=False,
    )

    group_sizes = groups.size()
    proportions = group_sizes / len(dataframe)

    allocations = proportions * n
    allocated = np.floor(allocations).astype(int)

    remaining = n - allocated.sum()

    if remaining > 0:
        remainders = (allocations - allocated).sort_values(
            ascending=False,
            kind="stable",
        )

        for index in remainders.index[:remaining]:
            allocated.loc[index] += 1

    rng = np.random.default_rng(random_state)
    sampled_parts = []

    for group_name, group in groups:
        group_n = int(allocated.loc[group_name])

        if group_n == 0:
            continue

        sampled_indices = rng.choice(
            group.index.to_numpy(),
            size=group_n,
            replace=False,
        )

        sampled_parts.append(group.loc[sampled_indices])

    result = pd.concat(sampled_parts)

    return result.sample(
        frac=1,
        random_state=random_state,
    )
