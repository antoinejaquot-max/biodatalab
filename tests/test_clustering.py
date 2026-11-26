import numpy as np
import pandas as pd
import pytest

from biodatalab.clustering import kmeans_clustering


def create_dataframe():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 10, 11, 12],
            "B": [1, 2, 3, 10, 11, 12],
        }
    )


def test_kmeans_basic():
    result = kmeans_clustering(create_dataframe())

    assert "labels" in result
    assert "centroids" in result
    assert "inertia" in result


def test_kmeans_labels_shape():
    result = kmeans_clustering(create_dataframe())

    assert len(result["labels"]) == 6


def test_kmeans_centroids_shape():
    result = kmeans_clustering(
        create_dataframe(),
        n_clusters=2,
    )

    assert result["centroids"].shape == (2, 2)


def test_kmeans_inertia_non_negative():
    result = kmeans_clustering(create_dataframe())

    assert result["inertia"] >= 0


def test_kmeans_reproducibility():
    first = kmeans_clustering(
        create_dataframe(),
        random_state=42,
    )

    second = kmeans_clustering(
        create_dataframe(),
        random_state=42,
    )

    assert np.array_equal(
        first["labels"].to_numpy(),
        second["labels"].to_numpy(),
    )


def test_kmeans_custom_clusters():
    result = kmeans_clustering(
        create_dataframe(),
        n_clusters=3,
    )

    assert result["n_clusters"] == 3


def test_kmeans_without_standardization():
    result = kmeans_clustering(
        create_dataframe(),
        standardize=False,
    )

    assert result["inertia"] >= 0


def test_kmeans_invalid_clusters():
    with pytest.raises(ValueError):
        kmeans_clustering(
            create_dataframe(),
            n_clusters=0,
        )


def test_kmeans_too_many_clusters():
    with pytest.raises(ValueError):
        kmeans_clustering(
            create_dataframe(),
            n_clusters=10,
        )


def test_kmeans_invalid_iterations():
    with pytest.raises(ValueError):
        kmeans_clustering(
            create_dataframe(),
            max_iter=0,
        )


def test_kmeans_missing_values():
    dataframe = create_dataframe()
    dataframe.loc[0, "A"] = np.nan

    with pytest.raises(ValueError):
        kmeans_clustering(dataframe)


def test_kmeans_non_numeric():
    dataframe = create_dataframe()
    dataframe["C"] = ["a", "b", "c", "d", "e", "f"]

    with pytest.raises(TypeError):
        kmeans_clustering(dataframe)


def test_kmeans_constant_column_standardized():
    dataframe = create_dataframe()
    dataframe["C"] = 5

    with pytest.raises(ValueError):
        kmeans_clustering(dataframe)


def test_kmeans_empty_dataframe():
    with pytest.raises(ValueError):
        kmeans_clustering(pd.DataFrame())


def test_kmeans_duplicate_columns():
    dataframe = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["A", "A"],
    )

    with pytest.raises(ValueError):
        kmeans_clustering(dataframe)
