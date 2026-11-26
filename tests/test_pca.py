import numpy as np
import pandas as pd
import pytest

from biodatalab.pca import perform_pca


def create_dataframe():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],
            "C": [5, 3, 4, 2, 1],
        }
    )


def test_pca_basic():
    result = perform_pca(create_dataframe())

    assert "scores" in result
    assert "loadings" in result
    assert "explained_variance" in result


def test_pca_scores_shape():
    result = perform_pca(
        create_dataframe(),
        n_components=2,
    )

    assert result["scores"].shape == (5, 2)


def test_pca_loadings_shape():
    result = perform_pca(
        create_dataframe(),
        n_components=2,
    )

    assert result["loadings"].shape == (3, 2)


def test_pca_variance_ratio():
    result = perform_pca(create_dataframe())

    ratios = result["explained_variance"][
        "explained_variance_ratio"
    ]

    assert np.all(ratios >= 0)
    assert np.all(ratios <= 1)
    assert ratios.sum() <= 1 + 1e-10


def test_pca_cumulative_variance():
    result = perform_pca(create_dataframe())

    cumulative = result["explained_variance"][
        "cumulative_variance_ratio"
    ]

    assert np.all(np.diff(cumulative) >= -1e-10)


def test_pca_standardization():
    result = perform_pca(
        create_dataframe(),
        standardize=True,
    )

    assert result["standardized"] is True


def test_pca_without_standardization():
    result = perform_pca(
        create_dataframe(),
        standardize=False,
    )

    assert result["standardized"] is False


def test_pca_invalid_components():
    with pytest.raises(ValueError):
        perform_pca(
            create_dataframe(),
            n_components=10,
        )


def test_pca_invalid_component_type():
    with pytest.raises(TypeError):
        perform_pca(
            create_dataframe(),
            n_components=1.5,
        )


def test_pca_missing_values():
    dataframe = create_dataframe()
    dataframe.loc[0, "A"] = np.nan

    with pytest.raises(ValueError):
        perform_pca(dataframe)


def test_pca_non_numeric():
    dataframe = create_dataframe()
    dataframe["D"] = ["a", "b", "c", "d", "e"]

    with pytest.raises(TypeError):
        perform_pca(dataframe)


def test_pca_constant_column_standardized():
    dataframe = create_dataframe()
    dataframe["D"] = 5

    with pytest.raises(ValueError):
        perform_pca(dataframe)


def test_pca_empty_dataframe():
    with pytest.raises(ValueError):
        perform_pca(pd.DataFrame())


def test_pca_single_observation():
    dataframe = pd.DataFrame({"A": [1], "B": [2]})

    with pytest.raises(ValueError):
        perform_pca(dataframe)


def test_pca_duplicate_columns():
    dataframe = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["A", "A"],
    )

    with pytest.raises(ValueError):
        perform_pca(dataframe)
