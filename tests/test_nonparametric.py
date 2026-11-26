
"""Edge-case tests for non-parametric methods."""

import numpy as np
import pytest

from biodatalab.nonparametric import (
    kruskal_wallis,
    mann_whitney,
    wilcoxon_signed_rank,
    spearman_correlation,
    holm_correction,
)


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf, 0, 1, -0.1])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        mann_whitney([1, 2], [3, 4], alpha=alpha)


def test_kruskal_identical_observations():
    result = kruskal_wallis({
        "A": [5, 5, 5],
        "B": [5, 5, 5],
    })
    assert result["statistic"] == 0
    assert result["p_value"] == 1
    assert result["significant"] is False


def test_kruskal_three_identical_groups():
    result = kruskal_wallis({
        "A": [5, 5],
        "B": [5, 5],
        "C": [5, 5],
    })
    assert result["p_value"] == 1


def test_kruskal_rejects_duplicate_string_group_names():
    with pytest.raises(ValueError):
        kruskal_wallis({
            1: [1, 2],
            "1": [3, 4],
        })


def test_wilcoxon_all_differences_zero():
    result = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert result["statistic"] == 0
    assert result["p_value"] == 1
    assert result["significant"] is False


def test_wilcoxon_mismatched_lengths():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1, 2], [1])


def test_spearman_constant_variable():
    with pytest.raises(ValueError):
        spearman_correlation([1, 1, 1], [1, 2, 3])


def test_spearman_too_few_pairs():
    with pytest.raises(ValueError):
        spearman_correlation([1, 2], [2, 3])


def test_holm_adjustment():
    result = holm_correction([0.01, 0.04, 0.03])
    np.testing.assert_allclose(
        result["adjusted_p_values"],
        [0.03, 0.06, 0.06],
    )


def test_holm_rejects_nan():
    with pytest.raises(ValueError):
        holm_correction([0.01, np.nan])


def test_holm_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        holm_correction([0.01, 1.2])
