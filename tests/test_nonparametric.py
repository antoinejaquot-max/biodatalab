
"""Tests for non-parametric statistical methods."""

import numpy as np
import pytest

from biodatalab.nonparametric import (
    mann_whitney,
    wilcoxon_signed_rank,
    kruskal_wallis,
    spearman_correlation,
    holm_correction,
)


# --------------------------------------------------
# Mann-Whitney U
# --------------------------------------------------

def test_mann_whitney_detects_difference():
    result = mann_whitney(
        [1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14],
    )

    assert result["test"] == "Mann-Whitney U"
    assert 0 <= result["p_value"] <= 1
    assert result["significant"] is True


def test_mann_whitney_sample_sizes():
    result = mann_whitney(
        [1, 2, 3],
        [4, 5, 6, 7],
    )

    assert result["sample_sizes"] == [3, 4]


def test_mann_whitney_invalid_alternative():
    with pytest.raises(ValueError):
        mann_whitney(
            [1, 2, 3],
            [4, 5, 6],
            alternative="invalid",
        )


def test_mann_whitney_invalid_alpha():
    with pytest.raises(ValueError):
        mann_whitney([1, 2], [3, 4], alpha=1.5)


def test_mann_whitney_missing_values():
    result = mann_whitney(
        [1, 2, np.nan, 3],
        [4, 5, 6],
    )

    assert result["sample_sizes"] == [3, 3]


# --------------------------------------------------
# Wilcoxon signed-rank
# --------------------------------------------------

def test_wilcoxon_detects_paired_difference():
    result = wilcoxon_signed_rank(
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 10],
    )

    assert result["test"] == "Wilcoxon signed-rank"
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_wilcoxon_equal_samples():
    values = [1, 2, 3, 4]

    result = wilcoxon_signed_rank(values, values)

    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_wilcoxon_unequal_lengths():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank(
            [1, 2, 3],
            [2, 3],
        )


def test_wilcoxon_invalid_alternative():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank(
            [1, 2, 3],
            [2, 3, 4],
            alternative="invalid",
        )


def test_wilcoxon_missing_values():
    result = wilcoxon_signed_rank(
        [1, np.nan, 3, 4],
        [2, 3, 4, 5],
    )

    assert result["sample_size"] == 3


# --------------------------------------------------
# Kruskal-Wallis
# --------------------------------------------------

def test_kruskal_detects_group_difference():
    result = kruskal_wallis({
        "Control": [1, 2, 3, 4, 5],
        "Treatment_A": [10, 11, 12, 13, 14],
        "Treatment_B": [20, 21, 22, 23, 24],
    })

    assert result["test"] == "Kruskal-Wallis"
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_kruskal_group_sizes():
    result = kruskal_wallis({
        "A": [1, 2, 3],
        "B": [4, 5, 6, 7],
    })

    assert result["groups"]["A"]["n"] == 3
    assert result["groups"]["B"]["n"] == 4


def test_kruskal_requires_two_groups():
    with pytest.raises(ValueError):
        kruskal_wallis({
            "Control": [1, 2, 3],
        })


# --------------------------------------------------
# Spearman correlation
# --------------------------------------------------

def test_spearman_positive_correlation():
    result = spearman_correlation(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
    )

    assert result["correlation"] == pytest.approx(1.0)
    assert result["p_value"] < 0.05


def test_spearman_negative_correlation():
    result = spearman_correlation(
        [1, 2, 3, 4, 5],
        [10, 8, 6, 4, 2],
    )

    assert result["correlation"] == pytest.approx(-1.0)


def test_spearman_unequal_lengths():
    with pytest.raises(ValueError):
        spearman_correlation(
            [1, 2, 3],
            [1, 2],
        )


def test_spearman_constant_variable():
    with pytest.raises(ValueError):
        spearman_correlation(
            [1, 1, 1, 1],
            [2, 3, 4, 5],
        )


def test_spearman_too_few_pairs():
    with pytest.raises(ValueError):
        spearman_correlation(
            [1, 2],
            [3, 4],
        )


def test_spearman_missing_values():
    result = spearman_correlation(
        [1, 2, np.nan, 4, 5],
        [2, 4, 6, 8, 10],
    )

    assert result["sample_size"] == 4


# --------------------------------------------------
# Holm-Bonferroni correction
# --------------------------------------------------

def test_holm_correction_adjusts_p_values():
    result = holm_correction([0.01, 0.04, 0.03])

    assert result["method"] == "Holm-Bonferroni"
    assert result["adjusted_p_values"] == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_holm_correction_preserves_original_order():
    result = holm_correction([0.04, 0.01, 0.03])

    assert result["adjusted_p_values"] == pytest.approx(
        [0.06, 0.03, 0.06]
    )


def test_holm_correction_invalid_p_values():
    with pytest.raises(ValueError):
        holm_correction([0.1, 1.5])


def test_holm_correction_empty_input():
    with pytest.raises(ValueError):
        holm_correction([])


def test_holm_correction_significance():
    result = holm_correction([0.001, 0.02, 0.5])

    assert result["significant"] == [True, False, False]
