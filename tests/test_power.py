import pytest

from biodatalab.power import calculate_power, estimate_sample_size


def test_calculate_power_basic():
    power = calculate_power(effect_size=0.5, sample_size=30)
    assert 0 < power < 1


def test_power_increases_with_sample_size():
    small = calculate_power(effect_size=0.5, sample_size=20)
    large = calculate_power(effect_size=0.5, sample_size=100)

    assert large > small


def test_power_increases_with_effect_size():
    small = calculate_power(effect_size=0.2, sample_size=50)
    large = calculate_power(effect_size=0.8, sample_size=50)

    assert large > small


def test_power_greater_alternative():
    power = calculate_power(
        effect_size=0.5,
        sample_size=30,
        alternative="greater",
    )

    assert 0 < power < 1


def test_power_less_alternative():
    power = calculate_power(
        effect_size=-0.5,
        sample_size=30,
        alternative="less",
    )

    assert 0 < power < 1


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        calculate_power(effect_size=0.5, sample_size=1)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        calculate_power(effect_size=0.5, sample_size=30, alpha=1)


def test_invalid_alternative():
    with pytest.raises(ValueError):
        calculate_power(
            effect_size=0.5,
            sample_size=30,
            alternative="invalid",
        )


def test_estimate_sample_size():
    n = estimate_sample_size(
        effect_size=0.5,
        target_power=0.80,
    )

    assert n >= 2

    power = calculate_power(
        effect_size=0.5,
        sample_size=n,
    )

    assert power >= 0.80


def test_estimated_sample_size_is_minimal():
    n = estimate_sample_size(
        effect_size=0.5,
        target_power=0.80,
    )

    if n > 2:
        previous_power = calculate_power(
            effect_size=0.5,
            sample_size=n - 1,
        )

        assert previous_power < 0.80


def test_zero_effect_size_sample_size():
    with pytest.raises(ValueError):
        estimate_sample_size(effect_size=0)


def test_invalid_target_power():
    with pytest.raises(ValueError):
        estimate_sample_size(
            effect_size=0.5,
            target_power=1,
        )


def test_max_sample_size():
    with pytest.raises(ValueError):
        estimate_sample_size(
            effect_size=0.01,
            target_power=0.99,
            max_sample_size=10,
        )


def test_power_reproducibility():
    assert calculate_power(
        effect_size=0.5,
        sample_size=30,
    ) == calculate_power(
        effect_size=0.5,
        sample_size=30,
    )
