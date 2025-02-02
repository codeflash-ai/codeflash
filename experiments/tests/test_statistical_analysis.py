import numpy as np
from codeflash.verification.statistical_analysis import series2_faster_95_confidence


def create_timing_series(size: int, mean: int, std_dev: int) -> list[int]:
    mu = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))
    rng = np.random.default_rng()
    return np.round(rng.lognormal(mu, sigma, size)).astype(int).tolist()


def test_compare_timing_series() -> None:
    original_timing_series = create_timing_series(10000, 2000, 60)
    optimized_timing_series = create_timing_series(10000, 1800, 48)
    result = series2_faster_95_confidence(original_timing_series, optimized_timing_series, 10000)
    assert result is not None
    assert 4 < result[0] < 16
    assert result[1] < 4
