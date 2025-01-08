from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

TWO_SIGMA = 2


def bootstrap_minima(series: list[int], bootstrap_size: int) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng()
    return np.array([np.min(rng.choice(series, len(series), replace=True)) for _ in range(bootstrap_size)])


def bootstrap_noise_floor(series: list[int], bootstrap_size: int) -> np.float64:
    return np.std(bootstrap_minima(series, bootstrap_size))


def combined_series_noise_floor(series1: list[int], series2: list[int], bootstrap_size: int) -> float:
    noise_floor1 = bootstrap_noise_floor(series1, bootstrap_size)
    noise_floor2 = bootstrap_noise_floor(series2, bootstrap_size)
    return math.sqrt(noise_floor1 * noise_floor1 + noise_floor2 * noise_floor2)


def series2_faster_95_confidence(
    series1: list[int], series2: list[int], bootstrap_size: int
) -> tuple[float, float] | None:
    min1 = min(series1)
    min_diff = min1 - min(series2)
    if min_diff <= 0:
        return None
    combined_noise_floor = combined_series_noise_floor(series1, series2, bootstrap_size)
    percent_diff = 100 * min_diff / min1
    uncertainty = TWO_SIGMA * combined_noise_floor / min1
    if combined_noise_floor == 0 or min_diff / combined_noise_floor > TWO_SIGMA:
        return percent_diff, uncertainty
    return None
