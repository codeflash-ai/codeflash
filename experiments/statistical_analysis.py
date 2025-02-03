from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, njit, prange

if TYPE_CHECKING:
    import numpy.typing as npt

TWO_SIGMA = 2


@njit(parallel=True, fastmath=True, cache=True)
def bootstrap_minima(
    series: list[int], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.int64]:
    num_threads = len(rngs)
    series_size = len(series)
    npseries = np.array(series)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    minima = np.empty(bootstrap_size)
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for i in prange(num_threads):
        thread_minima = minima[thread_idx[i] : thread_idx[i + 1]]
        for k in range(num_bootstraps_per_thread[i]):
            thread_minima[k] = min(npseries[rngs[i].integers(0, series_size, size=series_size)])
    return minima


@njit(parallel=True, fastmath=True, cache=True)
def bootstrap_minima_ratios(
    series1: list[int], series2: list[int], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.float64]:
    num_threads = len(rngs)
    series1_size = len(series1)
    series2_size = len(series2)
    npseries1 = np.array(series1)
    npseries2 = np.array(series2)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    minima_ratios = np.empty(bootstrap_size, dtype=np.float64)
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for i in prange(num_threads):
        thread_minima_ratios = minima_ratios[thread_idx[i] : thread_idx[i + 1]]
        for k in range(num_bootstraps_per_thread[i]):
            min2 = min(npseries2[rngs[i].integers(0, series2_size, size=series2_size)])
            if min2 == 0:
                thread_minima_ratios[k] = np.inf
            else:
                thread_minima_ratios[k] = min(npseries1[rngs[i].integers(0, series1_size, size=series1_size)]) / min2
    return minima_ratios


@njit(parallel=True, fastmath=True, cache=True)
def bootstrap_ratios_geomean(
    ratio_series: list[npt.NDArray[np.float64]], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.float64]:
    num_series = len(ratio_series)
    num_threads = len(rngs)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    combined_ratios = np.empty(bootstrap_size, dtype=np.float64)
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for i in prange(num_threads):
        thread_combined_ratios = combined_ratios[thread_idx[i] : thread_idx[i + 1]]
        for k in range(num_bootstraps_per_thread[i]):
            sum_log = 0.0
            for series in ratio_series:
                ratio = series[rngs[i].integers(0, len(series))]
                if ratio <= 0:
                    ratio = 1e-12
                sum_log += np.log(ratio)
            thread_combined_ratios[k] = np.exp(sum_log / num_series)
    return combined_ratios


def bootstrap_noise_floor(series: list[int], bootstrap_size: int) -> np.float64:
    rng = np.random.default_rng()
    return np.std(bootstrap_minima(series, tuple(rng.spawn(get_num_threads())), bootstrap_size))


def combined_series_noise_floor(series1: list[int], series2: list[int], bootstrap_size: int) -> np.float64:
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
    uncertainty = TWO_SIGMA * 100 * combined_noise_floor / min1
    if combined_noise_floor == 0 or min_diff / combined_noise_floor > TWO_SIGMA:
        return percent_diff, uncertainty
    return None


def analyze_series_speedup(
    multi_series1: list[list[int]], multi_series2: list[list[int]], bootstrap_size: int
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64]:
    rng = np.random.default_rng()
    combined_ratios = bootstrap_ratios_geomean(
        [
            bootstrap_minima_ratios(series1, series2, tuple(rng.spawn(get_num_threads())), bootstrap_size)
            for series1, series2 in zip(multi_series1, multi_series2)
        ],
        tuple(rng.spawn(get_num_threads())),
        bootstrap_size,
    )
    lower_bound_95_confidence = np.percentile(combined_ratios, 2.5)
    upper_bound_95_confidence = np.percentile(combined_ratios, 97.5)
    mean = np.mean(combined_ratios)
    probablility_faster = np.mean(combined_ratios > 1.0)
    return lower_bound_95_confidence, upper_bound_95_confidence, mean, probablility_faster
