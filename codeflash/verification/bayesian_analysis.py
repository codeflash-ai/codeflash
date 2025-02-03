from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@nb.njit(parallel=True, fastmath=True, cache=True)
def bayesian_bootstrap_runtime_means(
    runtimes: list[int], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.float64]:
    """Bayesian bootstrap for the mean of the runtimes list.

    Returns an array of shape (bootstrap_size,) with draws from the posterior of the mean.
    We draw random weights from Dirichlet(1,1,...,1) using the rngs random generators
    (one per computation thread), and compute the weighted mean.
    """
    num_timings = len(runtimes)
    np_runtimes = np.array(runtimes).astype(np.float64)
    draws = np.empty(bootstrap_size, dtype=np.float64)

    num_threads = len(rngs)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for thread_id in nb.prange(num_threads):
        thread_draws = draws[thread_idx[thread_id] : thread_idx[thread_id + 1]]
        for bootstrap_id in range(num_bootstraps_per_thread[thread_id]):
            # Dirichlet(1,...,1) is the normalized Gamma(1,1) distribution
            weights = rngs[thread_id].gamma(1.0, 1.0, size=num_timings)
            thread_draws[bootstrap_id] = np.dot(np_runtimes, weights / np.sum(weights))
    return draws


def compute_function_runtime_posterior_means(
    function_runtime_data: list[list[int]], bootstrap_size: int
) -> list[npt.NDArray[np.float64]]:
    """For each list of runtimes associated to a function input, do a Bayesian bootstrap to get a posterior of the mean.

    Returns an array of shape (bootstrap_size,) for each function input.
    """
    rng = np.random.default_rng()
    return [
        bayesian_bootstrap_runtime_means(input_runtime_data, tuple(rng.spawn(nb.get_num_threads())), bootstrap_size)
        for input_runtime_data in function_runtime_data
    ]


@nb.njit(parallel=True, fastmath=True, cache=True)
def bootstrap_combined_function_input_runtime_means(
    posterior_means: list[npt.NDArray[np.float64]], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.float64]:
    """Given a function, we have posterior draws for each input, and get an overall expected time across these inputs.

    We make random draws from each input's distribution using the rngs random generators (one per computation thread),
    and compute their arithmetic mean.
    Returns an array of shape (bootstrap_size,).
    """
    num_inputs = len(posterior_means)
    num_input_means = max([len(posterior_mean) for posterior_mean in posterior_means])
    draws = np.empty(bootstrap_size, dtype=np.float64)

    num_threads = len(rngs)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for thread_id in nb.prange(num_threads):
        thread_draws = draws[thread_idx[thread_id] : thread_idx[thread_id + 1]]
        for bootstrap_id in range(num_bootstraps_per_thread[thread_id]):
            thread_draws[bootstrap_id] = (
                sum([input_means[rngs[thread_id].integers(0, num_input_means)] for input_means in posterior_means])
                / num_inputs
            )
    return draws


@nb.njit(parallel=True, fastmath=True, cache=True)
def bootstrap_combined_function_input_runtime_sums(
    posterior_means: list[npt.NDArray[np.float64]], rngs: tuple[np.random.Generator, ...], bootstrap_size: int
) -> npt.NDArray[np.float64]:
    """Given a function, we have posterior draws for each input, and get an overall expected time across these inputs.

    We make random draws from each input's distribution using the rngs random generators (one per computation thread),
    and compute their arithmetic mean.
    Returns an array of shape (bootstrap_size,).
    """
    num_inputs = len(posterior_means)
    num_input_means = max([len(posterior_mean) for posterior_mean in posterior_means])
    draws = np.empty(bootstrap_size, dtype=np.float64)

    num_threads = len(rngs)
    thread_remainder = bootstrap_size % num_threads
    num_bootstraps_per_thread = np.array([bootstrap_size // num_threads] * num_threads) + np.array(
        [1] * thread_remainder + [0] * (num_threads - thread_remainder)
    )
    thread_idx = [0, *list(np.cumsum(num_bootstraps_per_thread))]

    for thread_id in nb.prange(num_threads):
        thread_draws = draws[thread_idx[thread_id] : thread_idx[thread_id + 1]]
        for bootstrap_id in range(num_bootstraps_per_thread[thread_id]):
            thread_draws[bootstrap_id] = sum(
                [input_means[rngs[thread_id].integers(0, num_input_means)] for input_means in posterior_means]
            )
    return draws


def compute_statistics(distribution: npt.NDArray[np.float64], gamma: float = 0.95) -> dict[str, np.float64]:
    lower_p = (1.0 - gamma) / 2 * 100
    return {
        "median": np.median(distribution),
        "credible_interval_lower_bound": np.percentile(distribution, lower_p),
        "credible_interval_upper_bound": np.percentile(distribution, 100 - lower_p),
    }


def analyze_function_runtime_data(
    function_runtime_data: list[list[int]], bootstrap_size: int
) -> tuple[npt.NDArray[np.float64], dict[str, np.float64]]:
    rng = np.random.default_rng()
    function_runtime_distribution = bootstrap_combined_function_input_runtime_means(
        compute_function_runtime_posterior_means(function_runtime_data, bootstrap_size),
        tuple(rng.spawn(nb.get_num_threads())),
        bootstrap_size,
    )
    return function_runtime_distribution, compute_statistics(function_runtime_distribution)


def analyze_function_runtime_sums_data(
    function_runtime_data: list[list[int]], bootstrap_size: int
) -> tuple[npt.NDArray[np.float64], dict[str, np.float64]]:
    rng = np.random.default_rng()
    function_runtime_distribution = bootstrap_combined_function_input_runtime_sums(
        compute_function_runtime_posterior_means(function_runtime_data, bootstrap_size),
        tuple(rng.spawn(nb.get_num_threads())),
        bootstrap_size,
    )
    return function_runtime_distribution, compute_statistics(function_runtime_distribution)


def compare_function_runtime_distributions(
    function1_runtime_distribution: npt.NDArray[np.float64], function2_runtime_distribution: npt.NDArray[np.float64]
) -> dict[str, np.float64]:
    speedup_distribution = function1_runtime_distribution / function2_runtime_distribution
    return compute_statistics(speedup_distribution)
