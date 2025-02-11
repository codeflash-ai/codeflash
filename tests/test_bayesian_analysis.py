import numpy as np
from codeflash.verification.bayesian_analysis import (
    analyse_function_runtime_data,
    compare_function_runtime_distributions,
)


def test_bayesian_analysis() -> None:
    functions = ["orig", "opt1", "opt2"]  # original + 2 optimization candidates
    inputs = ["inpA", "inpB", "inpC"]  # 3 benchmarks
    n = 5000  # repeated measurements per (function, input)

    # Let's simulate some data in a dictionary:
    # data[(fn, inp)] = array of shape (N,) with raw runtimes
    rng = np.random.default_rng(42)
    data = {}
    for fn in functions:
        for inp in inputs:
            # We'll do random times with some big outliers for demonstration
            base_time = 1.0
            if fn == "orig":
                factor = 1.0
            elif fn == "opt1":
                factor = 0.85  # 15% faster
            else:  # opt2
                factor = 0.95  # 5% faster
            # We'll also vary by input
            if inp == "inpA":
                factor_inp = 1.0
            elif inp == "inpB":
                factor_inp = 1.2
            else:  # inpC
                factor_inp = 0.9

            # final mean time = base_time * factor * factor_inp
            mu = base_time * factor * factor_inp
            # add noise, outliers
            times = mu + rng.normal(0, 0.2 * mu, size=n)
            times = np.clip(times, 0, None)  # no negative times
            # add some random big spikes
            if rng.random() < 0.1:
                times[rng.integers(0, n)] *= 5.0
            data[(fn, inp)] = times

    orig = [list(data[("orig", i)]) for i in inputs]
    opt1 = [list(data[("opt1", i)]) for i in inputs]
    opt2 = [list(data[("opt2", i)]) for i in inputs]

    original_distribution, original_stats = analyse_function_runtime_data(orig, 10000)
    optimized_distribution1, optimized_stats1 = analyse_function_runtime_data(opt1, 10000)
    optimized_distribution2, optimized_stats2 = analyse_function_runtime_data(opt2, 10000)

    speedup_stats1, faster_prob1 = compare_function_runtime_distributions(
        original_distribution, optimized_distribution1
    )
    assert (
        1.162
        < speedup_stats1["credible_interval_lower_bound"]
        < 1.165
        < speedup_stats1["median"]
        < 1.171
        < speedup_stats1["credible_interval_upper_bound"]
        < 1.174
    )
    assert faster_prob1 == 1.0

    speedup_stats2, faster_prob2 = compare_function_runtime_distributions(
        original_distribution, optimized_distribution2
    )
    assert (
        1.046
        < speedup_stats2["credible_interval_lower_bound"]
        < 1.051
        < speedup_stats2["median"]
        < 1.054
        < speedup_stats2["credible_interval_upper_bound"]
        < 1.057
    )
    assert faster_prob1 == 1.0
