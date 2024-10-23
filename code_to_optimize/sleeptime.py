import time


def accurate_sleepfunc(t) -> float:
    """T is in seconds"""
    start_time = time.perf_counter_ns()
    while True:
        if (time.perf_counter_ns() - start_time) / 10e9 >= t:
            break
    return t
