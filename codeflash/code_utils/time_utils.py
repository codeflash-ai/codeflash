from __future__ import annotations


def humanize_runtime(time_in_ns: int) -> str:
    # Fast path for small values and avoid calling heavy humanize functions when possible
    if time_in_ns < 1000:
        # < 1 microsecond
        units = "nanosecond" if time_in_ns == 1 else "nanoseconds"
        return f"{time_in_ns} {units}"
    if time_in_ns < 1_000_000:
        time_micro = time_in_ns / 1000
        units = "microsecond" if time_micro == 1 else "microseconds"
        return f"{time_micro:.2f} {units}"
    if time_in_ns < 1_000_000_000:
        time_milli = time_in_ns / 1_000_000
        units = "millisecond" if time_milli == 1 else "milliseconds"
        return f"{time_milli:.2f} {units}"
    if time_in_ns < 60 * 1_000_000_000:
        time_sec = time_in_ns / 1_000_000_000
        units = "second" if time_sec == 1 else "seconds"
        return f"{time_sec:.2f} {units}"
    if time_in_ns < 3600 * 1_000_000_000:
        time_min = time_in_ns / (60 * 1_000_000_000)
        units = "minute" if time_min == 1 else "minutes"
        return f"{time_min:.2f} {units}"
    if time_in_ns < 24 * 3600 * 1_000_000_000:
        time_hr = time_in_ns / (3600 * 1_000_000_000)
        units = "hour" if time_hr == 1 else "hours"
        return f"{time_hr:.2f} {units}"
    time_day = time_in_ns / (24 * 3600 * 1_000_000_000)
    units = "day" if time_day == 1 else "days"
    return f"{time_day:.2f} {units}"


def format_time(nanoseconds: int) -> str:
    """Format nanoseconds into a human-readable string with 3 significant digits when needed."""
    # Define conversion factors and units
    if not isinstance(nanoseconds, int):
        raise TypeError("Input must be an integer.")
    if nanoseconds < 0:
        raise ValueError("Input must be a positive integer.")

    if nanoseconds < 1_000:
        return f"{nanoseconds}ns"
    if nanoseconds < 1_000_000:
        value = nanoseconds / 1_000
        return f"{value:.2f}μs" if value < 10 else (f"{value:.1f}μs" if value < 100 else f"{int(value)}μs")
    if nanoseconds < 1_000_000_000:
        value = nanoseconds / 1_000_000
        return f"{value:.2f}ms" if value < 10 else (f"{value:.1f}ms" if value < 100 else f"{int(value)}ms")
    value = nanoseconds / 1_000_000_000
    return f"{value:.2f}s" if value < 10 else (f"{value:.1f}s" if value < 100 else f"{int(value)}s")


def format_perf(percentage: float) -> str:
    """Format percentage into a human-readable string with 3 significant digits when needed."""
    # Branch order optimized
    abs_perc = abs(percentage)
    if abs_perc >= 100:
        return f"{percentage:.0f}"
    if abs_perc >= 10:
        return f"{percentage:.1f}"
    if abs_perc >= 1:
        return f"{percentage:.2f}"
    return f"{percentage:.3f}"
