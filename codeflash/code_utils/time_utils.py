from __future__ import annotations

import math


def humanize_runtime(time_in_ns: int) -> str:
    # Fast path for sub-microsecond values
    if time_in_ns < 1000:
        units = "nanoseconds" if time_in_ns != 1 else "nanosecond"
        return f"{time_in_ns} {units}"

    time_micro = time_in_ns / 1000  # microseconds
    # Below logic maps direct unit selection and formatting with minimal overhead.

    if time_micro < 1000:
        value = time_micro
        unit_singular = "microsecond"
        unit_plural = "microseconds"
    elif time_micro < 1_000_000:
        value = time_micro / 1000
        unit_singular = "millisecond"
        unit_plural = "milliseconds"
    elif time_micro < 60_000_000:
        value = time_micro / 1_000_000
        unit_singular = "second"
        unit_plural = "seconds"
    elif time_micro < 3_600_000_000:
        value = time_micro / 60_000_000
        unit_singular = "minute"
        unit_plural = "minutes"
    elif time_micro < 86_400_000_000:
        value = time_micro / 3_600_000_000
        unit_singular = "hour"
        unit_plural = "hours"
    else:
        value = time_micro / 86_400_000_000
        unit_singular = "day"
        unit_plural = "days"

    # Smart formatting (similar to former logic)
    if value < 10:
        str_value = f"{value:.2f}"
    elif value < 100:
        str_value = f"{value:.1f}"
    else:
        str_value = f"{int(round(value))}"

    # Use plural unless it's very close to 1
    units = unit_singular if math.isclose(value, 1.0, abs_tol=1e-9) else unit_plural

    return f"{str_value} {units}"


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
