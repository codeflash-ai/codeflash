from __future__ import annotations

import datetime as dt
import re

import humanize


def humanize_runtime(time_in_ns: int) -> str:
    runtime_human: str
    units = "nanoseconds"
    if 1 <= time_in_ns < 2:
        units = "nanosecond"

    if time_in_ns >= 1_000:
        # Direct unit determination and formatting without external library
        if time_in_ns < 1_000_000:
            time_val = float(time_in_ns) / 1_000.0
            runtime_human = f"{time_val:.3g}"
            units = "microseconds" if time_val >= 2 else "microsecond"
        elif time_in_ns < 1_000_000_000:
            time_val = float(time_in_ns) / 1_000_000.0
            runtime_human = f"{time_val:.3g}"
            units = "milliseconds" if time_val >= 2 else "millisecond"
        elif time_in_ns < 60_000_000_000:
            time_val = float(time_in_ns) / 1_000_000_000.0
            runtime_human = f"{time_val:.3g}"
            units = "seconds" if time_val >= 2 else "second"
        elif time_in_ns < 3_600_000_000_000:
            time_val = float(time_in_ns) / 60_000_000_000.0
            runtime_human = f"{time_val:.3g}"
            units = "minutes" if time_val >= 2 else "minute"
        elif time_in_ns < 86_400_000_000_000:
            time_val = float(time_in_ns) / 3_600_000_000_000.0
            runtime_human = f"{time_val:.3g}"
            units = "hours" if time_val >= 2 else "hour"
        else:  # days
            time_val = float(time_in_ns) / 86_400_000_000_000.0
            runtime_human = f"{time_val:.3g}"
            units = "days" if time_val >= 2 else "day"
    else:
        runtime_human = str(time_in_ns)

    # Use partition instead of split to avoid list allocation
    head, sep, tail = runtime_human.partition(".")

    # Reproduce original formatting rules exactly
    if len(head) == 1:
        if head == "1" and sep:
            # original code appended 's' when integer part == "1" and there was a fractional part
            units = units + "s"
        if not sep:  # no fractional part
            runtime_human = f"{head}.00"
        elif len(tail) >= 2:
            runtime_human = f"{head}.{tail[0:2]}"
        else:
            runtime_human = f"{head}.{tail}{'0' * (2 - len(tail))}"
    elif len(head) == 2:
        if sep and tail:
            runtime_human = f"{head}.{tail[0]}"
        else:
            runtime_human = f"{head}.0"
    else:
        runtime_human = head

    return f"{runtime_human} {units}"


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
