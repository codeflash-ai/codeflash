from __future__ import annotations


def humanize_runtime(time_in_ns: int) -> str:
    runtime_human: str = str(time_in_ns)
    units = "nanoseconds"
    if 1 <= time_in_ns < 2:
        units = "nanosecond"

    if time_in_ns / 1000 >= 1:
        time_micro = float(time_in_ns) / 1000

        # Direct unit determination and formatting without external library
        if time_micro < 1000:
            runtime_human = f"{time_micro:.3g}"
            units = "microseconds" if time_micro >= 2 else "microsecond"
        elif time_micro < 1000000:
            time_milli = time_micro / 1000
            runtime_human = f"{time_milli:.3g}"
            units = "milliseconds" if time_milli >= 2 else "millisecond"
        elif time_micro < 60000000:
            time_sec = time_micro / 1000000
            runtime_human = f"{time_sec:.3g}"
            units = "seconds" if time_sec >= 2 else "second"
        elif time_micro < 3600000000:
            time_min = time_micro / 60000000
            runtime_human = f"{time_min:.3g}"
            units = "minutes" if time_min >= 2 else "minute"
        elif time_micro < 86400000000:
            time_hour = time_micro / 3600000000
            runtime_human = f"{time_hour:.3g}"
            units = "hours" if time_hour >= 2 else "hour"
        else:  # days
            time_day = time_micro / 86400000000
            runtime_human = f"{time_day:.3g}"
            units = "days" if time_day >= 2 else "day"

    runtime_human_parts = str(runtime_human).split(".")
    if len(runtime_human_parts[0]) == 1:
        if runtime_human_parts[0] == "1" and len(runtime_human_parts) > 1:
            units = units + "s"
        if len(runtime_human_parts) == 1:
            runtime_human = f"{runtime_human_parts[0]}.00"
        elif len(runtime_human_parts[1]) >= 2:
            runtime_human = f"{runtime_human_parts[0]}.{runtime_human_parts[1][0:2]}"
        else:
            runtime_human = (
                f"{runtime_human_parts[0]}.{runtime_human_parts[1]}{'0' * (2 - len(runtime_human_parts[1]))}"
            )
    elif len(runtime_human_parts[0]) == 2:
        if len(runtime_human_parts) > 1:
            runtime_human = f"{runtime_human_parts[0]}.{runtime_human_parts[1][0]}"
        else:
            runtime_human = f"{runtime_human_parts[0]}.0"
    else:
        runtime_human = runtime_human_parts[0]

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
