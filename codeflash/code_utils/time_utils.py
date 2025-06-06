import datetime as dt
import re

import humanize


def humanize_runtime(time_in_ns: int) -> str:
    runtime_human: str = str(time_in_ns)
    units = "nanoseconds"
    if 1 <= time_in_ns < 2:
        units = "nanosecond"

    if time_in_ns / 1000 >= 1:
        time_micro = float(time_in_ns) / 1000
        runtime_human = humanize.precisedelta(dt.timedelta(microseconds=time_micro), minimum_unit="microseconds")

        units = re.split(r",|\s", runtime_human)[1]

        if units in {"microseconds", "microsecond"}:
            runtime_human = f"{time_micro:.3g}"
        elif units in {"milliseconds", "millisecond"}:
            runtime_human = "%.3g" % (time_micro / 1000)
        elif units in {"seconds", "second"}:
            runtime_human = "%.3g" % (time_micro / (1000**2))
        elif units in {"minutes", "minute"}:
            runtime_human = "%.3g" % (time_micro / (60 * 1000**2))
        elif units in {"hour", "hours"}:  # hours
            runtime_human = "%.3g" % (time_micro / (3600 * 1000**2))
        else:  # days
            runtime_human = "%.3g" % (time_micro / (24 * 3600 * 1000**2))
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
    # Inlined significant digit check: >= 3 digits if value >= 100
    if nanoseconds < 1_000:
        return f"{nanoseconds}ns"
    if nanoseconds < 1_000_000:
        microseconds_int = nanoseconds // 1_000
        if microseconds_int >= 100:
            return f"{microseconds_int}μs"
        microseconds = nanoseconds / 1_000
        # Format with precision: 3 significant digits
        if microseconds >= 100:
            return f"{microseconds:.0f}μs"
        if microseconds >= 10:
            return f"{microseconds:.1f}μs"
        return f"{microseconds:.2f}μs"
    if nanoseconds < 1_000_000_000:
        milliseconds_int = nanoseconds // 1_000_000
        if milliseconds_int >= 100:
            return f"{milliseconds_int}ms"
        milliseconds = nanoseconds / 1_000_000
        if milliseconds >= 100:
            return f"{milliseconds:.0f}ms"
        if milliseconds >= 10:
            return f"{milliseconds:.1f}ms"
        return f"{milliseconds:.2f}ms"
    seconds_int = nanoseconds // 1_000_000_000
    if seconds_int >= 100:
        return f"{seconds_int}s"
    seconds = nanoseconds / 1_000_000_000
    if seconds >= 100:
        return f"{seconds:.0f}s"
    if seconds >= 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.2f}s"
