import humanize
import datetime as dt
import re


def humanize_runtime(time_in_ns):
    runtime_human = str(time_in_ns)
    units = "nanoseconds"
    if 1 <= time_in_ns < 2:
        units = "nanosecond"

    if time_in_ns / 1000 >= 1:
        time_micro = float(time_in_ns) / 1000
        runtime_human = humanize.precisedelta(
            dt.timedelta(microseconds=time_micro),
            minimum_unit="microseconds",
        )

        units = re.split(",|\s", runtime_human)[1]

        if units == "microseconds" or units == "microsecond":
            runtime_human = float("%.3g" % time_micro)
        elif units == "milliseconds" or units == "millisecond":
            runtime_human = float("%.3g" % (time_micro / 1000))
        elif units == "seconds" or units == "second":
            runtime_human = float("%.3g" % (time_micro / (1000**2)))
        elif units == "minutes" or units == "minute":
            runtime_human = float("%.3g" % (time_micro / (60 * 1000**2)))
        else:  # hours
            runtime_human = float("%.3g" % (time_micro / (3600 * 1000**2)))

    runtime_human = str(runtime_human).split(".")
    if len(runtime_human[0]) == 1:
        if len(runtime_human) == 1:
            runtime_human = runtime_human[0] + ".00"
        elif len(runtime_human[1]) >= 2:
            runtime_human = runtime_human[0] + "." + runtime_human[1][0:2]
        else:
            runtime_human = (
                runtime_human[0] + "." + runtime_human[1] + "0" * (2 - len(runtime_human[1]))
            )
    elif len(runtime_human[0]) == 2:
        if len(runtime_human) > 1:
            runtime_human = runtime_human[0] + "." + runtime_human[1][0]
        else:
            runtime_human = runtime_human[0] + ".0"
    else:
        runtime_human = runtime_human[0]

    runtime_human = runtime_human + " " + units

    return runtime_human
