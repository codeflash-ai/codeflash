import humanize
import datetime as dt
import re


def humanize_runtime(time_in_ns):
    runtime_human = str(time_in_ns) + " nanoseconds"

    if time_in_ns / 1000 >= 1:
        time_micro = float(time_in_ns) / 1000
        runtime_human = humanize.precisedelta(
            dt.timedelta(microseconds=time_micro),
            minimum_unit="microseconds",
        )

        units = re.split(",|\s", runtime_human)[1]

        if units == "microseconds" or units == "microsecond":
            runtime_human = float("%.3g" % time_micro)
            runtime_human = "%g" % runtime_human
        elif units == "milliseconds" or units == "millisecond":
            runtime_human = float("%.3g" % (time_micro / 1000))
            runtime_human = "%g" % runtime_human
        elif units == "seconds" or units == "second":
            runtime_human = float("%.3g" % (time_micro / (1000**2)))
            runtime_human = "%g" % runtime_human
        elif units == "minutes" or units == "minute":
            runtime_human = float("%.3g" % (time_micro / (60 * 1000**2)))
            runtime_human = "%g" % runtime_human
        else:  # hours
            runtime_human = float("%.3g" % (time_micro / (3600 * 1000**2)))
            runtime_human = "%g" % runtime_human

        runtime_human = str(runtime_human) + " " + units

    return runtime_human
