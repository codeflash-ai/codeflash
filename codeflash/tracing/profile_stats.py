import json
import os.path
import pstats
import sqlite3
from copy import copy


class ProfileStats(pstats.Stats):
    def __init__(self, trace_file_path: str, time_unit: str = "ns"):
        assert os.path.isfile(trace_file_path), f"Trace file {trace_file_path} does not exist"
        assert time_unit in ["ns", "us", "ms", "s"], f"Invalid time unit {time_unit}"
        self.trace_file_path = trace_file_path
        self.time_unit = time_unit
        print(hasattr(self, "create_stats"))
        super().__init__(copy(self))

    def create_stats(self):
        self.con = sqlite3.connect(self.trace_file_path)
        cur = self.con.cursor()
        pdata = cur.execute("SELECT * FROM pstats").fetchall()
        self.con.close()
        time_conversion_factor = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[self.time_unit]
        self.stats = {}
        for (
            filename,
            line_number,
            function,
            call_count_nonrecursive,
            num_callers,
            total_time_ns,
            cumulative_time_ns,
            callers,
        ) in pdata:
            loaded_callers = json.loads(callers)
            unmapped_callers = {caller["key"]: caller["value"] for caller in loaded_callers}
            self.stats[(filename, line_number, function)] = (
                call_count_nonrecursive,
                num_callers,
                total_time_ns / time_conversion_factor if time_conversion_factor != 1 else total_time_ns,
                cumulative_time_ns / time_conversion_factor
                if time_conversion_factor != 1
                else cumulative_time_ns,
                unmapped_callers,
            )

    def print_stats(self, *amount):
        # Copied from pstats.Stats.print_stats and modified to print the correct time unit
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = " " * 8
        for func in self.top_level:
            print(indent, func[2], file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end=" ", file=self.stream)
        time_unit = {"ns": "nanoseconds", "us": "microseconds", "ms": "milliseconds", "s": "seconds"}[
            self.time_unit
        ]
        print("in %.3f %s" % (self.total_tt, time_unit), file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self


def get_trace_total_run_time_ns(trace_file_path: str) -> int:
    if not os.path.isfile(trace_file_path):
        return 0
    con = sqlite3.connect(trace_file_path)
    cur = con.cursor()
    time_data = cur.execute("SELECT time_ns FROM total_time").fetchone()
    con.close()
    time_data = time_data[0] if time_data else 0
    return int(time_data)
