from __future__ import annotations

import pstats
import sqlite3
from copy import copy
from pathlib import Path
from typing import Any, TextIO

from codeflash.cli_cmds.console import logger


class ProfileStats(pstats.Stats):
    # Attributes set by pstats.Stats.init() — stubs don't expose them
    files: list[str]
    stream: TextIO
    top_level: set[tuple[str, int, str]]
    total_calls: int
    prim_calls: int
    total_tt: float
    max_name_len: int
    fcn_list: list[tuple[str, int, str]] | None
    sort_arg_dict: dict[str, tuple[Any, ...]]
    all_callees: dict[tuple[str, int, str], dict[tuple[str, int, str], tuple[int, int, float, float]]] | None
    stats: dict[tuple[str, int, str], tuple[int, int, int | float, int | float, dict[Any, Any]]]

    def __init__(self, trace_file_path: str, time_unit: str = "ns") -> None:
        assert Path(trace_file_path).is_file(), f"Trace file {trace_file_path} does not exist"
        assert time_unit in {"ns", "us", "ms", "s"}, f"Invalid time unit {time_unit}"
        self.trace_file_path = trace_file_path
        self.time_unit = time_unit
        logger.debug(hasattr(self, "create_stats"))
        super().__init__(copy(self))  # type: ignore[arg-type]  # pstats uses duck-typed create_stats interface

    def create_stats(self) -> None:
        self.con = sqlite3.connect(self.trace_file_path)
        cur = self.con.cursor()
        pdata = cur.execute(
            "SELECT filename, line_number, function, class_name,"
            " call_count_nonrecursive, num_callers, total_time_ns, cumulative_time_ns"
            " FROM pstats"
        ).fetchall()
        self.con.close()
        time_conversion_factor = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[self.time_unit]
        self.stats = {}
        for (
            filename,
            line_number,
            function,
            class_name,
            call_count_nonrecursive,
            num_callers,
            total_time_ns,
            cumulative_time_ns,
        ) in pdata:
            function_name = f"{class_name}.{function}" if class_name else function

            self.stats[(filename, line_number, function_name)] = (
                call_count_nonrecursive,
                num_callers,
                total_time_ns / time_conversion_factor if time_conversion_factor != 1 else total_time_ns,
                cumulative_time_ns / time_conversion_factor if time_conversion_factor != 1 else cumulative_time_ns,
                {},
            )

    def print_stats(self, *amount: str | float) -> ProfileStats:
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = " " * 8
        for func in self.top_level:
            print(indent, func[2], file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print(f"({self.prim_calls:d} primitive calls)", end=" ", file=self.stream)
        time_unit = {"ns": "nanoseconds", "us": "microseconds", "ms": "milliseconds", "s": "seconds"}[self.time_unit]
        print(f"in {self.total_tt:.3f} {time_unit}", file=self.stream)
        print(file=self.stream)
        _width, list_ = self.get_print_list(amount)
        if list_:
            self.print_title()
            for fn in list_:
                self.print_line(fn)
            print(file=self.stream)
            print(file=self.stream)
        return self


def get_trace_total_run_time_ns(trace_file_path: Path) -> int:
    if not trace_file_path.is_file():
        return 0
    con = sqlite3.connect(trace_file_path)
    cur = con.cursor()
    time_data = cur.execute("SELECT time_ns FROM total_time").fetchone()
    con.close()
    time_data = time_data[0] if time_data else 0
    return int(time_data)
