"""Adapted from line_profiler (https://github.com/pyutils/line_profiler) written by Enthought, Inc. (BSD License)."""

from __future__ import annotations

import inspect
import json
import linecache
import os
from typing import TYPE_CHECKING, Optional

import dill as pickle

from codeflash.code_utils.tabulate import tabulate
from codeflash.languages import is_python

if TYPE_CHECKING:
    from pathlib import Path


def show_func(
    filename: str, start_lineno: int, func_name: str, timings: list[tuple[int, int, float]], unit: float
) -> str:
    total_hits = sum(t[1] for t in timings)
    total_time = sum(t[2] for t in timings)
    out_table = ""
    table_rows = []
    if total_hits == 0:
        return ""
    scalar = 1
    sublines = []
    if os.path.exists(filename):  # noqa: PTH110
        out_table += f"## Function: {func_name}\n"
        # Clear the cache to ensure that we get up-to-date results.
        linecache.clearcache()
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno - 1 :])
    out_table += "## Total time: %g s\n" % (total_time * unit)
    # Define minimum column sizes so text fits and usually looks consistent
    default_column_sizes = {"hits": 9, "time": 12, "perhit": 8, "percent": 8}
    display = {}
    # Loop over each line to determine better column formatting.
    # Fallback to scientific notation if columns are larger than a threshold.
    for lineno, nhits, time in timings:
        percent = "" if total_time == 0 else "%5.1f" % (100 * time / total_time)

        time_disp = "%5.1f" % (time * scalar)
        if len(time_disp) > default_column_sizes["time"]:
            time_disp = "%5.1g" % (time * scalar)
        perhit_disp = "%5.1f" % (float(time) * scalar / nhits)
        if len(perhit_disp) > default_column_sizes["perhit"]:
            perhit_disp = "%5.1g" % (float(time) * scalar / nhits)
        nhits_disp = "%d" % nhits  # noqa: UP031
        if len(nhits_disp) > default_column_sizes["hits"]:
            nhits_disp = f"{nhits:g}"
        display[lineno] = (nhits_disp, time_disp, perhit_disp, percent)
    linenos = range(start_lineno, start_lineno + len(sublines))
    empty = ("", "", "", "")
    table_cols = ("Hits", "Time", "Per Hit", "% Time", "Line Contents")
    for lineno, line in zip(linenos, sublines):
        nhits, time, per_hit, percent = display.get(lineno, empty)
        line_ = line.rstrip("\n").rstrip("\r")
        if "def" in line_ or nhits != "":
            table_rows.append((nhits, time, per_hit, percent, line_))
    out_table += tabulate(
        headers=table_cols, tabular_data=table_rows, tablefmt="pipe", colglobalalign=None, preserve_whitespace=True
    )
    out_table += "\n"
    return out_table


def show_text(stats: dict) -> str:
    """Show text for the given timings."""
    out_table = ""
    out_table += "# Timer unit: {:g} s\n".format(stats["unit"])
    stats_order = sorted(stats["timings"].items())
    # Show detailed per-line information for each function.
    for (fn, lineno, name), _timings in stats_order:
        table_md = show_func(fn, lineno, name, stats["timings"][fn, lineno, name], stats["unit"])
        out_table += table_md
    return out_table


def show_text_non_python(
    stats: dict, line_contents: dict[tuple[str, int], str]
) -> str:
    """Show text for non-Python timings using profiler-provided line contents."""
    out_table = ""
    out_table += "# Timer unit: {:g} s\n".format(stats["unit"])
    stats_order = sorted(stats["timings"].items())
    for (fn, _lineno, name), timings in stats_order:
        total_hits = sum(t[1] for t in timings)
        total_time = sum(t[2] for t in timings)
        if total_hits == 0:
            continue

        out_table += f"## Function: {name}\n"
        out_table += "## Total time: %g s\n" % (total_time * stats["unit"])

        default_column_sizes = {"hits": 9, "time": 12, "perhit": 8, "percent": 8}
        table_rows = []
        for lineno, nhits, time in timings:
            percent = "" if total_time == 0 else "%5.1f" % (100 * time / total_time)
            time_disp = "%5.1f" % time
            if len(time_disp) > default_column_sizes["time"]:
                time_disp = "%5.1g" % time
            perhit = (float(time) / nhits) if nhits > 0 else 0.0
            perhit_disp = "%5.1f" % perhit
            if len(perhit_disp) > default_column_sizes["perhit"]:
                perhit_disp = "%5.1g" % perhit
            nhits_disp = "%d" % nhits  # noqa: UP031
            if len(nhits_disp) > default_column_sizes["hits"]:
                nhits_disp = f"{nhits:g}"

            table_rows.append((nhits_disp, time_disp, perhit_disp, percent, line_contents.get((fn, lineno), "")))

        table_cols = ("Hits", "Time", "Per Hit", "% Time", "Line Contents")
        out_table += tabulate(
            headers=table_cols,
            tabular_data=table_rows,
            tablefmt="pipe",
            colglobalalign=None,
            preserve_whitespace=True,
        )
        out_table += "\n"
    return out_table


def parse_line_profile_results(line_profiler_output_file: Optional[Path]) -> dict:
    if is_python():
        line_profiler_output_file = line_profiler_output_file.with_suffix(".lprof")
        stats_dict = {}
        if not line_profiler_output_file.exists():
            return {"timings": {}, "unit": 0, "str_out": ""}, None
        with line_profiler_output_file.open("rb") as f:
            stats = pickle.load(f)
            stats_dict["timings"] = stats.timings
            stats_dict["unit"] = stats.unit
            str_out = show_text(stats_dict)
            stats_dict["str_out"] = str_out
        return stats_dict, None

    stats_dict = {}
    if line_profiler_output_file is None or not line_profiler_output_file.exists():
        return {"timings": {}, "unit": 0, "str_out": ""}, None

    with line_profiler_output_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert Java/JS JSON output into Python line_profiler-compatible shape.
    # timings: {(filename, start_lineno, func_name): [(lineno, hits, time_raw), ...]}
    grouped_timings: dict[tuple[str, int, str], list[tuple[int, int, int]]] = {}
    lines_by_file: dict[str, list[tuple[int, int, int]]] = {}
    line_contents: dict[tuple[str, int], str] = {}
    for key, stats in raw_data.items():
        file_path = stats.get("file")
        line_num = stats.get("line")
        if file_path is None or line_num is None:
            file_path, line_str = key.rsplit(":", 1)
            line_num = int(line_str)
        line_num = int(line_num)

        lines_by_file.setdefault(file_path, []).append(
            (line_num, int(stats.get("hits", 0)), int(stats.get("time", 0)))
        )
        line_contents[(file_path, line_num)] = stats.get("content", "")

    for file_path, line_stats in lines_by_file.items():
        sorted_line_stats = sorted(line_stats, key=lambda t: t[0])
        if not sorted_line_stats:
            continue
        start_lineno = sorted_line_stats[0][0]
        grouped_timings[(file_path, start_lineno, os.path.basename(file_path))] = sorted_line_stats

    stats_dict["timings"] = grouped_timings
    stats_dict["unit"] = 1e-9
    stats_dict["str_out"] = show_text_non_python(stats_dict, line_contents)
    return stats_dict, None
