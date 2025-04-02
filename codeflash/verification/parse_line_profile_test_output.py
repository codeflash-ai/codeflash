import linecache
import inspect
from rich.console import Console
from rich.table import Table


class LineProfileFormatter:
    def __init__(self, unit: float = 1.0):
        self.unit = unit
        self.console = Console(record=True)

    def format_time(self, time: float) -> str:
        return f"{time * self.unit:5.1f}"

    def format_per_hit(self, time: float, hits: int) -> str:
        return f"{(time * self.unit) / hits:5.1f}" if hits else "0.0"

    def format_percent(self, part: float, total: float) -> str:
        return f"{100 * part / total:5.1f}" if total else ""

    def show_func(
        self,
        filename: str,
        start_lineno: int,
        func_name: str,
        timings: list[tuple[int, int, float]],
    ) -> str:
        total_time = sum(t[2] for t in timings)
        if not total_time:
            return ""

        table = self._create_timing_table(filename, start_lineno, timings, total_time)
        self.console.print(table)
        return f"## Function: {func_name}\n## Total time: {total_time * self.unit:.6g} s\n{self.console.export_text()}\n"

    def _create_timing_table(
        self,
        filename: str,
        start_lineno: int,
        timings: list[tuple[int, int, float]],
        total_time: float,
    ) -> Table:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Hits")
        table.add_column("Time")
        table.add_column("Per Hit")
        table.add_column("% Time")
        table.add_column("Line Contents", no_wrap=True)

        source_lines = self._get_source_lines(filename, start_lineno)
        if isinstance(source_lines, str):
            return Table(title=source_lines)

        for lineno, nhits, time in timings:
            line = source_lines[lineno - start_lineno].rstrip("\n").rstrip("\r")
            table.add_row(
                str(nhits),
                self.format_time(time),
                self.format_per_hit(time, nhits),
                self.format_percent(time, total_time),
                line,
            )
        return table

    def _get_source_lines(self, filename: str, start_lineno: int) -> list[str] | str:
        try:
            return inspect.getblock(
                linecache.getlines(str(filename))[start_lineno - 1 :]
            )
        except Exception:
            return f"File not found: {filename}"


def show_func(
    filename: str,
    start_lineno: int,
    func_name: str,
    timings: list[tuple[int, int, float]],
    unit: float = 1.0,
) -> str:
    formatter = LineProfileFormatter(unit)
    return formatter.show_func(filename, start_lineno, func_name, timings)
