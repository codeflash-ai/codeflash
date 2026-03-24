from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_core.models import FunctionToOptimize
from codeflash_python.benchmarking.profile_stats import ProfileStats
from codeflash_python.code_utils.config_consts import DEFAULT_IMPORTANCE_THRESHOLD

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.models import FunctionToOptimize


logger = logging.getLogger("codeflash_python")

pytest_patterns = {
    "<frozen",  # Frozen modules like runpy
    "<string>",  # Dynamically evaluated code
    "_pytest/",  # Pytest internals
    "pytest",  # Pytest files
    "pluggy/",  # Plugin system
    "_pydev",  # PyDev debugger
    "runpy.py",  # Python module runner
}
pytest_func_patterns = {"pytest_", "_pytest", "runtest"}


def is_pytest_infrastructure(filename: str, function_name: str) -> bool:
    """Check if a function is part of pytest infrastructure that should be excluded from ranking.

    This filters out pytest internal functions, hooks, and test framework code that
    would otherwise dominate the ranking but aren't candidates for optimization.
    """
    # Check filename patterns
    for pattern in pytest_patterns:
        if pattern in filename:
            return True

    return any(pattern in function_name.lower() for pattern in pytest_func_patterns)


class FunctionRanker:
    """Ranks and filters functions based on % of addressable time derived from profiling data.

    The % of addressable time is calculated as:
        addressable_time = own_time + (time_spent_in_callees / call_count)

    This represents the runtime of a function plus the runtime of its immediate dependent functions,
    as a fraction of overall runtime. It prioritizes functions that are computationally heavy themselves
    (high `own_time`) or that make expensive calls to other functions (high average `time_spent_in_callees`).

    Functions are first filtered by an importance threshold based on their `own_time` as a
    fraction of the total runtime. The remaining functions are then ranked by their % of addressable time
    to identify the best candidates for optimization.
    """

    def __init__(self, trace_file_path: Path) -> None:
        self.trace_file_path = trace_file_path
        self._profile_stats = ProfileStats(trace_file_path.as_posix())
        self._function_stats: dict[str, dict] = {}
        self._function_stats_by_name: dict[str, list[tuple[str, dict]]] = {}
        self.load_function_stats()

        # Build index for faster lookups: map function_name to list of (key, stats)
        for key, stats in self._function_stats.items():
            func_name = stats.get("function_name")
            if func_name:
                self._function_stats_by_name.setdefault(func_name, []).append((key, stats))

    def load_function_stats(self) -> None:
        try:
            pytest_filtered_count = 0
            for (filename, line_number, func_name), (
                call_count,
                _num_callers,
                total_time_ns,
                cumulative_time_ns,
                _callers,
            ) in self._profile_stats.stats.items():
                if call_count <= 0:
                    continue

                if is_pytest_infrastructure(filename, func_name):
                    pytest_filtered_count += 1
                    continue

                # Parse function name to handle methods within classes
                class_name, qualified_name, base_function_name = (None, func_name, func_name)
                if "." in func_name and not func_name.startswith("<"):
                    parts = func_name.split(".", 1)
                    if len(parts) == 2:
                        class_name, base_function_name = parts

                # Calculate own time (total time - time spent in subcalls)
                own_time_ns = total_time_ns
                time_in_callees_ns = cumulative_time_ns - total_time_ns

                # Calculate addressable time (own time + avg time in immediate callees)
                addressable_time_ns = own_time_ns + (time_in_callees_ns / call_count)

                function_key = f"{filename}:{qualified_name}"
                self._function_stats[function_key] = {
                    "filename": filename,
                    "function_name": base_function_name,
                    "qualified_name": qualified_name,
                    "class_name": class_name,
                    "line_number": line_number,
                    "call_count": call_count,
                    "own_time_ns": own_time_ns,
                    "cumulative_time_ns": cumulative_time_ns,
                    "time_in_callees_ns": time_in_callees_ns,
                    "addressable_time_ns": addressable_time_ns,
                }

            logger.debug(
                "Loaded timing stats for %s functions from trace using ProfileStats "
                "(filtered %s pytest infrastructure functions)",
                len(self._function_stats),
                pytest_filtered_count,
            )

        except Exception as e:
            logger.warning("Failed to process function stats from trace file %s: %s", self.trace_file_path, e)
            self._function_stats = {}

    def get_function_stats_summary(self, function_to_optimize: FunctionToOptimize) -> dict | None:
        target_filename = function_to_optimize.file_path.name
        candidates = self._function_stats_by_name.get(function_to_optimize.function_name)
        if not candidates:
            logger.debug(
                "Could not find stats for function %s in file %s", function_to_optimize.function_name, target_filename
            )
            return None

        for key, stats in candidates:
            # The check preserves exact logic: "key.endswith(f"/{target_filename}") or target_filename in key"
            if key.endswith(f"/{target_filename}") or target_filename in key:
                return stats

        logger.debug(
            "Could not find stats for function %s in file %s", function_to_optimize.function_name, target_filename
        )
        return None

    def get_function_addressable_time(self, function_to_optimize: FunctionToOptimize) -> float:
        """Get the addressable time in nanoseconds for a function.

        Addressable time = own_time + (time_in_callees / call_count)
        This represents the runtime of the function plus runtime of immediate dependent functions.
        """
        stats = self.get_function_stats_summary(function_to_optimize)
        return stats["addressable_time_ns"] if stats else 0.0

    def rank_functions(self, functions_to_optimize: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
        """Ranks and filters functions based on their % of addressable time and importance.

        Filters out functions whose own_time is less than DEFAULT_IMPORTANCE_THRESHOLD
        of file-relative runtime, then ranks the remaining functions by addressable time.

        Importance is calculated relative to functions in the same file(s) rather than
        total program time. This avoids filtering out functions due to test infrastructure
        overhead.

        The addressable time metric (own_time + avg time in immediate callees) prioritizes
        functions that are computationally heavy themselves or that make expensive calls
        to other functions.

        Args:
            functions_to_optimize: List of functions to rank.

        Returns:
            Important functions sorted in descending order of their addressable time.

        """
        if not self._function_stats:
            logger.warning("No function stats available to rank functions.")
            return []

        # Calculate total time from functions in the same file(s) as functions to optimize
        if functions_to_optimize:
            # Get unique files from functions to optimize
            target_files = {func.file_path.name for func in functions_to_optimize}
            # Calculate total time only from functions in these files
            total_program_time = sum(
                s["own_time_ns"]
                for s in self._function_stats.values()
                if s.get("own_time_ns", 0) > 0
                and any(
                    str(s.get("filename", "")).endswith("/" + target_file) or s.get("filename") == target_file
                    for target_file in target_files
                )
            )
            logger.debug(
                "Using file-relative importance for %s file(s): %s. Total file time: %s ns",
                len(target_files),
                target_files,
                format(total_program_time, ","),
            )
        else:
            total_program_time = sum(
                s["own_time_ns"] for s in self._function_stats.values() if s.get("own_time_ns", 0) > 0
            )

        if total_program_time == 0:
            logger.warning("Total program time is zero, cannot determine function importance.")
            functions_to_rank = functions_to_optimize
        else:
            functions_to_rank = []
            for func in functions_to_optimize:
                func_stats = self.get_function_stats_summary(func)
                if func_stats and func_stats.get("addressable_time_ns", 0) > 0:
                    importance = func_stats["addressable_time_ns"] / total_program_time
                    if importance >= DEFAULT_IMPORTANCE_THRESHOLD:
                        functions_to_rank.append(func)
                    else:
                        logger.debug(
                            "Filtering out function %s with importance %.2f%% (below threshold %.2f%%)",
                            func.qualified_name,
                            importance * 100,
                            DEFAULT_IMPORTANCE_THRESHOLD * 100,
                        )

            logger.info(
                "Filtered down to %s important functions from %s total functions",
                len(functions_to_rank),
                len(functions_to_optimize),
            )

        ranked = sorted(functions_to_rank, key=self.get_function_addressable_time, reverse=True)
        logger.debug(
            "Function ranking order: %s",
            [
                f"{func.function_name} (addressable_time={self.get_function_addressable_time(func):.2f}ns)"
                for func in ranked
            ],
        )
        return ranked
