from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_consts import DEFAULT_IMPORTANCE_THRESHOLD
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.tracing.profile_stats import ProfileStats

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.jfr_parser import JfrProfile

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

java_infra_patterns = {
    "org.junit.",
    "org.gradle.",
    "org.apache.maven.",
    "org.jacoco.",
    "com.codeflash.",
    "sun.",
    "jdk.",
    "java.lang.reflect.",
    "org.mockito.",
    "org.assertj.",
}


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


def is_java_infrastructure(class_name: str) -> bool:
    return any(class_name.startswith(pattern) for pattern in java_infra_patterns)


class JavaFunctionRanker:
    """Ranks Java functions using JFR profiling data."""

    def __init__(self, jfr_profile: JfrProfile) -> None:
        self._jfr_profile = jfr_profile
        self._ranking = jfr_profile.get_method_ranking()
        self._ranking_by_name: dict[str, dict[str, Any]] = {}
        for entry in self._ranking:
            name = entry["method_name"]
            if name not in self._ranking_by_name:
                self._ranking_by_name[name] = entry

    def get_function_stats_summary(self, function_to_optimize: FunctionToOptimize) -> dict[str, Any] | None:
        for entry in self._ranking:
            if entry["method_name"] == function_to_optimize.function_name:
                return {
                    "filename": "",
                    "function_name": entry["method_name"],
                    "qualified_name": f"{entry['class_name']}.{entry['method_name']}",
                    "class_name": entry["class_name"],
                    "line_number": 0,
                    "call_count": entry["sample_count"],
                    "own_time_ns": self._jfr_profile.get_addressable_time_ns(entry["class_name"], entry["method_name"]),
                    "addressable_time_ns": self._jfr_profile.get_addressable_time_ns(
                        entry["class_name"], entry["method_name"]
                    ),
                }
        return None

    def get_function_addressable_time(self, function_to_optimize: FunctionToOptimize) -> float:
        entry = self._ranking_by_name.get(function_to_optimize.function_name)
        if entry is None:
            return 0.0
        return self._jfr_profile.get_addressable_time_ns(entry["class_name"], entry["method_name"])

    def rank_functions(
        self, functions_to_optimize: list[FunctionToOptimize], min_functions: int = 5
    ) -> list[FunctionToOptimize]:
        if not self._ranking:
            logger.warning("No JFR profiling data available to rank functions.")
            return functions_to_optimize

        total_time = sum(
            self._jfr_profile.get_addressable_time_ns(e["class_name"], e["method_name"])
            for e in self._ranking
            if not is_java_infrastructure(e["class_name"])
        )

        if total_time == 0:
            return functions_to_optimize

        functions_with_time = []
        functions_without_time = []
        for func in functions_to_optimize:
            addr_time = self.get_function_addressable_time(func)
            if addr_time > 0:
                importance = addr_time / total_time
                if importance >= DEFAULT_IMPORTANCE_THRESHOLD:
                    functions_with_time.append(func)
                else:
                    logger.debug(
                        f"Filtering out Java function {func.qualified_name} with importance "
                        f"{importance:.2%} (below threshold {DEFAULT_IMPORTANCE_THRESHOLD:.2%})"
                    )
                    functions_without_time.append(func)
            else:
                functions_without_time.append(func)

        ranked = sorted(functions_with_time, key=self.get_function_addressable_time, reverse=True)

        # Guarantee at least min_functions pass through even when JFR data is sparse.
        # Functions without JFR samples may still benefit from optimization.
        if len(ranked) < min_functions:
            shortfall = min_functions - len(ranked)
            ranked_set = {id(f) for f in ranked}
            for func in functions_without_time[:shortfall]:
                if id(func) not in ranked_set:
                    ranked.append(func)
            if shortfall > 0:
                logger.info(
                    f"JFR data only covered {len(functions_with_time)} functions; "
                    f"added {min(shortfall, len(functions_without_time))} more to meet minimum of {min_functions}"
                )

        return ranked


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
                f"Loaded timing stats for {len(self._function_stats)} functions from trace using ProfileStats "
                f"(filtered {pytest_filtered_count} pytest infrastructure functions)"
            )

        except Exception as e:
            logger.warning(f"Failed to process function stats from trace file {self.trace_file_path}: {e}")
            self._function_stats = {}

    def get_function_stats_summary(self, function_to_optimize: FunctionToOptimize) -> dict | None:
        target_filename = function_to_optimize.file_path.name
        candidates = self._function_stats_by_name.get(function_to_optimize.function_name)
        if not candidates:
            logger.debug(
                f"Could not find stats for function {function_to_optimize.function_name} in file {target_filename}"
            )
            return None

        for key, stats in candidates:
            # The check preserves exact logic: "key.endswith(f"/{target_filename}") or target_filename in key"
            if key.endswith(f"/{target_filename}") or target_filename in key:
                return stats

        logger.debug(
            f"Could not find stats for function {function_to_optimize.function_name} in file {target_filename}"
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
                f"Using file-relative importance for {len(target_files)} file(s): {target_files}. "
                f"Total file time: {total_program_time:,} ns"
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
                            f"Filtering out function {func.qualified_name} with importance "
                            f"{importance:.2%} (below threshold {DEFAULT_IMPORTANCE_THRESHOLD:.2%})"
                        )

            logger.info(
                f"Filtered down to {len(functions_to_rank)} important functions "
                f"from {len(functions_to_optimize)} total functions"
            )

        ranked = sorted(functions_to_rank, key=self.get_function_addressable_time, reverse=True)
        logger.debug(
            f"Function ranking order: {[f'{func.function_name} (addressable_time={self.get_function_addressable_time(func):.2f}ns)' for func in ranked]}"
        )
        return ranked
