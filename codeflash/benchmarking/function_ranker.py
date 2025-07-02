from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.config_consts import DEFAULT_IMPORTANCE_THRESHOLD
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.tracing.profile_stats import ProfileStats

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class FunctionRanker:
    """Ranks and filters functions for optimization based on profiling trace data using the ttX scoring method.

    The FunctionRanker analyzes function-level timing statistics from a trace file and assigns a ttX score to each function:

        ttX = own_time + (time_spent_in_callees x call_count)

    This scoring prioritizes functions that:
      1. Consume significant time themselves (own_time)
      2. Are called frequently and have expensive subcalls (time_spent_in_callees x call_count)

    first, filters out functions whose own_time is less than a specified percentage (importance_threshold = minimum fraction of total runtime a function must account for to be considered important) of the total runtime, considering them unimportant for optimization.

    The remaining functions are then ranked in descending order by their ttX score, prioritizing those most likely to yield performance improvements if optimized.
    """

    def __init__(self, trace_file_path: Path) -> None:
        self.trace_file_path = trace_file_path
        self._profile_stats = ProfileStats(trace_file_path.as_posix())
        self._function_stats: dict[str, dict] = {}
        self.load_function_stats()

    def load_function_stats(self) -> None:
        try:
            for (filename, line_number, func_name), (
                call_count,
                _num_callers,
                total_time_ns,
                cumulative_time_ns,
                _callers,
            ) in self._profile_stats.stats.items():
                if call_count <= 0:
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

                # Calculate ttX score
                ttx_score = own_time_ns + (time_in_callees_ns * call_count)

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
                    "ttx_score": ttx_score,
                }

            logger.debug(f"Loaded timing stats for {len(self._function_stats)} functions from trace using ProfileStats")

        except Exception as e:
            logger.warning(f"Failed to process function stats from trace file {self.trace_file_path}: {e}")
            self._function_stats = {}

    def _get_function_stats(self, function_to_optimize: FunctionToOptimize) -> dict | None:
        # First try qualified_name, then function_name, avoid allocating a list
        key1 = f"{function_to_optimize.file_path}:{function_to_optimize.qualified_name}"
        stats = self._function_stats.get(key1)
        if stats is not None:
            return stats
        key2 = f"{function_to_optimize.file_path}:{function_to_optimize.function_name}"
        return self._function_stats.get(key2, None)

    def get_function_ttx_score(self, function_to_optimize: FunctionToOptimize) -> float:
        stats = self._get_function_stats(function_to_optimize)
        return stats["ttx_score"] if stats else 0.0

    def rank_functions(self, functions_to_optimize: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
        return sorted(functions_to_optimize, key=self.get_function_ttx_score, reverse=True)

    def get_function_stats_summary(self, function_to_optimize: FunctionToOptimize) -> dict | None:
        return self._get_function_stats(function_to_optimize)

    def rerank_and_filter_functions(self, functions_to_optimize: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
        """Reranks and filters functions based on their impact on total runtime.

        This method first calculates the total runtime of all profiled functions.
        It then filters out functions whose own_time is less than a specified
        percentage of the total runtime (importance_threshold).

        The remaining 'important' functions are then ranked by their ttX score.
        """
        stats_map = self._function_stats
        if not stats_map:
            return []

        total_program_time = sum(s["own_time_ns"] for s in stats_map.values() if s.get("own_time_ns", 0) > 0)

        if total_program_time == 0:
            logger.warning("Total program time is zero, cannot determine function importance.")
            return self.rank_functions(functions_to_optimize)

        important_functions = []
        for func in functions_to_optimize:
            func_stats = self._get_function_stats(func)
            if func_stats and func_stats.get("own_time_ns", 0) > 0:
                importance = func_stats["own_time_ns"] / total_program_time
                if importance >= DEFAULT_IMPORTANCE_THRESHOLD:
                    important_functions.append(func)
                else:
                    logger.debug(
                        f"Filtering out function {func.qualified_name} with importance "
                        f"{importance:.2%} (below threshold {DEFAULT_IMPORTANCE_THRESHOLD:.2%})"
                    )

        logger.info(
            f"Filtered down to {len(important_functions)} important functions from {len(functions_to_optimize)} total functions"
        )
        console.rule()

        return self.rank_functions(important_functions)
