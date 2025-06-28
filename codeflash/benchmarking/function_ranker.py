from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.tracing.profile_stats import ProfileStats

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class FunctionRanker:
    """Ranks functions for optimization based on trace data using ttX scoring.

    ttX = own_time + (time_spent_in_callees x call_count)

    This prioritizes functions that:
    1. Take significant time themselves (own_time)
    2. Are called frequently and have expensive subcalls (time_spent_in_callees x call_count)
    """

    def __init__(self, trace_file_path: Path) -> None:
        self.trace_file_path = trace_file_path
        self._function_stats = None

    def load_function_stats(self) -> dict[str, dict]:
        """Load function timing statistics from trace database using ProfileStats."""
        if self._function_stats is not None:
            return self._function_stats

        self._function_stats = {}

        try:
            profile_stats = ProfileStats(self.trace_file_path.as_posix())

            # Access the stats dictionary directly from ProfileStats
            for (filename, line_number, function_name), (
                call_count,
                _num_callers,
                total_time_ns,
                cumulative_time_ns,
                _callers,
            ) in profile_stats.stats.items():
                if call_count <= 0:
                    continue

                if "." in function_name and not function_name.startswith("<"):
                    parts = function_name.split(".", 1)
                    if len(parts) == 2:
                        class_name, method_name = parts
                        qualified_name = function_name
                        base_function_name = method_name
                    else:
                        class_name = None
                        qualified_name = function_name
                        base_function_name = function_name
                else:
                    class_name = None
                    qualified_name = function_name
                    base_function_name = function_name

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
            logger.warning(f"Failed to load function stats from trace file {self.trace_file_path}: {e}")
            self._function_stats = {}

        return self._function_stats

    def get_function_ttx_score(self, function_to_optimize: FunctionToOptimize) -> float:
        stats = self.load_function_stats()

        possible_keys = [
            f"{function_to_optimize.file_path}:{function_to_optimize.qualified_name}",
            f"{function_to_optimize.file_path}:{function_to_optimize.function_name}",
        ]

        for key in possible_keys:
            if key in stats:
                return stats[key]["ttx_score"]

        return 0.0

    def rank_functions(self, functions_to_optimize: list[FunctionToOptimize]) -> list[FunctionToOptimize]:
        # Calculate ttX scores for all functions
        function_scores = []
        for func in functions_to_optimize:
            ttx_score = self.get_function_ttx_score(func)
            function_scores.append((func, ttx_score))

        # Sort by ttX score descending (highest impact first)
        function_scores.sort(key=lambda x: x[1], reverse=True)

        # logger.info("Function ranking by ttX score:")
        # for i, (func, score) in enumerate(function_scores[:10]):  # Top 10
        #     logger.info(f"  {i + 1}. {func.qualified_name} (ttX: {score:.0f}ns)")

        ranked_functions = [func for func, _ in function_scores]
        logger.info(f"Ranked {len(ranked_functions)} functions by optimization priority")

        return ranked_functions

    def get_function_stats_summary(self, function_to_optimize: FunctionToOptimize) -> dict | None:
        stats = self.load_function_stats()

        possible_keys = [
            f"{function_to_optimize.file_path}:{function_to_optimize.qualified_name}",
            f"{function_to_optimize.file_path}:{function_to_optimize.function_name}",
        ]

        for key in possible_keys:
            if key in stats:
                return stats[key]

        return None
