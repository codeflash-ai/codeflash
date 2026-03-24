"""Optimizer class for codeflash_python optimization pipeline."""

from __future__ import annotations

import copy
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash_python.code_utils.git_utils import git_root_dir

logger = logging.getLogger("codeflash_python")

if TYPE_CHECKING:
    import ast
    from argparse import Namespace

    from codeflash_core.models import FunctionToOptimize
    from codeflash_python.context.types import DependencyResolver
    from codeflash_python.function_optimizer import FunctionOptimizer
    from codeflash_python.models.models import BenchmarkKey, FunctionCalledInTest

try:
    from codeflash_core.config import TestConfig
except ImportError:
    # Stub if not available
    class TestConfig:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


class Optimizer:
    """Main optimizer class for coordinating the optimization pipeline."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            tests_project_rootdir=args.test_project_root,
            project_root=args.project_root,
            test_command=args.pytest_cmd if hasattr(args, "pytest_cmd") and args.pytest_cmd else "pytest",
            benchmark_tests_root=args.benchmarks_root if hasattr(args, "benchmarks_root") else None,
        )

        from codeflash_python.api.aiservice import AiServiceClient

        self.aiservice_client = AiServiceClient()
        self.replay_tests_dir = None
        self.original_args_and_test_cfg: tuple[Namespace, TestConfig] | None = None
        self.cached_callee_counts: dict[tuple[Path, str], int] = {}

    def create_function_optimizer(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_optimize_ast: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        function_to_tests: dict[str, set[FunctionCalledInTest]] | None = None,
        function_to_optimize_source_code: str | None = "",
        function_benchmark_timings: dict[str, dict[BenchmarkKey, float]] | None = None,
        total_benchmark_timings: dict[BenchmarkKey, float] | None = None,
        call_graph: DependencyResolver | None = None,
        effort_override: str | None = None,
    ) -> FunctionOptimizer | None:
        from codeflash_python.models.function_types import qualified_name_with_modules_from_root

        qualified_name_w_module = qualified_name_with_modules_from_root(function_to_optimize, self.args.project_root)

        function_specific_timings = None
        if (
            hasattr(self.args, "benchmark")
            and self.args.benchmark
            and function_benchmark_timings
            and qualified_name_w_module in function_benchmark_timings
            and total_benchmark_timings
        ):
            function_specific_timings = function_benchmark_timings[qualified_name_w_module]

        from codeflash_python.function_optimizer import FunctionOptimizer

        # Convert float values to int for benchmark timings
        function_specific_timings_int: dict[BenchmarkKey, int] | None = None
        if function_specific_timings:
            function_specific_timings_int = {k: int(v) for k, v in function_specific_timings.items()}

        total_benchmark_timings_int: dict[BenchmarkKey, int] | None = None
        if total_benchmark_timings and function_specific_timings:
            total_benchmark_timings_int = {k: int(v) for k, v in total_benchmark_timings.items()}

        function_optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=self.test_cfg,  # type: ignore[arg-type]
            function_to_optimize_source_code=function_to_optimize_source_code or "",
            function_to_tests=function_to_tests,
            function_to_optimize_ast=function_to_optimize_ast,
            aiservice_client=self.aiservice_client,
            args=self.args,
            function_benchmark_timings=function_specific_timings_int,
            total_benchmark_timings=total_benchmark_timings_int,
            replay_tests_dir=self.replay_tests_dir,
            call_graph=call_graph,
            effort_override=effort_override,
        )
        if function_optimizer.function_to_optimize_ast is None and function_optimizer.requires_function_ast():
            logger.info(
                "Function %s not found in %s.\nSkipping optimization.",
                function_to_optimize.qualified_name,
                function_to_optimize.file_path,
            )
            return None
        return function_optimizer

    def rank_all_functions_globally(
        self,
        file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]],
        trace_file_path: Path | None,
        call_graph: DependencyResolver | None = None,
        test_count_cache: dict[tuple[Path, str], int] | None = None,
    ) -> list[tuple[Path, FunctionToOptimize]]:
        all_functions: list[tuple[Path, FunctionToOptimize]] = []
        for file_path, functions in file_to_funcs_to_optimize.items():
            all_functions.extend((file_path, func) for func in functions)

        if not trace_file_path or not trace_file_path.exists():
            if call_graph is not None:
                return self.rank_by_dependency_count(all_functions, call_graph, test_count_cache=test_count_cache)
            logger.debug("No trace file available, using original function order")
            return all_functions

        try:
            from codeflash_python.benchmarking.function_ranker import FunctionRanker

            logger.info("loading|Ranking functions globally by performance impact...")
            ranker = FunctionRanker(trace_file_path)
            functions_only = [func for _, func in all_functions]
            ranked_functions = ranker.rank_functions(functions_only)

            func_to_file_map = {}
            for file_path, func in all_functions:
                key: tuple[Path, str, int | None] = (func.file_path, func.qualified_name, func.starting_line)
                func_to_file_map[key] = file_path
            ranked_with_metadata: list[tuple[Path, FunctionToOptimize, float, int]] = []
            for rank_index, func in enumerate(ranked_functions):
                key = (func.file_path, func.qualified_name, func.starting_line)
                file_path = func_to_file_map.get(key)
                if file_path:
                    ranked_with_metadata.append(
                        (file_path, func, ranker.get_function_addressable_time(func), rank_index)
                    )

            if test_count_cache:
                ranked_with_metadata.sort(
                    key=lambda item: (-item[2], -test_count_cache.get((item[0], item[1].qualified_name), 0), item[3])
                )

            globally_ranked = [
                (file_path, func) for file_path, func, _addressable_time, _rank_index in ranked_with_metadata
            ]

            logger.info(
                "Globally ranked %s functions by addressable time (filtered %s low-importance functions)",
                len(ranked_functions),
                len(functions_only) - len(ranked_functions),
            )

        except Exception as e:
            logger.warning("Could not perform global ranking: %s", e)
            logger.debug("Falling back to original function order")
            return all_functions
        else:
            return globally_ranked

    def rank_by_dependency_count(
        self,
        all_functions: list[tuple[Path, FunctionToOptimize]],
        call_graph: DependencyResolver,
        test_count_cache: dict[tuple[Path, str], int] | None = None,
    ) -> list[tuple[Path, FunctionToOptimize]]:
        file_to_qns: dict[Path, set[str]] = defaultdict(set)
        for file_path, func in all_functions:
            file_to_qns[file_path].add(func.qualified_name)
        callee_counts = call_graph.count_callees_per_function(dict(file_to_qns))
        self.cached_callee_counts = callee_counts

        if test_count_cache:
            ranked = sorted(
                enumerate(all_functions),
                key=lambda x: (
                    -callee_counts.get((x[1][0], x[1][1].qualified_name), 0),
                    -test_count_cache.get((x[1][0], x[1][1].qualified_name), 0),
                    x[0],
                ),
            )
        else:
            ranked = sorted(
                enumerate(all_functions), key=lambda x: (-callee_counts.get((x[1][0], x[1][1].qualified_name), 0), x[0])
            )
        logger.debug("Ranked %s functions by dependency count (most complex first)", len(ranked))
        return [item for _, item in ranked]

    @staticmethod
    def find_leftover_instrumented_test_files(test_root: Path) -> list[Path]:
        """Search for all paths within the test_root that match instrumented test file patterns.

        Patterns:
        - 'test.*__perf_test_{0,1}.py'
        - 'test_.*__unit_test_{0,1}.py'
        - 'test_.*__perfinstrumented.py'
        - 'test_.*__perfonlyinstrumented.py'

        Returns:
            A list of matching file paths.

        """
        pattern = re.compile(
            r"(?:"
            r"test.*__perf_test_\d?\.py|test_.*__unit_test_\d?\.py|"
            r"test_.*__perfinstrumented\.py|test_.*__perfonlyinstrumented\.py"
            r")$"
        )

        return [
            file_path for file_path in test_root.rglob("*") if file_path.is_file() and pattern.match(file_path.name)
        ]

    def mirror_paths_for_worktree_mode(self, worktree_dir: Path) -> None:
        """Mirror file paths from original git root to worktree directory.

        This updates all paths in args and test_cfg to point to their
        corresponding locations in the worktree.

        Args:
            worktree_dir: The worktree directory to mirror paths to.

        """
        original_args = copy.deepcopy(self.args)
        original_test_cfg = copy.deepcopy(self.test_cfg)
        self.original_args_and_test_cfg = (original_args, original_test_cfg)

        original_git_root = git_root_dir()

        # mirror project_root
        self.args.project_root = mirror_path(self.args.project_root, original_git_root, worktree_dir)
        self.test_cfg.project_root = mirror_path(self.test_cfg.project_root, original_git_root, worktree_dir)  # type: ignore[assignment]

        # mirror module_root
        self.args.module_root = mirror_path(self.args.module_root, original_git_root, worktree_dir)

        # mirror target file
        if hasattr(self.args, "file") and self.args.file:
            self.args.file = mirror_path(self.args.file, original_git_root, worktree_dir)

        if hasattr(self.args, "all") and self.args.all:
            # the args.all path is the same as module_root.
            self.args.all = mirror_path(self.args.all, original_git_root, worktree_dir)

        # mirror tests root
        self.args.tests_root = mirror_path(self.args.tests_root, original_git_root, worktree_dir)
        self.test_cfg.tests_root = mirror_path(self.test_cfg.tests_root, original_git_root, worktree_dir)  # type: ignore[assignment]

        # mirror tests project root
        self.args.test_project_root = mirror_path(self.args.test_project_root, original_git_root, worktree_dir)
        self.test_cfg.tests_project_rootdir = mirror_path(  # type: ignore[assignment,unresolved-attribute]
            self.test_cfg.tests_project_rootdir,
            original_git_root,
            worktree_dir,  # type: ignore[unresolved-attribute]
        )

        # mirror benchmarks root paths
        if hasattr(self.args, "benchmarks_root") and self.args.benchmarks_root:
            self.args.benchmarks_root = mirror_path(self.args.benchmarks_root, original_git_root, worktree_dir)
        if hasattr(self.test_cfg, "benchmark_tests_root") and self.test_cfg.benchmark_tests_root:
            self.test_cfg.benchmark_tests_root = mirror_path(  # type: ignore[assignment]
                self.test_cfg.benchmark_tests_root,
                original_git_root,
                worktree_dir,  # type: ignore[arg-type]
            )


def mirror_path(path: Path | str, src_root: Path, dest_root: Path) -> Path:
    """Mirror a path from src_root to dest_root, preserving relative structure."""
    relative_path = Path(path).resolve().relative_to(src_root.resolve())
    return Path(dest_root / relative_path)
