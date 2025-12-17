from __future__ import annotations

import ast
import copy
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.api.cfapi import send_completion_email
from codeflash.cli_cmds.console import console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import cleanup_paths, get_run_tmp_file
from codeflash.code_utils.env_utils import get_pr_number, is_pr_draft
from codeflash.code_utils.git_utils import check_running_in_git_repo, git_root_dir
from codeflash.code_utils.git_worktree_utils import (
    create_detached_worktree,
    create_diff_patch_from_worktree,
    create_worktree_snapshot_commit,
    remove_worktree,
)
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.either import is_successful
from codeflash.models.models import ValidCode
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.benchmarking.function_ranker import FunctionRanker
    from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import BenchmarkKey, FunctionCalledInTest
    from codeflash.optimization.function_optimizer import FunctionOptimizer


class Optimizer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            tests_project_rootdir=args.test_project_root,
            project_root_path=args.project_root,
            pytest_cmd=args.pytest_cmd if hasattr(args, "pytest_cmd") and args.pytest_cmd else "pytest",
            benchmark_tests_root=args.benchmarks_root if "benchmark" in args and "benchmarks_root" in args else None,
        )

        self.aiservice_client = AiServiceClient()
        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None
        self.replay_tests_dir = None
        self.trace_file: Path | None = None
        self.functions_checkpoint: CodeflashRunCheckpoint | None = None
        self.current_function_being_optimized: FunctionToOptimize | None = None  # current only for the LSP
        self.current_function_optimizer: FunctionOptimizer | None = None
        self.current_worktree: Path | None = None
        self.original_args_and_test_cfg: tuple[Namespace, TestConfig] | None = None
        self.patch_files: list[Path] = []

    def run_benchmarks(
        self, file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]], num_optimizable_functions: int
    ) -> tuple[dict[str, dict[BenchmarkKey, float]], dict[BenchmarkKey, float]]:
        """Run benchmarks for the functions to optimize and collect timing information."""
        function_benchmark_timings: dict[str, dict[BenchmarkKey, float]] = {}
        total_benchmark_timings: dict[BenchmarkKey, float] = {}

        if not (hasattr(self.args, "benchmark") and self.args.benchmark and num_optimizable_functions > 0):
            return function_benchmark_timings, total_benchmark_timings

        from codeflash.benchmarking.instrument_codeflash_trace import instrument_codeflash_trace_decorator
        from codeflash.benchmarking.plugin.plugin import CodeFlashBenchmarkPlugin
        from codeflash.benchmarking.replay_test import generate_replay_test
        from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
        from codeflash.benchmarking.utils import print_benchmark_table, validate_and_format_benchmark_table

        console.rule()
        with progress_bar(
            f"Running benchmarks in {self.args.benchmarks_root}", transient=True, revert_to_print=bool(get_pr_number())
        ):
            # Insert decorator
            file_path_to_source_code = defaultdict(str)
            for file in file_to_funcs_to_optimize:
                with file.open("r", encoding="utf8") as f:
                    file_path_to_source_code[file] = f.read()
            try:
                instrument_codeflash_trace_decorator(file_to_funcs_to_optimize)
                self.trace_file = Path(self.args.benchmarks_root) / "benchmarks.trace"
                if self.trace_file.exists():
                    self.trace_file.unlink()

                self.replay_tests_dir = Path(
                    tempfile.mkdtemp(prefix="codeflash_replay_tests_", dir=self.args.benchmarks_root)
                )
                trace_benchmarks_pytest(
                    self.args.benchmarks_root, self.args.tests_root, self.args.project_root, self.trace_file
                )  # Run all tests that use pytest-benchmark
                replay_count = generate_replay_test(self.trace_file, self.replay_tests_dir)
                if replay_count == 0:
                    logger.info(
                        f"No valid benchmarks found in {self.args.benchmarks_root} for functions to optimize, continuing optimization"
                    )
                else:
                    function_benchmark_timings = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(
                        self.trace_file
                    )
                    total_benchmark_timings = CodeFlashBenchmarkPlugin.get_benchmark_timings(self.trace_file)
                    function_to_results = validate_and_format_benchmark_table(
                        function_benchmark_timings, total_benchmark_timings
                    )
                    print_benchmark_table(function_to_results)
            except Exception as e:
                logger.info(f"Error while tracing existing benchmarks: {e}")
                logger.info("Information on existing benchmarks will not be available for this run.")
            finally:
                # Restore original source code
                for file in file_path_to_source_code:
                    with file.open("w", encoding="utf8") as f:
                        f.write(file_path_to_source_code[file])
        console.rule()
        return function_benchmark_timings, total_benchmark_timings

    def get_optimizable_functions(self) -> tuple[dict[Path, list[FunctionToOptimize]], int, Path | None]:
        """Discover functions to optimize."""
        from codeflash.discovery.functions_to_optimize import get_functions_to_optimize

        return get_functions_to_optimize(
            optimize_all=self.args.all,
            replay_test=self.args.replay_test,
            file=self.args.file,
            only_get_this_function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=self.args.project_root,
            module_root=self.args.module_root,
            previous_checkpoint_functions=self.args.previous_checkpoint_functions,
        )

    def create_function_optimizer(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_optimize_ast: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        function_to_tests: dict[str, set[FunctionCalledInTest]] | None = None,
        function_to_optimize_source_code: str | None = "",
        function_benchmark_timings: dict[str, dict[BenchmarkKey, float]] | None = None,
        total_benchmark_timings: dict[BenchmarkKey, float] | None = None,
        original_module_ast: ast.Module | None = None,
        original_module_path: Path | None = None,
    ) -> FunctionOptimizer | None:
        from codeflash.code_utils.static_analysis import get_first_top_level_function_or_method_ast
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        if function_to_optimize_ast is None and original_module_ast is not None:
            function_to_optimize_ast = get_first_top_level_function_or_method_ast(
                function_to_optimize.function_name, function_to_optimize.parents, original_module_ast
            )
            if function_to_optimize_ast is None:
                logger.info(
                    f"Function {function_to_optimize.qualified_name} not found in {original_module_path}.\n"
                    f"Skipping optimization."
                )
                return None

        qualified_name_w_module = function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root)

        function_specific_timings = None
        if (
            hasattr(self.args, "benchmark")
            and self.args.benchmark
            and function_benchmark_timings
            and qualified_name_w_module in function_benchmark_timings
            and total_benchmark_timings
        ):
            function_specific_timings = function_benchmark_timings[qualified_name_w_module]

        return FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=self.test_cfg,
            function_to_optimize_source_code=function_to_optimize_source_code,
            function_to_tests=function_to_tests,
            function_to_optimize_ast=function_to_optimize_ast,
            aiservice_client=self.aiservice_client,
            args=self.args,
            function_benchmark_timings=function_specific_timings,
            total_benchmark_timings=total_benchmark_timings if function_specific_timings else None,
            replay_tests_dir=self.replay_tests_dir,
        )

    def prepare_module_for_optimization(
        self, original_module_path: Path
    ) -> tuple[dict[Path, ValidCode], ast.Module] | None:
        from codeflash.code_utils.code_replacer import normalize_code, normalize_node
        from codeflash.code_utils.static_analysis import analyze_imported_modules

        logger.info(f"loading|Examining file {original_module_path!s}")
        console.rule()

        original_module_code: str = original_module_path.read_text(encoding="utf8")
        try:
            original_module_ast = ast.parse(original_module_code)
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing code in {original_module_path}: {e}")
            logger.info("Skipping optimization due to file error.")
            return None
        normalized_original_module_code = ast.unparse(normalize_node(original_module_ast))
        validated_original_code: dict[Path, ValidCode] = {
            original_module_path: ValidCode(
                source_code=original_module_code, normalized_code=normalized_original_module_code
            )
        }

        imported_module_analyses = analyze_imported_modules(
            original_module_code, original_module_path, self.args.project_root
        )

        has_syntax_error = False
        for analysis in imported_module_analyses:
            callee_original_code = analysis.file_path.read_text(encoding="utf8")
            try:
                normalized_callee_original_code = normalize_code(callee_original_code)
            except SyntaxError as e:
                logger.warning(f"Syntax error parsing code in callee module {analysis.file_path}: {e}")
                logger.info("Skipping optimization due to helper file error.")
                has_syntax_error = True
                break
            validated_original_code[analysis.file_path] = ValidCode(
                source_code=callee_original_code, normalized_code=normalized_callee_original_code
            )

        if has_syntax_error:
            return None

        return validated_original_code, original_module_ast

    def discover_tests(
        self, file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]]
    ) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
        from codeflash.discovery.discover_unit_tests import discover_unit_tests

        console.rule()
        start_time = time.time()
        logger.info("lsp,loading|Discovering existing function tests...")
        function_to_tests, num_discovered_tests, num_discovered_replay_tests = discover_unit_tests(
            self.test_cfg, file_to_funcs_to_optimize=file_to_funcs_to_optimize
        )
        console.rule()
        logger.info(
            f"Discovered {num_discovered_tests} existing unit tests and {num_discovered_replay_tests} replay tests in {(time.time() - start_time):.1f}s at {self.test_cfg.tests_root}"
        )
        console.rule()
        ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})
        return function_to_tests, num_discovered_tests

    def display_global_ranking(
        self, globally_ranked: list[tuple[Path, FunctionToOptimize]], ranker: FunctionRanker, show_top_n: int = 15
    ) -> None:
        from rich.table import Table

        if not globally_ranked:
            return

        # Show top N functions
        display_count = min(show_top_n, len(globally_ranked))

        table = Table(
            title=f"Function Ranking (Top {display_count} of {len(globally_ranked)})",
            title_style="bold cyan",
            border_style="cyan",
            show_lines=False,
        )

        table.add_column("Priority", style="bold yellow", justify="center", width=8)
        table.add_column("Function", style="cyan", width=40)
        table.add_column("File", style="dim", width=25)
        table.add_column("Addressable Time", justify="right", style="green", width=12)
        table.add_column("Impact", justify="center", style="bold", width=8)

        # Get addressable time for display
        for i, (file_path, func) in enumerate(globally_ranked[:display_count], 1):
            addressable_time = ranker.get_function_addressable_time(func)

            # Format function name
            func_name = func.qualified_name
            if len(func_name) > 38:
                func_name = func_name[:35] + "..."

            # Format file name
            file_name = file_path.name
            if len(file_name) > 23:
                file_name = "..." + file_name[-20:]

            # Format addressable time
            if addressable_time >= 1e9:
                time_display = f"{addressable_time / 1e9:.2f}s"
            elif addressable_time >= 1e6:
                time_display = f"{addressable_time / 1e6:.1f}ms"
            elif addressable_time >= 1e3:
                time_display = f"{addressable_time / 1e3:.1f}µs"
            else:
                time_display = f"{addressable_time:.0f}ns"

            # Impact indicator
            if i <= 5:
                impact = "🔥"
                impact_style = "bold red"
            elif i <= 10:
                impact = "⚡"
                impact_style = "bold yellow"
            else:
                impact = "💡"
                impact_style = "bold blue"

            table.add_row(f"#{i}", func_name, file_name, time_display, impact, style=impact_style if i <= 5 else None)

        console.print(table)

        if len(globally_ranked) > display_count:
            console.print(f"[dim]... and {len(globally_ranked) - display_count} more functions[/dim]")

    def rank_all_functions_globally(
        self, file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]], trace_file_path: Path | None
    ) -> list[tuple[Path, FunctionToOptimize]]:
        """Rank all functions globally across all files based on trace data.

        This performs global ranking instead of per-file ranking, ensuring that
        high-impact functions are optimized first regardless of which file they're in.

        Args:
            file_to_funcs_to_optimize: Mapping of file paths to functions to optimize
            trace_file_path: Path to trace file with performance data

        Returns:
            List of (file_path, function) tuples in globally ranked order by addressable time.
            If no trace file or ranking fails, returns functions in original file order.

        """
        all_functions: list[tuple[Path, FunctionToOptimize]] = []
        for file_path, functions in file_to_funcs_to_optimize.items():
            all_functions.extend((file_path, func) for func in functions)

        # If no trace file, return in original order
        if not trace_file_path or not trace_file_path.exists():
            logger.debug("No trace file available, using original function order")
            return all_functions

        try:
            from codeflash.benchmarking.function_ranker import FunctionRanker

            console.rule()
            logger.info("loading|Ranking functions globally by performance impact...")
            console.rule()
            # Create ranker with trace data
            ranker = FunctionRanker(trace_file_path)

            # Extract just the functions for ranking (without file paths)
            functions_only = [func for _, func in all_functions]

            # Rank globally
            ranked_functions = ranker.rank_functions(functions_only)

            # Reconstruct with file paths by looking up original file for each ranked function
            # Build reverse mapping: function -> file path
            # Since FunctionToOptimize is unhashable (contains list), we compare by identity
            func_to_file_map = {}
            for file_path, func in all_functions:
                # Use a tuple of unique identifiers as the key
                key: tuple[Path, str, int | None] = (func.file_path, func.qualified_name, func.starting_line)
                func_to_file_map[key] = file_path
            globally_ranked = []
            for func in ranked_functions:
                key = (func.file_path, func.qualified_name, func.starting_line)
                file_path = func_to_file_map.get(key)
                if file_path:
                    globally_ranked.append((file_path, func))

            console.rule()
            logger.info(
                f"Globally ranked {len(ranked_functions)} functions by addressable time "
                f"(filtered {len(functions_only) - len(ranked_functions)} low-importance functions)"
            )

            # Display ranking table for user visibility
            self.display_global_ranking(globally_ranked, ranker)
            console.rule()

        except Exception as e:
            logger.warning(f"Could not perform global ranking: {e}")
            logger.debug("Falling back to original function order")
            return all_functions
        else:
            return globally_ranked

    def run(self) -> None:
        from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint

        ph("cli-optimize-run-start")
        logger.info("Running optimizer.")
        console.rule()
        if not env_utils.ensure_codeflash_api_key():
            return
        if self.args.no_draft and is_pr_draft():
            logger.warning("PR is in draft mode, skipping optimization")
            return

        if self.args.worktree:
            self.worktree_mode()

        if not self.args.replay_test and self.test_cfg.tests_root.exists():
            leftover_trace_files = list(self.test_cfg.tests_root.glob("*.trace"))
            if leftover_trace_files:
                logger.debug(f"Cleaning up {len(leftover_trace_files)} leftover trace file(s) from previous runs")
                cleanup_paths(leftover_trace_files)

        cleanup_paths(Optimizer.find_leftover_instrumented_test_files(self.test_cfg.tests_root))

        function_optimizer = None
        file_to_funcs_to_optimize, num_optimizable_functions, trace_file_path = self.get_optimizable_functions()
        if self.args.all:
            three_min_in_ns = int(1.8e11)
            console.rule()
            pr_message = (
                "\nCodeflash will keep opening pull requests as it finds optimizations." if not self.args.no_pr else ""
            )
            logger.info(
                f"It might take about {humanize_runtime(num_optimizable_functions * three_min_in_ns)} to fully optimize this project.{pr_message}"
            )

        function_benchmark_timings, total_benchmark_timings = self.run_benchmarks(
            file_to_funcs_to_optimize, num_optimizable_functions
        )
        optimizations_found: int = 0
        self.test_cfg.concolic_test_root_dir = Path(
            tempfile.mkdtemp(dir=self.args.tests_root, prefix="codeflash_concolic_")
        )
        try:
            ph("cli-optimize-functions-to-optimize", {"num_functions": num_optimizable_functions})
            if num_optimizable_functions == 0:
                logger.info("No functions found to optimize. Exiting…")
                return

            function_to_tests, _ = self.discover_tests(file_to_funcs_to_optimize)
            if self.args.all:
                self.functions_checkpoint = CodeflashRunCheckpoint(self.args.module_root)

            # GLOBAL RANKING: Rank all functions together before optimizing
            globally_ranked_functions = self.rank_all_functions_globally(file_to_funcs_to_optimize, trace_file_path)
            # Cache for module preparation (avoid re-parsing same files)
            prepared_modules: dict[Path, tuple[dict[Path, ValidCode], ast.Module]] = {}

            # Optimize functions in globally ranked order
            for i, (original_module_path, function_to_optimize) in enumerate(globally_ranked_functions):
                # Prepare module if not already cached
                if original_module_path not in prepared_modules:
                    module_prep_result = self.prepare_module_for_optimization(original_module_path)
                    if module_prep_result is None:
                        logger.warning(f"Skipping functions in {original_module_path} due to preparation error")
                        continue
                    prepared_modules[original_module_path] = module_prep_result

                validated_original_code, original_module_ast = prepared_modules[original_module_path]

                function_iterator_count = i + 1
                logger.info(
                    f"Optimizing function {function_iterator_count} of {len(globally_ranked_functions)}: "
                    f"{function_to_optimize.qualified_name} (in {original_module_path.name})"
                )
                console.rule()
                function_optimizer = None
                try:
                    function_optimizer = self.create_function_optimizer(
                        function_to_optimize,
                        function_to_tests=function_to_tests,
                        function_to_optimize_source_code=validated_original_code[original_module_path].source_code,
                        function_benchmark_timings=function_benchmark_timings,
                        total_benchmark_timings=total_benchmark_timings,
                        original_module_ast=original_module_ast,
                        original_module_path=original_module_path,
                    )
                    if function_optimizer is None:
                        continue

                    self.current_function_optimizer = (
                        function_optimizer  # needed to clean up from the outside of this function
                    )
                    best_optimization = function_optimizer.optimize_function()
                    if self.functions_checkpoint:
                        self.functions_checkpoint.add_function_to_checkpoint(
                            function_to_optimize.qualified_name_with_modules_from_root(self.args.project_root)
                        )
                    if is_successful(best_optimization):
                        optimizations_found += 1
                        # create a diff patch for successful optimization
                        if self.current_worktree:
                            best_opt = best_optimization.unwrap()
                            read_writable_code = best_opt.code_context.read_writable_code
                            relative_file_paths = [
                                code_string.file_path for code_string in read_writable_code.code_strings
                            ]
                            patch_path = create_diff_patch_from_worktree(
                                self.current_worktree, relative_file_paths, fto_name=function_to_optimize.qualified_name
                            )
                            self.patch_files.append(patch_path)
                            if i < len(globally_ranked_functions) - 1:
                                _, next_func = globally_ranked_functions[i + 1]
                                create_worktree_snapshot_commit(
                                    self.current_worktree, f"Optimizing {next_func.qualified_name}"
                                )
                    else:
                        logger.warning(best_optimization.failure())
                        console.rule()
                        continue
                finally:
                    if function_optimizer is not None:
                        function_optimizer.executor.shutdown(wait=True)
                        function_optimizer.cleanup_generated_files()

            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if len(self.patch_files) > 0:
                logger.info(
                    f"Created {len(self.patch_files)} patch(es) ({[str(patch_path) for patch_path in self.patch_files]})"
                )
            if self.functions_checkpoint:
                self.functions_checkpoint.cleanup()
            if hasattr(self.args, "command") and self.args.command == "optimize":
                self.cleanup_replay_tests()
            if optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
                response = send_completion_email()  # TODO: Include more details in the email
                if response.ok:
                    logger.info("✅ Completion email sent successfully.")
                else:
                    logger.warning("⚠️ Failed to send completion email. Status")
        finally:
            if function_optimizer:
                function_optimizer.cleanup_generated_files()

            self.cleanup_temporary_paths()

    @staticmethod
    def find_leftover_instrumented_test_files(test_root: Path) -> list[Path]:
        """Search for all paths within the test_root that match the following patterns.

        - 'test.*__perf_test_{0,1}.py'
        - 'test_.*__unit_test_{0,1}.py'
        - 'test_.*__perfinstrumented.py'
        - 'test_.*__perfonlyinstrumented.py'
        Returns a list of matching file paths.
        """
        import re

        pattern = re.compile(
            r"(?:test.*__perf_test_\d?\.py|test_.*__unit_test_\d?\.py|test_.*__perfinstrumented\.py|test_.*__perfonlyinstrumented\.py)$"
        )

        return [
            file_path for file_path in test_root.rglob("*") if file_path.is_file() and pattern.match(file_path.name)
        ]

    def cleanup_replay_tests(self) -> None:
        paths_to_cleanup = []
        if self.replay_tests_dir and self.replay_tests_dir.exists():
            logger.debug(f"Cleaning up replay tests directory: {self.replay_tests_dir}")
            paths_to_cleanup.append(self.replay_tests_dir)
        if self.trace_file and self.trace_file.exists():
            logger.debug(f"Cleaning up trace file: {self.trace_file}")
            paths_to_cleanup.append(self.trace_file)
        if paths_to_cleanup:
            cleanup_paths(paths_to_cleanup)

    def cleanup_temporary_paths(self) -> None:
        if hasattr(get_run_tmp_file, "tmpdir"):
            get_run_tmp_file.tmpdir.cleanup()
            del get_run_tmp_file.tmpdir

        if self.current_worktree:
            remove_worktree(self.current_worktree)
            return

        if self.current_function_optimizer:
            self.current_function_optimizer.cleanup_generated_files()
        paths_to_cleanup = [self.test_cfg.concolic_test_root_dir, self.replay_tests_dir]
        if self.trace_file:
            paths_to_cleanup.append(self.trace_file)
        if self.test_cfg.tests_root.exists():
            for trace_file in self.test_cfg.tests_root.glob("*.trace"):
                if trace_file not in paths_to_cleanup:
                    paths_to_cleanup.append(trace_file)
        cleanup_paths(paths_to_cleanup)

    def worktree_mode(self) -> None:
        if self.current_worktree:
            return

        if check_running_in_git_repo(self.args.module_root):
            worktree_dir = create_detached_worktree(self.args.module_root)
            if worktree_dir is None:
                logger.warning("Failed to create worktree. Skipping optimization.")
                return
            self.current_worktree = worktree_dir
            self.mirror_paths_for_worktree_mode(worktree_dir)
            # make sure the tests dir is created in the worktree, this can happen if the original tests dir is empty
            Path(self.args.tests_root).mkdir(parents=True, exist_ok=True)

    def mirror_paths_for_worktree_mode(self, worktree_dir: Path) -> None:
        original_args = copy.deepcopy(self.args)
        original_test_cfg = copy.deepcopy(self.test_cfg)
        self.original_args_and_test_cfg = (original_args, original_test_cfg)

        original_git_root = git_root_dir()

        # mirror project_root
        self.args.project_root = mirror_path(self.args.project_root, original_git_root, worktree_dir)
        self.test_cfg.project_root_path = mirror_path(self.test_cfg.project_root_path, original_git_root, worktree_dir)

        # mirror module_root
        self.args.module_root = mirror_path(self.args.module_root, original_git_root, worktree_dir)

        # mirror target file
        if self.args.file:
            self.args.file = mirror_path(self.args.file, original_git_root, worktree_dir)

        if self.args.all:
            # the args.all path is the same as module_root.
            self.args.all = mirror_path(self.args.all, original_git_root, worktree_dir)

        # mirror tests root
        self.args.tests_root = mirror_path(self.args.tests_root, original_git_root, worktree_dir)
        self.test_cfg.tests_root = mirror_path(self.test_cfg.tests_root, original_git_root, worktree_dir)

        # mirror tests project root
        self.args.test_project_root = mirror_path(self.args.test_project_root, original_git_root, worktree_dir)
        self.test_cfg.tests_project_rootdir = mirror_path(
            self.test_cfg.tests_project_rootdir, original_git_root, worktree_dir
        )

        # mirror benchmarks root paths
        if self.args.benchmarks_root:
            self.args.benchmarks_root = mirror_path(self.args.benchmarks_root, original_git_root, worktree_dir)
        if self.test_cfg.benchmark_tests_root:
            self.test_cfg.benchmark_tests_root = mirror_path(
                self.test_cfg.benchmark_tests_root, original_git_root, worktree_dir
            )


def mirror_path(path: Path, src_root: Path, dest_root: Path) -> Path:
    relative_path = path.relative_to(src_root)
    return dest_root / relative_path


def run_with_args(args: Namespace) -> None:
    optimizer = None
    try:
        optimizer = Optimizer(args)
        optimizer.run()
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Cleaning up and exiting, please wait…")
        if optimizer:
            optimizer.cleanup_temporary_paths()

        raise SystemExit from None
