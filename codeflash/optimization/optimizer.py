from __future__ import annotations

import ast
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.cli_cmds.console import console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import cleanup_paths, get_run_tmp_file
from codeflash.code_utils.env_utils import get_pr_number, is_pr_draft
from codeflash.either import is_successful
from codeflash.models.models import ValidCode
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

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
            test_framework=args.test_framework,
            pytest_cmd=args.pytest_cmd,
            benchmark_tests_root=args.benchmarks_root if "benchmark" in args and "benchmarks_root" in args else None,
        )

        self.aiservice_client = AiServiceClient()
        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None
        self.replay_tests_dir = None
        self.functions_checkpoint: CodeflashRunCheckpoint | None = None
        self.current_function_being_optimized: FunctionToOptimize | None = None  # current only for the LSP
        self.current_function_optimizer: FunctionOptimizer | None = None

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
                trace_file = Path(self.args.benchmarks_root) / "benchmarks.trace"
                if trace_file.exists():
                    trace_file.unlink()

                self.replay_tests_dir = Path(
                    tempfile.mkdtemp(prefix="codeflash_replay_tests_", dir=self.args.benchmarks_root)
                )
                trace_benchmarks_pytest(
                    self.args.benchmarks_root, self.args.tests_root, self.args.project_root, trace_file
                )  # Run all tests that use pytest-benchmark
                replay_count = generate_replay_test(trace_file, self.replay_tests_dir)
                if replay_count == 0:
                    logger.info(
                        f"No valid benchmarks found in {self.args.benchmarks_root} for functions to optimize, continuing optimization"
                    )
                else:
                    function_benchmark_timings = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(trace_file)
                    total_benchmark_timings = CodeFlashBenchmarkPlugin.get_benchmark_timings(trace_file)
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
        function_to_optimize_ast: ast.FunctionDef | None = None,
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

        logger.info(f"Examining file {original_module_path!s}…")
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
        function_to_tests, num_discovered_tests = discover_unit_tests(
            self.test_cfg, file_to_funcs_to_optimize=file_to_funcs_to_optimize
        )
        console.rule()
        logger.info(
            f"Discovered {num_discovered_tests} existing unit tests in {(time.time() - start_time):.1f}s at {self.test_cfg.tests_root}"
        )
        console.rule()
        ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})
        return function_to_tests, num_discovered_tests

    def run(self) -> None:
        from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint

        ph("cli-optimize-run-start")
        logger.info("Running optimizer.")
        console.rule()
        if not env_utils.ensure_codeflash_api_key():
            return
        if not env_utils.check_formatter_installed(self.args.formatter_cmds):
            return
        if self.args.no_draft and is_pr_draft():
            logger.warning("PR is in draft mode, skipping optimization")
            return
        cleanup_paths(Optimizer.find_leftover_instrumented_test_files(self.test_cfg.tests_root))

        function_optimizer = None
        file_to_funcs_to_optimize, num_optimizable_functions, trace_file_path = self.get_optimizable_functions()
        function_benchmark_timings, total_benchmark_timings = self.run_benchmarks(
            file_to_funcs_to_optimize, num_optimizable_functions
        )
        optimizations_found: int = 0
        function_iterator_count: int = 0
        if self.args.test_framework == "pytest":
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

            for original_module_path in file_to_funcs_to_optimize:
                module_prep_result = self.prepare_module_for_optimization(original_module_path)
                if module_prep_result is None:
                    continue

                validated_original_code, original_module_ast = module_prep_result

                functions_to_optimize = file_to_funcs_to_optimize[original_module_path]
                if trace_file_path and trace_file_path.exists() and len(functions_to_optimize) > 1:
                    try:
                        from codeflash.benchmarking.function_ranker import FunctionRanker

                        ranker = FunctionRanker(trace_file_path)
                        functions_to_optimize = ranker.rank_functions(functions_to_optimize)
                        logger.info(
                            f"Ranked {len(functions_to_optimize)} functions by performance impact in {original_module_path}"
                        )
                        console.rule()
                    except Exception as e:
                        logger.debug(f"Could not rank functions in {original_module_path}: {e}")

                for function_to_optimize in functions_to_optimize:
                    function_iterator_count += 1
                    logger.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions}: "
                        f"{function_to_optimize.qualified_name}"
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
                        else:
                            logger.warning(best_optimization.failure())
                            console.rule()
                            continue
                    finally:
                        if function_optimizer is not None:
                            function_optimizer.cleanup_generated_files()

            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if self.functions_checkpoint:
                self.functions_checkpoint.cleanup()
            if hasattr(self.args, "command") and self.args.command == "optimize":
                self.cleanup_replay_tests()
            if optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
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
        if self.replay_tests_dir and self.replay_tests_dir.exists():
            logger.debug(f"Cleaning up replay tests directory: {self.replay_tests_dir}")
            cleanup_paths([self.replay_tests_dir])

    def cleanup_temporary_paths(self) -> None:
        if self.current_function_optimizer:
            self.current_function_optimizer.cleanup_generated_files()

        if hasattr(get_run_tmp_file, "tmpdir"):
            get_run_tmp_file.tmpdir.cleanup()
            del get_run_tmp_file.tmpdir
        cleanup_paths([self.test_cfg.concolic_test_root_dir, self.replay_tests_dir])


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
