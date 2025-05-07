from __future__ import annotations

import ast
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.benchmarking.instrument_codeflash_trace import instrument_codeflash_trace_decorator
from codeflash.benchmarking.plugin.plugin import CodeFlashBenchmarkPlugin
from codeflash.benchmarking.replay_test import generate_replay_test
from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest
from codeflash.benchmarking.utils import print_benchmark_table, validate_and_format_benchmark_table
from codeflash.cli_cmds.console import console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint, ask_should_use_checkpoint_get_functions
from codeflash.code_utils.code_replacer import normalize_code, normalize_node
from codeflash.code_utils.code_utils import cleanup_paths, get_run_tmp_file
from codeflash.code_utils.static_analysis import analyze_imported_modules, get_first_top_level_function_or_method_ast
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.discovery.functions_to_optimize import get_functions_to_optimize
from codeflash.either import is_successful
from codeflash.models.models import BenchmarkKey, TestType, ValidCode
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionCalledInTest


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
        self.file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]] | None = None
        self.num_optimizable_functions: int | None = None
        self.function_to_tests: dict[str, list[FunctionCalledInTest]] | None = None
        self.num_discovered_tests: int | None = None

    def create_function_optimizer(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_optimize_ast: ast.FunctionDef | None = None,
        function_to_tests: dict[str, list[FunctionCalledInTest]] | None = None,
        function_to_optimize_source_code: str | None = "",
        all_function_benchmark_timings: dict[str, dict[BenchmarkKey, int]] | None = None,
        overall_total_benchmark_timings: dict[BenchmarkKey, int] | None = None,
        qualified_name_w_module_for_benchmarks: str | None = None,
    ) -> FunctionOptimizer:
        specific_function_timings_to_pass = None
        total_benchmark_timings_to_pass = None

        if (
            self.args.benchmark
            and all_function_benchmark_timings
            and qualified_name_w_module_for_benchmarks
            and qualified_name_w_module_for_benchmarks in all_function_benchmark_timings
            and overall_total_benchmark_timings
        ):
            specific_function_timings_to_pass = all_function_benchmark_timings[qualified_name_w_module_for_benchmarks]
            total_benchmark_timings_to_pass = overall_total_benchmark_timings

        return FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=self.test_cfg,
            function_to_optimize_source_code=function_to_optimize_source_code,
            function_to_tests=function_to_tests,
            function_to_optimize_ast=function_to_optimize_ast,
            aiservice_client=self.aiservice_client,
            args=self.args,
            function_benchmark_timings=specific_function_timings_to_pass,
            total_benchmark_timings=total_benchmark_timings_to_pass,
            replay_tests_dir=self.replay_tests_dir,
        )

    def discover_functions(self) -> None:
        previous_checkpoint_functions = ask_should_use_checkpoint_get_functions(self.args)
        self.file_to_funcs_to_optimize, self.num_optimizable_functions = get_functions_to_optimize(
            optimize_all=self.args.all,
            replay_test=self.args.replay_test,
            file=self.args.file,
            only_get_this_function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=self.args.project_root,
            module_root=self.args.module_root,
            previous_checkpoint_functions=previous_checkpoint_functions,
        )

    def discover_unit_tests(self) -> None:
        console.rule()
        start_time = time.time()
        self.function_to_tests = discover_unit_tests(self.test_cfg)
        self.num_discovered_tests = sum([len(value) for value in self.function_to_tests.values()])
        console.rule()
        logger.info(
            f"Discovered {self.num_discovered_tests} existing unit tests in {(time.time() - start_time):.1f}s at {self.test_cfg.tests_root}"
        )
        console.rule()
        ph("cli-optimize-discovered-tests", {"num_tests": self.num_discovered_tests})

    def _run_benchmarks(self) -> tuple[dict[str, dict[BenchmarkKey, int]], dict[BenchmarkKey, int], Path | None]:
        function_benchmark_timings: dict[str, dict[BenchmarkKey, int]] = {}
        total_benchmark_timings: dict[BenchmarkKey, int] = {}
        trace_file: Path | None = None
        if self.args.benchmark and self.num_optimizable_functions > 0:
            with progress_bar(f"Running benchmarks in {self.args.benchmarks_root}", transient=True):
                file_path_to_source_code = defaultdict(str)
                for file_path in self.file_to_funcs_to_optimize:
                    with file_path.open("r", encoding="utf8") as f:
                        file_path_to_source_code[file_path] = f.read()
                try:
                    instrument_codeflash_trace_decorator(self.file_to_funcs_to_optimize)
                    trace_file = Path(self.args.benchmarks_root) / "benchmarks.trace"
                    if trace_file.exists():
                        trace_file.unlink()

                    self.replay_tests_dir = Path(
                        tempfile.mkdtemp(prefix="codeflash_replay_tests_", dir=self.args.benchmarks_root)
                    )
                    trace_benchmarks_pytest(
                        self.args.benchmarks_root, self.args.tests_root, self.args.project_root, trace_file
                    )
                    replay_count = generate_replay_test(trace_file, self.replay_tests_dir)
                    if replay_count == 0:
                        logger.info(
                            f"No valid benchmarks found in {self.args.benchmarks_root} for functions to optimize, "
                            "continuing optimization"
                        )
                    else:
                        function_benchmark_timings = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(trace_file)
                        total_benchmark_timings = CodeFlashBenchmarkPlugin.get_benchmark_timings(trace_file)
                        function_to_results = validate_and_format_benchmark_table(
                            function_benchmark_timings, total_benchmark_timings
                        )
                        print_benchmark_table(function_to_results)
                except Exception as e:  # TODO: Consider more specific exception handling
                    logger.info(f"Error while tracing existing benchmarks: {e}")
                    logger.info("Information on existing benchmarks will not be available for this run.")
                finally:
                    for file_path in file_path_to_source_code:
                        with file_path.open("w", encoding="utf8") as f:
                            f.write(file_path_to_source_code[file_path])
        return function_benchmark_timings, total_benchmark_timings, trace_file

    def run(self) -> None:
        ph("cli-optimize-run-start")
        logger.info("Running optimizer.")
        console.rule()
        if not env_utils.ensure_codeflash_api_key():
            return
        function_optimizer = None

        self.discover_functions()

        function_benchmark_timings, total_benchmark_timings, trace_file = self._run_benchmarks()
        optimizations_found: int = 0
        function_iterator_count: int = 0
        if self.args.test_framework == "pytest":
            self.test_cfg.concolic_test_root_dir = Path(
                tempfile.mkdtemp(dir=self.args.tests_root, prefix="codeflash_concolic_")
            )
        try:
            ph("cli-optimize-functions-to-optimize", {"num_functions": self.num_optimizable_functions})
            if self.num_optimizable_functions == 0:
                logger.info("No functions found to optimize. Exiting…")
                return

            self.discover_unit_tests()

            for original_module_path in self.file_to_funcs_to_optimize:
                logger.info(f"Examining file {original_module_path!s}…")
                console.rule()

                original_module_code: str = original_module_path.read_text(encoding="utf8")
                try:
                    original_module_ast = ast.parse(original_module_code)
                except SyntaxError as e:
                    logger.warning(f"Syntax error parsing code in {original_module_path}: {e}")
                    logger.info("Skipping optimization due to file error.")
                    continue
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
                    continue

                for function_to_optimize in self.file_to_funcs_to_optimize[original_module_path]:
                    function_iterator_count += 1
                    logger.info(
                        f"Optimizing function {function_iterator_count} of {self.num_optimizable_functions}: "
                        f"{function_to_optimize.qualified_name}"
                    )
                    console.rule()
                    if not (
                        function_to_optimize_ast := get_first_top_level_function_or_method_ast(
                            function_to_optimize.function_name, function_to_optimize.parents, original_module_ast
                        )
                    ):
                        logger.info(
                            f"Function {function_to_optimize.qualified_name} not found in {original_module_path}.\n"
                            f"Skipping optimization."
                        )
                        continue
                    qualified_name_w_module = function_to_optimize.qualified_name_with_modules_from_root(
                        self.args.project_root
                    )
                    function_optimizer = self.create_function_optimizer(
                        function_to_optimize,
                        function_to_optimize_ast,
                        self.function_to_tests,
                        validated_original_code[original_module_path].source_code,
                        function_benchmark_timings,
                        total_benchmark_timings,
                        qualified_name_w_module,
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
            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if self.functions_checkpoint:
                self.functions_checkpoint.cleanup()
            if optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
        finally:
            if function_optimizer:
                function_optimizer.cleanup_generated_files()
                trace_file.unlink(missing_ok=True)


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
