from __future__ import annotations

import copy
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.api.cfapi import send_completion_email
from codeflash.cli_cmds.console import call_graph_live_display, console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import cleanup_paths, get_run_tmp_file
from codeflash.code_utils.config_consts import HIGH_EFFORT_TOP_N, EffortLevel
from codeflash.code_utils.env_utils import get_pr_number, is_pr_draft
from codeflash.code_utils.git_utils import check_running_in_git_repo, git_root_dir, mirror_path
from codeflash.code_utils.git_worktree_utils import (
    create_detached_worktree,
    create_diff_patch_from_worktree,
    create_worktree_snapshot_commit,
    remove_worktree,
)
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.languages import set_current_language
from codeflash.lsp.helpers import is_subagent_mode
from codeflash.telemetry.posthog_cf import ph
from codeflash_core.config import TestConfig
from codeflash_python.plugin import PythonPlugin

if TYPE_CHECKING:
    import ast
    from argparse import Namespace

    from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint
    from codeflash_core.models import FunctionToOptimize
    from codeflash_python.context.types import DependencyResolver
    from codeflash_python.function_optimizer import FunctionOptimizer
    from codeflash_python.models.models import BenchmarkKey, FunctionCalledInTest, ValidCode


class Optimizer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            tests_project_rootdir=args.test_project_root,
            project_root=args.project_root,
            test_command=args.pytest_cmd if hasattr(args, "pytest_cmd") and args.pytest_cmd else "pytest",
            benchmark_tests_root=args.benchmarks_root if "benchmark" in args and "benchmarks_root" in args else None,
        )

        self.plugin = PythonPlugin(args.project_root)
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

        from codeflash_python.benchmarking.instrument_codeflash_trace import instrument_codeflash_trace_decorator
        from codeflash_python.benchmarking.plugin.plugin import CodeFlashBenchmarkPlugin
        from codeflash_python.benchmarking.replay_test import generate_replay_test
        from codeflash_python.benchmarking.trace_benchmarks import trace_benchmarks_pytest
        from codeflash_python.benchmarking.utils import print_benchmark_table, validate_and_format_benchmark_table

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
        from codeflash_python.discovery.functions_to_optimize import get_functions_to_optimize

        # In worktree mode for git-diff discovery, file paths come from the original repo
        # (via get_git_diff using cwd), but module_root/project_root have been mirrored to
        # the worktree. Use the original roots for filtering so path comparisons match,
        # then remap the discovered file paths to the worktree.
        project_root = self.args.project_root
        module_root = self.args.module_root
        use_original_roots = (
            self.current_worktree and self.original_args_and_test_cfg and not self.args.all and not self.args.file
        )
        if use_original_roots:
            assert self.original_args_and_test_cfg is not None
            original_args, _ = self.original_args_and_test_cfg
            project_root = original_args.project_root
            module_root = original_args.module_root

        result = get_functions_to_optimize(
            optimize_all=self.args.all,
            replay_test=self.args.replay_test,
            file=self.args.file,
            only_get_this_function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=project_root,
            module_root=module_root,
            previous_checkpoint_functions=self.args.previous_checkpoint_functions,
        )

        # Remap discovered file paths from the original repo to the worktree so
        # downstream optimization reads/writes happen in the worktree.
        if use_original_roots:
            import dataclasses

            assert self.current_worktree is not None
            original_git_root = git_root_dir()
            file_to_funcs, count, trace = result
            remapped: dict[Path, list[FunctionToOptimize]] = {}
            for file_path, funcs in file_to_funcs.items():
                new_path = mirror_path(Path(file_path), original_git_root, self.current_worktree)
                remapped[new_path] = [
                    dataclasses.replace(
                        func, file_path=mirror_path(func.file_path, original_git_root, self.current_worktree)
                    )
                    for func in funcs
                ]
            return remapped, count, trace

        return result

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

        from codeflash_python.function_optimizer import FunctionOptimizer as PythonFunctionOptimizer

        function_optimizer = PythonFunctionOptimizer(
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
            call_graph=call_graph,
            effort_override=effort_override,
        )
        if function_optimizer.function_to_optimize_ast is None and function_optimizer.requires_function_ast():
            logger.info(
                f"Function {function_to_optimize.qualified_name} not found in "
                f"{function_to_optimize.file_path}.\nSkipping optimization."
            )
            return None
        return function_optimizer

    def prepare_module_for_optimization(
        self, original_module_path: Path
    ) -> tuple[dict[Path, ValidCode], ast.Module | None] | None:
        logger.info(f"loading|Examining file {original_module_path!s}")
        console.rule()

        original_module_code: str = original_module_path.read_text(encoding="utf8")

        from codeflash_python.optimizer import prepare_python_module

        return prepare_python_module(original_module_code, original_module_path, self.args.project_root)

    def discover_tests(
        self, file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]]
    ) -> tuple[dict[str, set[FunctionCalledInTest]], int]:
        from codeflash_python.discovery.discover_unit_tests import discover_unit_tests

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

        self.plugin.cleanup_run(self.test_cfg.tests_root)

        function_optimizer = None
        file_to_funcs_to_optimize, num_optimizable_functions, trace_file_path = self.get_optimizable_functions()

        # Set language global singleton based on discovered functions
        if file_to_funcs_to_optimize:
            for funcs in file_to_funcs_to_optimize.values():
                if funcs and funcs[0].language:
                    set_current_language(funcs[0].language)
                    self.plugin.setup_test_config(self.test_cfg)
                    break

        if self.args.all:
            three_min_in_ns = int(1.8e11)
            console.rule()
            logger.info(
                f"It might take about {humanize_runtime(num_optimizable_functions * three_min_in_ns)} to fully optimize this project."
            )
            if not self.args.no_pr:
                logger.info("Codeflash will keep opening pull requests as it finds optimizations.")
            console.rule()

        function_benchmark_timings, total_benchmark_timings = self.run_benchmarks(
            file_to_funcs_to_optimize, num_optimizable_functions
        )

        # Build call graph index via the Python plugin (skips CI internally)
        if file_to_funcs_to_optimize:
            source_files = [f for f in file_to_funcs_to_optimize if f.suffix == ".py"]
            if source_files:
                with call_graph_live_display(len(source_files), project_root=self.args.project_root) as on_progress:
                    self.plugin.build_index(source_files, on_progress=on_progress)
                console.rule()

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
            if self.args.all and not self.args.subagent:
                self.functions_checkpoint = CodeflashRunCheckpoint(self.args.module_root)

            # Pre-compute test counts once for ranking and logging
            test_count_cache: dict[tuple[Path, str], int]
            if function_to_tests:
                from codeflash_python.discovery.discover_unit_tests import existing_unit_test_count

                test_count_cache = {
                    (fp, fn.qualified_name): existing_unit_test_count(fn, self.args.project_root, function_to_tests)
                    for fp, fns in file_to_funcs_to_optimize.items()
                    for fn in fns
                }
            else:
                test_count_cache = {}

            # GLOBAL RANKING: Rank all functions via the plugin (trace → dependency count → original order)
            all_functions = [func for funcs in file_to_funcs_to_optimize.values() for func in funcs]
            ranked_functions = self.plugin.rank_functions(
                all_functions, trace_file=trace_file_path, test_counts=test_count_cache
            )
            globally_ranked_functions: list[tuple[Path, FunctionToOptimize]] = [
                (func.file_path, func) for func in ranked_functions
            ]
            # Cache for module preparation (avoid re-parsing same files)
            prepared_modules: dict[Path, tuple[dict[Path, ValidCode], ast.Module | None]] = {}

            # Get dependency counts from the plugin (populated during rank_functions)
            dep_counts = self.plugin.get_dependency_counts()
            callee_counts: dict[tuple[Path, str], int] = {
                (fp, fn.qualified_name): dep_counts.get(fn.qualified_name, 0) for fp, fn in globally_ranked_functions
            }

            # Optimize functions in globally ranked order
            for i, (original_module_path, function_to_optimize) in enumerate(globally_ranked_functions):
                # Prepare module if not already cached
                if original_module_path not in prepared_modules:
                    module_prep_result = self.prepare_module_for_optimization(original_module_path)
                    if module_prep_result is None:
                        logger.warning(f"Skipping functions in {original_module_path} due to preparation error")
                        continue
                    prepared_modules[original_module_path] = module_prep_result

                validated_original_code, _original_module_ast = prepared_modules[original_module_path]

                function_iterator_count = i + 1
                line_suffix = f":{function_to_optimize.starting_line}" if function_to_optimize.starting_line else ""

                callee_count = callee_counts.get((original_module_path, function_to_optimize.qualified_name), 0)
                callee_suffix = f", {callee_count} callees" if callee_count else ""

                test_count = test_count_cache.get((original_module_path, function_to_optimize.qualified_name), 0)
                test_suffix = f", {test_count} tests" if test_count else ""

                effort_override: str | None = None
                if i < HIGH_EFFORT_TOP_N and self.args.effort == EffortLevel.MEDIUM.value:
                    effort_override = EffortLevel.HIGH.value
                    logger.debug(
                        f"Escalating effort for {function_to_optimize.qualified_name} from medium to high"
                        f" (top {HIGH_EFFORT_TOP_N} ranked)"
                    )

                logger.info(
                    f"Optimizing function {function_iterator_count} of {len(globally_ranked_functions)}: "
                    f"{function_to_optimize.qualified_name} (in {original_module_path}{line_suffix}{callee_suffix}{test_suffix})"
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
                        call_graph=self.plugin.get_call_graph_index(),
                        effort_override=effort_override,
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
                    if best_optimization.is_ok():
                        optimizations_found += 1
                        # create a diff patch for successful optimization
                        if self.current_worktree and not is_subagent_mode():
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
                        logger.warning(best_optimization.error)
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
            if is_subagent_mode():
                if optimizations_found == 0:
                    import sys

                    sys.stdout.write("<codeflash-summary>No optimizations found.</codeflash-summary>\n")
            elif optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
                response = send_completion_email()  # TODO: Include more details in the email
                if response.ok:
                    logger.info("✅ Completion email sent successfully.")
                else:
                    logger.warning("⚠️ Failed to send completion email. Status")
        finally:
            self.plugin.cleanup_run(self.test_cfg.tests_root)

            if function_optimizer:
                function_optimizer.cleanup_generated_files()

            self.cleanup_temporary_paths()

    @staticmethod
    def find_leftover_instrumented_test_files(test_root: Path) -> list[Path]:
        """Search for all paths within the test_root that match instrumented test file patterns.

        Python patterns:
        - 'test.*__perf_test_{0,1}.py'
        - 'test_.*__unit_test_{0,1}.py'
        - 'test_.*__perfinstrumented.py'
        - 'test_.*__perfonlyinstrumented.py'

        JavaScript/TypeScript patterns:
        - '*__perfinstrumented.test.{js,ts,jsx,tsx}'
        - '*__perfonlyinstrumented.test.{js,ts,jsx,tsx}'
        - '*__perfinstrumented.spec.{js,ts,jsx,tsx}'
        - '*__perfonlyinstrumented.spec.{js,ts,jsx,tsx}'

        Returns a list of matching file paths.
        """
        import re

        pattern = re.compile(
            r"(?:"
            # Python patterns
            r"test.*__perf_test_\d?\.py|test_.*__unit_test_\d?\.py|test_.*__perfinstrumented\.py|test_.*__perfonlyinstrumented\.py|"
            # JavaScript/TypeScript patterns (new naming with .test/.spec preserved)
            r".*__perfinstrumented\.(?:test|spec)\.(?:js|ts|jsx|tsx)|.*__perfonlyinstrumented\.(?:test|spec)\.(?:js|ts|jsx|tsx)|"
            # Java patterns
            r".*__perfinstrumented(?:_\d+)?\.java|.*__perfonlyinstrumented(?:_\d+)?\.java"
            r")$"
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
        from codeflash.languages.java.test_runner import CompilationCache

        CompilationCache.clear()

        if hasattr(get_run_tmp_file, "tmpdir"):
            get_run_tmp_file.tmpdir.cleanup()
            del get_run_tmp_file.tmpdir
        if hasattr(get_run_tmp_file, "tmpdir_path"):
            del get_run_tmp_file.tmpdir_path

        # Always clean up concolic test directory
        cleanup_paths([self.test_cfg.concolic_test_root_dir])

        if self.current_worktree:
            remove_worktree(self.current_worktree)
            return

        if self.current_function_optimizer:
            self.current_function_optimizer.cleanup_generated_files()
        paths_to_cleanup = [self.replay_tests_dir]
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
        self.test_cfg.project_root = mirror_path(self.test_cfg.project_root, original_git_root, worktree_dir)

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
