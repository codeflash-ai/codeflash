from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.agents.coordinator import AgentCoordinator
from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import cleanup_paths
from codeflash.either import is_successful
from codeflash.languages import is_javascript, set_current_language
from codeflash.state.store import StateStore
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class AgenticOptimizer:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.state_store = StateStore()
        self.coordinator = AgentCoordinator(state_store=self.state_store)

        self.test_cfg = TestConfig(
            tests_root=args.tests_root,
            tests_project_rootdir=args.test_project_root,
            project_root_path=args.project_root,
            pytest_cmd=args.pytest_cmd if hasattr(args, "pytest_cmd") and args.pytest_cmd else "pytest",
            benchmark_tests_root=args.benchmarks_root if "benchmark" in args and "benchmarks_root" in args else None,
        )

    def run(self) -> None:
        ph("cli-agentic-optimize-run-start")
        logger.info("Running agentic optimizer.")
        console.rule()

        if not env_utils.ensure_codeflash_api_key():
            return

        self.coordinator.register_agents()

        file_to_funcs, num_functions, trace_file_path = self._get_optimizable_functions()

        if file_to_funcs:
            for file_path, funcs in file_to_funcs.items():
                if funcs and funcs[0].language:
                    set_current_language(funcs[0].language)
                    self.test_cfg.set_language(funcs[0].language)
                    if is_javascript():
                        self.test_cfg.js_project_root = self._find_js_project_root(file_path)
                    break

        if num_functions == 0:
            logger.info("No functions found to optimize. Exiting…")
            return

        function_to_tests = self._discover_tests(file_to_funcs)

        self.test_cfg.concolic_test_root_dir = Path(
            tempfile.mkdtemp(dir=self.args.tests_root, prefix="codeflash_concolic_")
        )

        try:
            optimizations_found = 0

            globally_ranked = self._rank_functions(file_to_funcs, trace_file_path)

            for i, (file_path, function_to_optimize) in enumerate(globally_ranked):
                logger.info(
                    f"Optimizing function {i + 1} of {len(globally_ranked)}: "
                    f"{function_to_optimize.qualified_name} (in {file_path.name})"
                )
                console.rule()

                result = self.coordinator.run_optimization_pipeline(
                    function_to_optimize=function_to_optimize,
                    test_cfg=self.test_cfg,
                    args=self.args,
                    function_to_tests=function_to_tests,
                )

                if is_successful(result):
                    optimizations_found += 1
                    logger.info(f"Successfully optimized {function_to_optimize.qualified_name}")
                else:
                    logger.warning(f"Failed to optimize {function_to_optimize.qualified_name}: {result.failure()}")

                console.rule()

            ph("cli-agentic-optimize-run-finished", {"optimizations_found": optimizations_found})

            if optimizations_found == 0:
                logger.info("No optimizations found.")
            else:
                logger.info(f"Found {optimizations_found} optimization(s).")

        finally:
            cleanup_paths([self.test_cfg.concolic_test_root_dir])

    def _get_optimizable_functions(self) -> tuple[dict[Path, list[FunctionToOptimize]], int, Path | None]:
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

    def _discover_tests(
        self, file_to_funcs: dict[Path, list[FunctionToOptimize]]
    ) -> dict[str, set]:
        from codeflash.discovery.discover_unit_tests import discover_unit_tests

        console.rule()
        start_time = time.time()
        logger.info("Discovering existing function tests...")

        function_to_tests, num_tests, num_replay_tests = discover_unit_tests(
            self.test_cfg, file_to_funcs_to_optimize=file_to_funcs
        )

        console.rule()
        logger.info(
            f"Discovered {num_tests} existing tests and {num_replay_tests} replay tests "
            f"in {(time.time() - start_time):.1f}s"
        )
        console.rule()

        return function_to_tests

    def _rank_functions(
        self,
        file_to_funcs: dict[Path, list[FunctionToOptimize]],
        trace_file_path: Path | None,
    ) -> list[tuple[Path, FunctionToOptimize]]:
        all_functions: list[tuple[Path, FunctionToOptimize]] = []
        for file_path, functions in file_to_funcs.items():
            all_functions.extend((file_path, func) for func in functions)

        if not trace_file_path or not trace_file_path.exists():
            return all_functions

        try:
            from codeflash.benchmarking.function_ranker import FunctionRanker

            logger.info("Ranking functions by performance impact...")
            ranker = FunctionRanker(trace_file_path)
            functions_only = [func for _, func in all_functions]
            ranked_functions = ranker.rank_functions(functions_only)

            func_to_file_map = {}
            for file_path, func in all_functions:
                key = (func.file_path, func.qualified_name, func.starting_line)
                func_to_file_map[key] = file_path

            globally_ranked = []
            for func in ranked_functions:
                key = (func.file_path, func.qualified_name, func.starting_line)
                file_path = func_to_file_map.get(key)
                if file_path:
                    globally_ranked.append((file_path, func))

            logger.info(f"Ranked {len(ranked_functions)} functions by addressable time")
            return globally_ranked

        except Exception as e:
            logger.warning(f"Could not perform ranking: {e}")
            return all_functions

    def _find_js_project_root(self, file_path: Path) -> Path | None:
        current = file_path.parent if file_path.is_file() else file_path
        while current != current.parent:
            if (
                (current / "package.json").exists()
                or (current / "jest.config.js").exists()
                or (current / "jest.config.ts").exists()
                or (current / "tsconfig.json").exists()
            ):
                return current
            current = current.parent
        return None


def run_agentic_with_args(args: Namespace) -> None:
    optimizer = None
    try:
        optimizer = AgenticOptimizer(args)
        optimizer.run()
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Cleaning up and exiting...")
        raise SystemExit from None
