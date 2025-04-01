from __future__ import annotations

import ast
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.api.aiservice import AiServiceClient, LocalAiServiceClient
from codeflash.cli_cmds.console import console, logger, progress_bar
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_replacer import normalize_code, normalize_node
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.static_analysis import analyze_imported_modules, get_first_top_level_function_or_method_ast
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.discovery.functions_to_optimize import get_functions_to_optimize
from codeflash.either import is_successful
from codeflash.models.models import TestType, ValidCode
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
        )

        self.aiservice_client = AiServiceClient()
        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None

    def create_function_optimizer(
        self,
        function_to_optimize: FunctionToOptimize,
        function_to_optimize_ast: ast.FunctionDef | None = None,
        function_to_tests: dict[str, list[FunctionCalledInTest]] | None = None,
        function_to_optimize_source_code: str | None = "",
    ) -> FunctionOptimizer:
        return FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=self.test_cfg,
            function_to_optimize_source_code=function_to_optimize_source_code,
            function_to_tests=function_to_tests,
            function_to_optimize_ast=function_to_optimize_ast,
            aiservice_client=self.aiservice_client,
            args=self.args,
        )

    def run(self) -> None:
        ph("cli-optimize-run-start")
        logger.info("Running optimizer.")
        console.rule()
        if not env_utils.ensure_codeflash_api_key():
            return
        function_optimizer = None
        file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]]
        num_optimizable_functions: int
        (file_to_funcs_to_optimize, num_optimizable_functions) = get_functions_to_optimize(
            optimize_all=self.args.all,
            replay_test=self.args.replay_test,
            file=self.args.file,
            only_get_this_function=self.args.function,
            test_cfg=self.test_cfg,
            ignore_paths=self.args.ignore_paths,
            project_root=self.args.project_root,
            module_root=self.args.module_root,
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

            console.rule()
            function_to_tests: dict[str, list[FunctionCalledInTest]] = discover_unit_tests(self.test_cfg)
            num_discovered_tests: int = sum([len(value) for value in function_to_tests.values()])
            console.rule()
            logger.info(f"Discovered {num_discovered_tests} existing unit tests in {self.test_cfg.tests_root}")
            console.rule()
            ph("cli-optimize-discovered-tests", {"num_tests": num_discovered_tests})

            for original_module_path in file_to_funcs_to_optimize:
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

                for function_to_optimize in file_to_funcs_to_optimize[original_module_path]:
                    function_iterator_count += 1
                    logger.info(
                        f"Optimizing function {function_iterator_count} of {num_optimizable_functions}: "
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
                    function_optimizer = self.create_function_optimizer(
                        function_to_optimize,
                        function_to_optimize_ast,
                        function_to_tests,
                        validated_original_code[original_module_path].source_code,
                    )
                    best_optimization = function_optimizer.optimize_function()
                    if is_successful(best_optimization):
                        optimizations_found += 1
                    else:
                        logger.warning(best_optimization.failure())
                        console.rule()
                        continue
            ph("cli-optimize-run-finished", {"optimizations_found": optimizations_found})
            if optimizations_found == 0:
                logger.info("❌ No optimizations found.")
            elif self.args.all:
                logger.info("✨ All functions have been optimized! ✨")
        finally:
            if function_optimizer:
                for test_file in function_optimizer.test_files.get_by_type(TestType.GENERATED_REGRESSION).test_files:
                    test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
                    test_file.benchmarking_file_path.unlink(missing_ok=True)
                for test_file in function_optimizer.test_files.get_by_type(TestType.EXISTING_UNIT_TEST).test_files:
                    test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
                    test_file.benchmarking_file_path.unlink(missing_ok=True)
                for test_file in function_optimizer.test_files.get_by_type(TestType.CONCOLIC_COVERAGE_TEST).test_files:
                    test_file.instrumented_behavior_file_path.unlink(missing_ok=True)
                if function_optimizer.test_cfg.concolic_test_root_dir:
                    shutil.rmtree(function_optimizer.test_cfg.concolic_test_root_dir, ignore_errors=True)
            if hasattr(get_run_tmp_file, "tmpdir"):
                get_run_tmp_file.tmpdir.cleanup()


def run_with_args(args: Namespace) -> None:
    optimizer = Optimizer(args)
    optimizer.run()
