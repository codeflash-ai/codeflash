from __future__ import annotations

import ast
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash.code_utils.concolic_utils import clean_concolic_tests
from codeflash.code_utils.static_analysis import has_typed_parameters
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.telemetry.posthog_cf import ph
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionCalledInTest


def generate_concolic_tests(
    test_cfg: TestConfig, args: Namespace, function_to_optimize: FunctionToOptimize, function_to_optimize_ast: ast.AST
) -> tuple[dict[str, set[FunctionCalledInTest]], str]:
    start_time = time.perf_counter()
    function_to_concolic_tests = {}
    concolic_test_suite_code = ""
    if (
        test_cfg.concolic_test_root_dir
        and isinstance(function_to_optimize_ast, (ast.FunctionDef, ast.AsyncFunctionDef))
        and has_typed_parameters(function_to_optimize_ast, function_to_optimize.parents)
    ):
        logger.info("Generating concolic opcode coverage tests for the original code…")
        console.rule()
        try:
            cover_result = subprocess.run(
                [
                    SAFE_SYS_EXECUTABLE,
                    "-m",
                    "crosshair",
                    "cover",
                    "--example_output_format=pytest",
                    "--per_condition_timeout=64",
                    ".".join(
                        [
                            function_to_optimize.file_path.relative_to(args.project_root)
                            .with_suffix("")
                            .as_posix()
                            .replace("/", "."),
                            function_to_optimize.qualified_name,
                        ]
                    ),
                ],
                capture_output=True,
                text=True,
                cwd=args.project_root,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            logger.debug("CrossHair Cover test generation timed out")
            return function_to_concolic_tests, concolic_test_suite_code

        if cover_result.returncode == 0:
            generated_concolic_test: str = cover_result.stdout
            concolic_test_suite_code: str = clean_concolic_tests(generated_concolic_test)
            concolic_test_suite_dir = Path(tempfile.mkdtemp(dir=test_cfg.concolic_test_root_dir))
            concolic_test_suite_path = concolic_test_suite_dir / "test_concolic_coverage.py"
            concolic_test_suite_path.write_text(concolic_test_suite_code, encoding="utf8")

            concolic_test_cfg = TestConfig(
                tests_root=concolic_test_suite_dir,
                tests_project_rootdir=test_cfg.concolic_test_root_dir,
                project_root_path=args.project_root,
                test_framework=args.test_framework,
                pytest_cmd=args.pytest_cmd,
            )
            function_to_concolic_tests, num_discovered_concolic_tests = discover_unit_tests(concolic_test_cfg)
            logger.info(
                f"Created {num_discovered_concolic_tests} "
                f"concolic unit test case{'s' if num_discovered_concolic_tests != 1 else ''} "
            )
            console.rule()
            ph("cli-optimize-concolic-tests", {"num_tests": num_discovered_concolic_tests})

        else:
            logger.debug(f"Error running CrossHair Cover {': ' + cover_result.stderr if cover_result.stderr else '.'}")
            console.rule()
    end_time = time.perf_counter()
    logger.debug(f"Generated concolic tests in {end_time - start_time:.2f} seconds")
    return function_to_concolic_tests, concolic_test_suite_code
