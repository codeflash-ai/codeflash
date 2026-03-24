"""Concolic test generation using CrossHair."""

from __future__ import annotations

import ast
import importlib.util
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash_core.config import TestConfig
from codeflash_python.code_utils.compat import SAFE_SYS_EXECUTABLE
from codeflash_python.code_utils.shell_utils import make_env_with_project_root
from codeflash_python.discovery.discover_unit_tests import discover_unit_tests
from codeflash_python.static_analysis.concolic_utils import clean_concolic_tests, is_valid_concolic_test
from codeflash_python.static_analysis.static_analysis import has_typed_parameters
from codeflash_python.telemetry.posthog_cf import ph

if TYPE_CHECKING:
    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger(__name__)


def generate_concolic_tests(
    test_cfg: TestConfig, project_root: Path, function_to_optimize: FunctionToOptimize, function_to_optimize_ast: Any
) -> tuple[dict, str]:
    crosshair_available = importlib.util.find_spec("crosshair") is not None

    start_time = time.perf_counter()
    function_to_concolic_tests: dict = {}
    concolic_test_suite_code = ""

    if not crosshair_available:
        logger.debug("Skipping concolic test generation (crosshair-tool is not installed)")
        return function_to_concolic_tests, concolic_test_suite_code

    if (
        test_cfg.concolic_test_root_dir
        and isinstance(function_to_optimize_ast, ast.FunctionDef)
        and has_typed_parameters(function_to_optimize_ast, function_to_optimize.parents)
    ):
        logger.info("Generating concolic opcode coverage tests for the original code...")
        try:
            env = make_env_with_project_root(project_root)
            cover_result = subprocess.run(
                [
                    SAFE_SYS_EXECUTABLE,
                    "-m",
                    "crosshair",
                    "cover",
                    "--example_output_format=pytest",
                    "--per_condition_timeout=20",
                    ".".join(
                        [
                            function_to_optimize.file_path.relative_to(project_root)
                            .with_suffix("")
                            .as_posix()
                            .replace("/", "."),
                            function_to_optimize.qualified_name,
                        ]
                    ),
                ],
                capture_output=True,
                text=True,
                cwd=project_root,
                check=False,
                timeout=600,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.debug("CrossHair Cover test generation timed out")
            return function_to_concolic_tests, concolic_test_suite_code

        if cover_result.returncode == 0:
            generated_concolic_test: str = cover_result.stdout
            if not is_valid_concolic_test(generated_concolic_test, project_root=str(project_root)):
                logger.debug("CrossHair generated invalid test, skipping")
                return function_to_concolic_tests, concolic_test_suite_code
            concolic_test_suite_code = clean_concolic_tests(generated_concolic_test)
            concolic_test_suite_dir = Path(tempfile.mkdtemp(dir=test_cfg.concolic_test_root_dir))
            concolic_test_suite_path = concolic_test_suite_dir / "test_concolic_coverage.py"
            concolic_test_suite_path.write_text(concolic_test_suite_code, encoding="utf8")

            concolic_test_cfg = TestConfig(
                tests_root=concolic_test_suite_dir,
                tests_project_rootdir=test_cfg.concolic_test_root_dir,
                project_root=project_root,
            )
            function_to_concolic_tests, num_discovered_concolic_tests, _ = discover_unit_tests(concolic_test_cfg)
            logger.info(
                "Created %d concolic unit test case%s ",
                num_discovered_concolic_tests,
                "s" if num_discovered_concolic_tests != 1 else "",
            )
            ph("cli-optimize-concolic-tests", {"num_tests": num_discovered_concolic_tests})

        else:
            logger.debug("Error running CrossHair Cover%s", ": " + cover_result.stderr if cover_result.stderr else ".")
    end_time = time.perf_counter()
    logger.debug("Generated concolic tests in %.2f seconds", end_time - start_time)
    return function_to_concolic_tests, concolic_test_suite_code
