from __future__ import annotations

import ast
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import get_qualified_function_path
from codeflash.code_utils.formatter import format_code
from codeflash.code_utils.static_analysis import has_typed_parameters
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionCalledInTest


def remove_functions_with_only_any_type(code_string: str) -> str:
    """Remove functions that have only Any type annotations.

    This filters out functions where all parameters are annotated with typing.Any,
    as these don't provide useful type information for property-based testing.
    """
    tree = ast.parse(code_string)
    new_body = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new_body.append(node)
        elif isinstance(node, ast.FunctionDef):
            all_any = True
            has_args = False

            for arg in node.args.args:
                has_args = True
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        if arg.annotation.id != "Any":
                            all_any = False
                    elif isinstance(arg.annotation, ast.Attribute):
                        if arg.annotation.attr != "Any":
                            all_any = False
                    elif isinstance(arg.annotation, ast.Subscript):
                        all_any = False
                    else:
                        all_any = False
                else:
                    all_any = False

            if (has_args and not all_any) or not has_args:
                new_body.append(node)

        else:
            new_body.append(node)

    new_tree = ast.Module(body=new_body, type_ignores=[])
    return ast.unparse(new_tree)


def make_hypothesis_tests_deterministic(code: str) -> str:
    """Add @settings(derandomize=True) decorator to make Hypothesis tests deterministic."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    settings_imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "hypothesis"
        and any(alias.name == "settings" for alias in node.names)
        for node in tree.body
    )

    if not settings_imported:
        tree.body.insert(0, ast.parse("from hypothesis import settings").body[0])

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            settings_decorator = next(
                (
                    d
                    for d in node.decorator_list
                    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "settings"
                ),
                None,
            )

            if settings_decorator:
                if not any(k.arg == "derandomize" for k in settings_decorator.keywords):
                    settings_decorator.keywords.append(ast.keyword(arg="derandomize", value=ast.Constant(value=True)))
            else:
                node.decorator_list.append(
                    ast.Call(
                        func=ast.Name(id="settings", ctx=ast.Load()),
                        args=[],
                        keywords=[ast.keyword(arg="derandomize", value=ast.Constant(value=True))],
                    )
                )

    return ast.unparse(tree)


def generate_hypothesis_tests(
    test_cfg: TestConfig, args: Namespace, function_to_optimize: FunctionToOptimize, function_to_optimize_ast: ast.AST
) -> tuple[dict[str, list[FunctionCalledInTest]], str]:
    """Generate property-based tests using Hypothesis ghostwriter.

    This function:
    1. Uses Hypothesis CLI to generate property-based tests for the target function
    2. Filters generated tests to only include the target function
    3. Removes functions with only Any type annotations
    4. Makes tests deterministic by adding @settings(derandomize=True)
    5. Formats the tests with the project formatter

    Returns:
        Tuple of (function_to_tests_map, test_suite_code)

    """
    start_time = time.perf_counter()
    function_to_hypothesis_tests: dict[str, list[FunctionCalledInTest]] = {}
    hypothesis_test_suite_code: str = ""

    if (
        test_cfg.project_root_path
        and isinstance(function_to_optimize_ast, (ast.FunctionDef, ast.AsyncFunctionDef))
        and has_typed_parameters(function_to_optimize_ast, function_to_optimize.parents)
    ):
        logger.info("Generating Hypothesis tests for the original code…")
        console.rule()

        try:
            qualified_function_path = get_qualified_function_path(
                function_to_optimize.file_path, args.project_root, function_to_optimize.qualified_name
            )
            logger.info(f"command: hypothesis write {qualified_function_path}")

            hypothesis_result = subprocess.run(
                ["hypothesis", "write", qualified_function_path],
                capture_output=True,
                text=True,
                cwd=args.project_root,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.debug("Hypothesis test generation timed out")
            end_time = time.perf_counter()
            logger.debug(f"Hypothesis test generation completed in {end_time - start_time:.2f} seconds")
            return function_to_hypothesis_tests, hypothesis_test_suite_code

        if hypothesis_result.returncode == 0:
            hypothesis_test_suite_code = hypothesis_result.stdout
            hypothesis_test_suite_dir = Path(tempfile.mkdtemp(dir=test_cfg.tests_root))
            hypothesis_path = hypothesis_test_suite_dir / "test_hypothesis.py"
            hypothesis_path.write_text(hypothesis_test_suite_code, encoding="utf8")

            hypothesis_config = TestConfig(
                tests_root=hypothesis_test_suite_dir,
                tests_project_rootdir=test_cfg.tests_project_rootdir,
                project_root_path=args.project_root,
                test_framework=args.test_framework,
                pytest_cmd=args.pytest_cmd,
            )
            function_to_hypothesis_tests, num_discovered_hypothesis_tests, _ = discover_unit_tests(hypothesis_config)
            with hypothesis_path.open("r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            class TestFunctionRemover(ast.NodeTransformer):
                def visit_FunctionDef(self, node):  # noqa: ANN001, ANN202
                    if node.name.startswith("test_") and function_to_optimize.function_name in node.name:
                        return node
                    return None

            modified_tree = TestFunctionRemover().visit(tree)
            ast.fix_missing_locations(modified_tree)
            unparsed = ast.unparse(modified_tree)

            console.print(f"modified src: {unparsed}")

            hypothesis_test_suite_code = format_code(
                args.formatter_cmds,
                hypothesis_path,
                optimized_code=make_hypothesis_tests_deterministic(remove_functions_with_only_any_type(unparsed)),
            )
            with hypothesis_path.open("w", encoding="utf-8") as f:
                f.write(hypothesis_test_suite_code)
            function_to_hypothesis_tests, num_discovered_hypothesis_tests, _ = discover_unit_tests(hypothesis_config)
            logger.info(
                f"Created {num_discovered_hypothesis_tests} "
                f"hypothesis unit test case{'s' if num_discovered_hypothesis_tests != 1 else ''} "
            )
            console.rule()
            end_time = time.perf_counter()
            logger.debug(f"Generated hypothesis tests in {end_time - start_time:.2f} seconds")
            return function_to_hypothesis_tests, hypothesis_test_suite_code

        logger.debug(
            f"Error running hypothesis write {': ' + hypothesis_result.stderr if hypothesis_result.stderr else '.'}"
        )
        console.rule()

    end_time = time.perf_counter()
    logger.debug(f"Hypothesis test generation completed in {end_time - start_time:.2f} seconds")
    return function_to_hypothesis_tests, hypothesis_test_suite_code
