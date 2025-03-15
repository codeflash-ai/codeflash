from __future__ import annotations

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from black import FileMode, format_str

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.static_analysis import has_typed_parameters
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionCalledInTest


def remove_functions_with_only_any_type(code_string: str) -> str:
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
                    if isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "settings"
                ),
                None,
            )

            if settings_decorator:
                if not any(k.arg == "derandomize" for k in settings_decorator.keywords):
                    settings_decorator.keywords.append(
                        ast.keyword(arg="derandomize", value=ast.Constant(value=True))
                    )
            else:
                node.decorator_list.append(
                    ast.Call(
                        func=ast.Name(id="settings", ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(
                                arg="derandomize", value=ast.Constant(value=True)
                            )
                        ],
                    )
                )

    return ast.unparse(tree)


def generate_hypothesis_tests(
    test_cfg: TestConfig,
    args: Namespace,
    function_to_optimize: FunctionToOptimize,
    function_to_optimize_ast: ast.AST,
) -> tuple[dict[str, list[FunctionCalledInTest]], str, Path | None]:
    function_to_hypothesis_tests: dict[str, list[FunctionCalledInTest]] = {}
    hypothesis_test_suite_code: str = ""
    hypothesis_test_suite_dir: Path | None = None

    if (
        test_cfg.project_root_path
        and isinstance(
            function_to_optimize_ast, (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and has_typed_parameters(function_to_optimize_ast, function_to_optimize.parents)
    ):
        logger.info("Generating Hypothesis tests for the original code…")
        console.rule()

        try:
            logger.info("Running Hypothesis write with the following command:")
            hypothesis_result = subprocess.run(
                [
                    "hypothesis",
                    "write",
                    ".".join(
                        [
                            function_to_optimize.file_path.relative_to(
                                args.project_root
                            )
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
            logger.debug("Hypothesis test generation timed out")
            return (
                function_to_hypothesis_tests,
                hypothesis_test_suite_code,
                hypothesis_test_suite_dir,
            )

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
            function_to_hypothesis_tests = discover_unit_tests(hypothesis_config)
            with hypothesis_path.open("r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            class TestFunctionRemover(ast.NodeTransformer):
                def visit_FunctionDef(self, node):  # noqa: ANN001, ANN202, N802
                    if function_to_optimize.function_name in node.name:
                        node.name = f"{node.name}_hypothesis"
                        return node
                    return None

            modified_tree = TestFunctionRemover().visit(tree)
            ast.fix_missing_locations(modified_tree)
            unparsed = ast.unparse(modified_tree)

            hypothesis_test_suite_code = format_str(
                make_hypothesis_tests_deterministic(
                    remove_functions_with_only_any_type(unparsed)
                ),
                mode=FileMode(),
            )
            with hypothesis_path.open("w", encoding="utf-8") as f:
                f.write(hypothesis_test_suite_code)
            function_to_hypothesis_tests = discover_unit_tests(hypothesis_config)
            num_discovered_hypothesis_tests: int = sum(
                [len(value) for value in function_to_hypothesis_tests.values()]
            )
            logger.info(
                f"Created {num_discovered_hypothesis_tests} "
                f"hypothesis unit test case{'s' if num_discovered_hypothesis_tests != 1 else ''} "
            )
            console.rule()
            return (
                function_to_hypothesis_tests,
                hypothesis_test_suite_code,
                hypothesis_test_suite_dir,
            )

        logger.debug(
            f"Error running hypothesis write {': ' + hypothesis_result.stderr if hypothesis_result.stderr else '.'}"
        )
        console.rule()

    return (
        function_to_hypothesis_tests,
        hypothesis_test_suite_code,
        hypothesis_test_suite_dir,
    )
