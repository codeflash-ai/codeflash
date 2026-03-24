from __future__ import annotations

import ast
import logging
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, TypeVar

import libcst as cst
from libcst.metadata import PositionProvider

from codeflash_core.models import FunctionParent
from codeflash_python.code_utils.config_parser import find_conftest_files
from codeflash_python.code_utils.formatter import sort_imports
from codeflash_python.static_analysis.code_replacer_base import get_optimized_code_for_module
from codeflash_python.static_analysis.global_code_transforms import (
    add_global_assignments,
    find_insertion_index_after_imports,
)
from codeflash_python.static_analysis.import_analysis import add_needed_imports_from_module
from codeflash_python.static_analysis.line_profile_utils import ImportAdder

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_python.models.models import CodeStringsMarkdown

logger = logging.getLogger("codeflash_python")

ASTNodeT = TypeVar("ASTNodeT", bound=ast.AST)


def normalize_node(node: ASTNodeT) -> ASTNodeT:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
        node.body = node.body[1:]
    if hasattr(node, "body"):
        node.body = [normalize_node(n) for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]  # type: ignore[assignment,attr-defined]
    return node


@lru_cache(maxsize=3)
def normalize_code(code: str) -> str:
    return ast.unparse(normalize_node(ast.parse(code)))


def has_autouse_fixture(node: cst.FunctionDef) -> bool:
    for decorator in node.decorators:
        dec = decorator.decorator
        if not isinstance(dec, cst.Call):
            continue
        is_fixture = (
            isinstance(dec.func, cst.Attribute)
            and isinstance(dec.func.value, cst.Name)
            and dec.func.attr.value == "fixture"
            and dec.func.value.value == "pytest"
        ) or (isinstance(dec.func, cst.Name) and dec.func.value == "fixture")
        if is_fixture:
            for arg in dec.args:
                if (
                    arg.keyword
                    and arg.keyword.value == "autouse"
                    and isinstance(arg.value, cst.Name)
                    and arg.value.value == "True"
                ):
                    return True
    return False


class AddRequestArgument(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if not has_autouse_fixture(original_node):
            return updated_node

        args = updated_node.params.params
        arg_names = {arg.name.value for arg in args}

        if "request" in arg_names:
            return updated_node

        request_param = cst.Param(name=cst.Name("request"))

        if args:
            first_arg = args[0].name.value
            if first_arg in {"self", "cls"}:
                new_params = [args[0], request_param, *args[1:]]
            else:
                new_params = [request_param, *args]
        else:
            new_params = [request_param]

        new_param_list = updated_node.params.with_changes(params=new_params)
        return updated_node.with_changes(params=new_param_list)


class PytestMarkAdder(cst.CSTTransformer):
    """Transformer that adds pytest marks to test functions."""

    def __init__(self, mark_name: str) -> None:
        super().__init__()
        self.mark_name = mark_name
        self.has_pytest_import = False

    def visit_Module(self, node: cst.Module) -> None:
        """Check if pytest is already imported."""
        for statement in node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Import):
                        for import_alias in stmt.names:
                            if isinstance(import_alias, cst.ImportAlias) and import_alias.name.value == "pytest":
                                self.has_pytest_import = True

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add pytest import if not present."""
        if not self.has_pytest_import:
            # Create import statement
            import_stmt = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("pytest"))])])
            # Add import at the beginning
            updated_node = updated_node.with_changes(body=[import_stmt, *updated_node.body])
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Add pytest mark to test functions."""
        # Check if the mark already exists
        for decorator in updated_node.decorators:
            if self.is_pytest_mark(decorator.decorator, self.mark_name):
                return updated_node

        # Create the pytest mark decorator
        mark_decorator = self.create_pytest_mark()

        # Add the decorator
        new_decorators = [*list(updated_node.decorators), mark_decorator]
        return updated_node.with_changes(decorators=new_decorators)

    def is_pytest_mark(self, decorator: cst.BaseExpression, mark_name: str) -> bool:
        """Check if a decorator is a specific pytest mark."""
        if isinstance(decorator, cst.Attribute):
            if (
                isinstance(decorator.value, cst.Attribute)
                and isinstance(decorator.value.value, cst.Name)
                and decorator.value.value.value == "pytest"
                and decorator.value.attr.value == "mark"
                and decorator.attr.value == mark_name
            ):
                return True
        elif isinstance(decorator, cst.Call) and isinstance(decorator.func, cst.Attribute):
            return self.is_pytest_mark(decorator.func, mark_name)
        return False

    def create_pytest_mark(self) -> cst.Decorator:
        """Create a pytest mark decorator."""
        # Base: pytest.mark.{mark_name}
        mark_attr = cst.Attribute(
            value=cst.Attribute(value=cst.Name("pytest"), attr=cst.Name("mark")), attr=cst.Name(self.mark_name)
        )
        decorator = mark_attr
        return cst.Decorator(decorator=decorator)


class AutouseFixtureModifier(cst.CSTTransformer):
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if not has_autouse_fixture(original_node):
            return updated_node

        else_block = cst.Else(body=updated_node.body)
        if_test = cst.parse_expression('request.node.get_closest_marker("codeflash_no_autouse")')
        yield_statement = cst.parse_statement("yield")
        if_body = cst.IndentedBlock(body=[yield_statement])
        new_if_statement = cst.If(test=if_test, body=if_body, orelse=else_block)
        return updated_node.with_changes(body=cst.IndentedBlock(body=[new_if_statement]))


def disable_autouse(test_path: Path) -> str:
    file_content = test_path.read_text(encoding="utf-8")
    module = cst.parse_module(file_content)
    add_request_argument = AddRequestArgument()
    disable_autouse_fixture = AutouseFixtureModifier()
    modified_module = module.visit(add_request_argument)
    modified_module = modified_module.visit(disable_autouse_fixture)
    test_path.write_text(modified_module.code, encoding="utf-8")
    return file_content


def modify_autouse_fixture(test_paths: list[Path]) -> dict[Path, str]:
    # find fixutre definition in conftetst.py (the one closest to the test)
    # get fixtures present in override-fixtures in pyproject.toml
    # add if marker closest return
    file_content_map = {}
    conftest_files = find_conftest_files(test_paths)
    for cf_file in conftest_files:
        # iterate over all functions in the file
        # if function has autouse fixture, modify function to bypass with custom marker
        original_content = disable_autouse(cf_file)
        file_content_map[cf_file] = original_content
    return file_content_map


# # reuse line profiler utils to add decorator and import to test fns
def add_custom_marker_to_all_tests(test_paths: list[Path]) -> None:
    for test_path in test_paths:
        # read file
        file_content = test_path.read_text(encoding="utf-8")
        module = cst.parse_module(file_content)
        importadder = ImportAdder("import pytest")
        modified_module = module.visit(importadder)
        modified_module = cst.parse_module(sort_imports(code=modified_module.code, float_to_top=True))
        pytest_mark_adder = PytestMarkAdder("codeflash_no_autouse")
        modified_module = modified_module.visit(pytest_mark_adder)
        test_path.write_text(modified_module.code, encoding="utf-8")


def replace_functions_in_file(
    source_code: str,
    original_function_names: list[str],
    optimized_code: str,
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]],
) -> str:
    parsed_function_names = []
    for original_function_name in original_function_names:
        if original_function_name.count(".") == 0:
            class_name, function_name = None, original_function_name
        elif original_function_name.count(".") == 1:
            class_name, function_name = original_function_name.split(".")
        else:
            msg = f"Unable to find {original_function_name}. Returning unchanged source code."
            logger.error(msg)
            return source_code
        parsed_function_names.append((class_name, function_name))

    # Collect functions from optimized code without using MetadataWrapper
    optimized_module = cst.parse_module(optimized_code)
    modified_functions: dict[tuple[str | None, str], cst.FunctionDef] = {}
    new_functions: list[cst.FunctionDef] = []
    new_class_functions: dict[str, list[cst.FunctionDef]] = defaultdict(list)
    new_classes: list[cst.ClassDef] = []
    modified_init_functions: dict[str, cst.FunctionDef] = {}

    function_names_set = set(parsed_function_names)

    for node in optimized_module.body:
        if isinstance(node, cst.FunctionDef):
            key = (None, node.name.value)
            if key in function_names_set:
                modified_functions[key] = node
            elif preexisting_objects and (node.name.value, ()) not in preexisting_objects:
                new_functions.append(node)

        elif isinstance(node, cst.ClassDef):
            class_name = node.name.value
            parents = (FunctionParent(name=class_name, type="ClassDef"),)

            if (class_name, ()) not in preexisting_objects:
                new_classes.append(node)

            for child in node.body.body:
                if isinstance(child, cst.FunctionDef):
                    method_key = (class_name, child.name.value)
                    if method_key in function_names_set:
                        modified_functions[method_key] = child
                    elif (
                        child.name.value == "__init__"
                        and preexisting_objects
                        and (class_name, ()) in preexisting_objects
                    ):
                        modified_init_functions[class_name] = child
                    elif preexisting_objects and (child.name.value, parents) not in preexisting_objects:
                        new_class_functions[class_name].append(child)

    original_module = cst.parse_module(source_code)

    max_function_index = None
    max_class_index = None
    for index, _node in enumerate(original_module.body):
        if isinstance(_node, cst.FunctionDef):
            max_function_index = index
        if isinstance(_node, cst.ClassDef):
            max_class_index = index

    new_body: list[cst.CSTNode] = []
    existing_class_names = set()

    for node in original_module.body:
        if isinstance(node, cst.FunctionDef):
            key = (None, node.name.value)
            if key in modified_functions:
                modified_func = modified_functions[key]
                new_body.append(node.with_changes(body=modified_func.body, decorators=modified_func.decorators))
            else:
                new_body.append(node)

        elif isinstance(node, cst.ClassDef):
            class_name = node.name.value
            existing_class_names.add(class_name)

            new_members: list[cst.CSTNode] = []
            for child in node.body.body:
                if isinstance(child, cst.FunctionDef):
                    key = (class_name, child.name.value)
                    if key in modified_functions:
                        modified_func = modified_functions[key]
                        new_members.append(
                            child.with_changes(body=modified_func.body, decorators=modified_func.decorators)
                        )
                    elif child.name.value == "__init__" and class_name in modified_init_functions:
                        new_members.append(modified_init_functions[class_name])
                    else:
                        new_members.append(child)
                else:
                    new_members.append(child)

            if class_name in new_class_functions:
                new_members.extend(new_class_functions[class_name])

            new_body.append(node.with_changes(body=node.body.with_changes(body=new_members)))
        else:
            new_body.append(node)

    if new_classes:
        unique_classes = [nc for nc in new_classes if nc.name.value not in existing_class_names]
        if unique_classes:
            new_classes_insertion_idx = (
                max_class_index if max_class_index is not None else find_insertion_index_after_imports(original_module)
            )
            new_body = list(
                chain(new_body[:new_classes_insertion_idx], unique_classes, new_body[new_classes_insertion_idx:])
            )

    if new_functions:
        if max_function_index is not None:
            new_body = [*new_body[: max_function_index + 1], *new_functions, *new_body[max_function_index + 1 :]]
        elif max_class_index is not None:
            new_body = [*new_body[: max_class_index + 1], *new_functions, *new_body[max_class_index + 1 :]]
        else:
            new_body = [*new_functions, *new_body]

    updated_module = original_module.with_changes(body=new_body)
    return updated_module.code


def replace_functions_and_add_imports(
    source_code: str,
    function_names: list[str],
    optimized_code: str,
    module_abspath: Path,
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]],
    project_root_path: Path,
) -> str:
    return add_needed_imports_from_module(
        optimized_code,
        replace_functions_in_file(source_code, function_names, optimized_code, preexisting_objects),
        module_abspath,
        module_abspath,
        project_root_path,
    )


def replace_function_definitions_in_module(
    function_names: list[str],
    optimized_code: CodeStringsMarkdown,
    module_abspath: Path,
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]],
    project_root_path: Path,
    should_add_global_assignments: bool = True,
) -> bool:
    source_code: str = module_abspath.read_text(encoding="utf8")
    code_to_apply = get_optimized_code_for_module(module_abspath.relative_to(project_root_path), optimized_code)

    new_code: str = replace_functions_and_add_imports(
        # adding the global assignments before replacing the code, not after
        # because of an "edge case" where the optimized code intoduced a new import and a global assignment using that import
        # and that import wasn't used before, so it was ignored when calling AddImportsVisitor.add_needed_import inside replace_functions_and_add_imports (because the global assignment wasn't added yet)
        # this was added at https://github.com/codeflash-ai/codeflash/pull/448
        add_global_assignments(code_to_apply, source_code) if should_add_global_assignments else source_code,
        function_names,
        code_to_apply,
        module_abspath,
        preexisting_objects,
        project_root_path,
    )
    if is_zero_diff(source_code, new_code):
        return False
    module_abspath.write_text(new_code, encoding="utf8")
    return True


def is_zero_diff(original_code: str, new_code: str) -> bool:
    return normalize_code(original_code) == normalize_code(new_code)
