from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Optional, TypeVar

import libcst as cst
from libcst.metadata import PositionProvider

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_parser import find_conftest_files
from codeflash.code_utils.formatter import sort_imports
from codeflash.languages import is_python
from codeflash.languages.python.static_analysis.code_extractor import (
    add_global_assignments,
    add_needed_imports_from_module,
    find_insertion_index_after_imports,
)
from codeflash.languages.python.static_analysis.line_profile_utils import ImportAdder
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import Language, LanguageSupport
    from codeflash.languages.javascript.treesitter import TreeSitterAnalyzer
    from codeflash.models.models import CodeOptimizationContext, CodeStringsMarkdown, OptimizedCandidate, ValidCode

ASTNodeT = TypeVar("ASTNodeT", bound=ast.AST)


def normalize_node(node: ASTNodeT) -> ASTNodeT:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
        node.body = node.body[1:]
    if hasattr(node, "body"):
        node.body = [normalize_node(n) for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return node


@lru_cache(maxsize=3)
def normalize_code(code: str) -> str:
    return ast.unparse(normalize_node(ast.parse(code)))


class AddRequestArgument(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Matcher for '@fixture' or '@pytest.fixture'
        for decorator in original_node.decorators:
            dec = decorator.decorator

            if isinstance(dec, cst.Call):
                func_name = ""
                if isinstance(dec.func, cst.Attribute) and isinstance(dec.func.value, cst.Name):
                    if dec.func.attr.value == "fixture" and dec.func.value.value == "pytest":
                        func_name = "pytest.fixture"
                elif isinstance(dec.func, cst.Name) and dec.func.value == "fixture":
                    func_name = "fixture"

                if func_name:
                    for arg in dec.args:
                        if (
                            arg.keyword
                            and arg.keyword.value == "autouse"
                            and isinstance(arg.value, cst.Name)
                            and arg.value.value == "True"
                        ):
                            args = updated_node.params.params
                            arg_names = {arg.name.value for arg in args}

                            # Skip if 'request' is already present
                            if "request" in arg_names:
                                return updated_node

                            # Create a new 'request' param
                            request_param = cst.Param(name=cst.Name("request"))

                            # Add 'request' as the first argument (after 'self' or 'cls' if needed)
                            if args:
                                first_arg = args[0].name.value
                                if first_arg in {"self", "cls"}:
                                    new_params = [args[0], request_param] + list(args[1:])  # noqa: RUF005
                                else:
                                    new_params = [request_param] + list(args)  # noqa: RUF005
                            else:
                                new_params = [request_param]

                            new_param_list = updated_node.params.with_changes(params=new_params)
                            return updated_node.with_changes(params=new_param_list)
        return updated_node


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
            if self._is_pytest_mark(decorator.decorator, self.mark_name):
                return updated_node

        # Create the pytest mark decorator
        mark_decorator = self._create_pytest_mark()

        # Add the decorator
        new_decorators = [*list(updated_node.decorators), mark_decorator]
        return updated_node.with_changes(decorators=new_decorators)

    def _is_pytest_mark(self, decorator: cst.BaseExpression, mark_name: str) -> bool:
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
            return self._is_pytest_mark(decorator.func, mark_name)
        return False

    def _create_pytest_mark(self) -> cst.Decorator:
        """Create a pytest mark decorator."""
        # Base: pytest.mark.{mark_name}
        mark_attr = cst.Attribute(
            value=cst.Attribute(value=cst.Name("pytest"), attr=cst.Name("mark")), attr=cst.Name(self.mark_name)
        )
        decorator = mark_attr
        return cst.Decorator(decorator=decorator)


class AutouseFixtureModifier(cst.CSTTransformer):
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Matcher for '@fixture' or '@pytest.fixture'
        for decorator in original_node.decorators:
            dec = decorator.decorator

            if isinstance(dec, cst.Call):
                func_name = ""
                if isinstance(dec.func, cst.Attribute) and isinstance(dec.func.value, cst.Name):
                    if dec.func.attr.value == "fixture" and dec.func.value.value == "pytest":
                        func_name = "pytest.fixture"
                elif isinstance(dec.func, cst.Name) and dec.func.value == "fixture":
                    func_name = "fixture"

                if func_name:
                    for arg in dec.args:
                        if (
                            arg.keyword
                            and arg.keyword.value == "autouse"
                            and isinstance(arg.value, cst.Name)
                            and arg.value.value == "True"
                        ):
                            # Found a matching fixture with autouse=True

                            # 1. The original body of the function will become the 'else' block.
                            #    updated_node.body is an IndentedBlock, which is what cst.Else expects.
                            else_block = cst.Else(body=updated_node.body)

                            # 2. Create the new 'if' block that will exit the fixture early.
                            if_test = cst.parse_expression('request.node.get_closest_marker("codeflash_no_autouse")')
                            yield_statement = cst.parse_statement("yield")
                            if_body = cst.IndentedBlock(body=[yield_statement])

                            # 3. Construct the full if/else statement.
                            new_if_statement = cst.If(test=if_test, body=if_body, orelse=else_block)

                            # 4. Replace the entire function's body with our new single statement.
                            return updated_node.with_changes(body=cst.IndentedBlock(body=[new_if_statement]))
        return updated_node


def disable_autouse(test_path: Path) -> str:
    file_content = test_path.read_text(encoding="utf-8")
    module = cst.parse_module(file_content)
    add_request_argument = AddRequestArgument()
    disable_autouse_fixture = AutouseFixtureModifier()
    modified_module = module.visit(add_request_argument)
    modified_module = modified_module.visit(disable_autouse_fixture)
    test_path.write_text(modified_module.code, encoding="utf-8")
    return file_content


def modify_autouse_fixture(test_paths: list[Path]) -> dict[Path, list[str]]:
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


class OptimFunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(
        self,
        preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] | None = None,
        function_names: set[tuple[str | None, str]] | None = None,
    ) -> None:
        super().__init__()
        self.preexisting_objects = preexisting_objects if preexisting_objects is not None else set()

        self.function_names = function_names  # set of (class_name, function_name)
        self.modified_functions: dict[
            tuple[str | None, str], cst.FunctionDef
        ] = {}  # keys are (class_name, function_name)
        self.new_functions: list[cst.FunctionDef] = []
        self.new_class_functions: dict[str, list[cst.FunctionDef]] = defaultdict(list)
        self.new_classes: list[cst.ClassDef] = []
        self.current_class = None
        self.modified_init_functions: dict[str, cst.FunctionDef] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if (self.current_class, node.name.value) in self.function_names:
            self.modified_functions[(self.current_class, node.name.value)] = node
        elif self.current_class and node.name.value == "__init__":
            self.modified_init_functions[self.current_class] = node
        elif (
            self.preexisting_objects
            and (node.name.value, ()) not in self.preexisting_objects
            and self.current_class is None
        ):
            self.new_functions.append(node)
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self.current_class:
            return False  # If already in a class, do not recurse deeper
        self.current_class = node.name.value

        parents = (FunctionParent(name=node.name.value, type="ClassDef"),)

        if (node.name.value, ()) not in self.preexisting_objects:
            self.new_classes.append(node)

        for child_node in node.body.body:
            if (
                self.preexisting_objects
                and isinstance(child_node, cst.FunctionDef)
                and (child_node.name.value, parents) not in self.preexisting_objects
            ):
                self.new_class_functions[node.name.value].append(child_node)

        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        if self.current_class:
            self.current_class = None


class OptimFunctionReplacer(cst.CSTTransformer):
    def __init__(
        self,
        modified_functions: Optional[dict[tuple[str | None, str], cst.FunctionDef]] = None,
        new_classes: Optional[list[cst.ClassDef]] = None,
        new_functions: Optional[list[cst.FunctionDef]] = None,
        new_class_functions: Optional[dict[str, list[cst.FunctionDef]]] = None,
        modified_init_functions: Optional[dict[str, cst.FunctionDef]] = None,
    ) -> None:
        super().__init__()
        self.modified_functions = modified_functions if modified_functions is not None else {}
        self.new_functions = new_functions if new_functions is not None else []
        self.new_classes = new_classes if new_classes is not None else []
        self.new_class_functions = new_class_functions if new_class_functions is not None else defaultdict(list)
        self.modified_init_functions: dict[str, cst.FunctionDef] = (
            modified_init_functions if modified_init_functions is not None else {}
        )
        self.current_class = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if (self.current_class, original_node.name.value) in self.modified_functions:
            node = self.modified_functions[(self.current_class, original_node.name.value)]
            return updated_node.with_changes(body=node.body, decorators=node.decorators)
        if original_node.name.value == "__init__" and self.current_class in self.modified_init_functions:
            return self.modified_init_functions[self.current_class]

        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self.current_class:
            return False  # If already in a class, do not recurse deeper
        self.current_class = node.name.value
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        if self.current_class and self.current_class == original_node.name.value:
            self.current_class = None
            if original_node.name.value in self.new_class_functions:
                return updated_node.with_changes(
                    body=updated_node.body.with_changes(
                        body=(list(updated_node.body.body) + list(self.new_class_functions[original_node.name.value]))
                    )
                )
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        node = updated_node
        max_function_index = None
        max_class_index = None
        for index, _node in enumerate(node.body):
            if isinstance(_node, cst.FunctionDef):
                max_function_index = index
            if isinstance(_node, cst.ClassDef):
                max_class_index = index

        if self.new_classes:
            existing_class_names = {_node.name.value for _node in node.body if isinstance(_node, cst.ClassDef)}

            unique_classes = [
                new_class for new_class in self.new_classes if new_class.name.value not in existing_class_names
            ]
            if unique_classes:
                new_classes_insertion_idx = max_class_index or find_insertion_index_after_imports(node)
                new_body = list(
                    chain(node.body[:new_classes_insertion_idx], unique_classes, node.body[new_classes_insertion_idx:])
                )
                node = node.with_changes(body=new_body)

        if max_function_index is not None:
            node = node.with_changes(
                body=(*node.body[: max_function_index + 1], *self.new_functions, *node.body[max_function_index + 1 :])
            )
        elif max_class_index is not None:
            node = node.with_changes(
                body=(*node.body[: max_class_index + 1], *self.new_functions, *node.body[max_class_index + 1 :])
            )
        else:
            node = node.with_changes(body=(*self.new_functions, *node.body))
        return node


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
                    elif child.name.value == "__init__" and preexisting_objects:
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
    function_to_optimize: Optional[FunctionToOptimize] = None,
) -> bool:
    # Route to language-specific implementation for non-Python languages
    if not is_python():
        return replace_function_definitions_for_language(
            function_names, optimized_code, module_abspath, project_root_path, function_to_optimize
        )

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


def replace_function_definitions_for_language(
    function_names: list[str],
    optimized_code: CodeStringsMarkdown,
    module_abspath: Path,
    project_root_path: Path,
    function_to_optimize: Optional[FunctionToOptimize] = None,
) -> bool:
    """Replace function definitions for non-Python languages.

    Uses the language support abstraction to perform code replacement.

    Args:
        function_names: List of qualified function names to replace.
        optimized_code: The optimized code to apply.
        module_abspath: Path to the module file.
        project_root_path: Root of the project.
        function_to_optimize: The function being optimized (needed for line info).

    Returns:
        True if the code was modified, False if no changes.

    """
    from codeflash.languages import get_language_support
    from codeflash.languages.base import Language

    original_source_code: str = module_abspath.read_text(encoding="utf8")
    code_to_apply = get_optimized_code_for_module(module_abspath.relative_to(project_root_path), optimized_code)

    if not code_to_apply.strip():
        return False

    # Get language support
    language = Language(optimized_code.language)
    lang_support = get_language_support(language)

    # Add any new global declarations from the optimized code to the original source
    original_source_code = _add_global_declarations_for_language(
        optimized_code=code_to_apply,
        original_source=original_source_code,
        module_abspath=module_abspath,
        language=language,
    )

    # If we have function_to_optimize with line info and this is the main file, use it for precise replacement
    if (
        function_to_optimize
        and function_to_optimize.starting_line
        and function_to_optimize.ending_line
        and function_to_optimize.file_path == module_abspath
    ):
        # Extract just the target function from the optimized code
        optimized_func = _extract_function_from_code(
            lang_support, code_to_apply, function_to_optimize.function_name, module_abspath
        )
        if optimized_func:
            new_code = lang_support.replace_function(original_source_code, function_to_optimize, optimized_func)
        else:
            # Fallback: use the entire optimized code (for simple single-function files)
            new_code = lang_support.replace_function(original_source_code, function_to_optimize, code_to_apply)
    else:
        # For helper files or when we don't have precise line info:
        # Find each function by name in both original and optimized code
        # Then replace with the corresponding optimized version
        new_code = original_source_code
        modified = False

        # Get the list of function names to replace
        functions_to_replace = list(function_names)

        for func_name in functions_to_replace:
            # Re-discover functions from current code state to get correct line numbers
            current_functions = lang_support.discover_functions_from_source(new_code, module_abspath)

            # Find the function in current code
            func = None
            for f in current_functions:
                if func_name in (f.qualified_name, f.function_name):
                    func = f
                    break

            if func is None:
                continue

            # Extract just this function from the optimized code
            optimized_func = _extract_function_from_code(
                lang_support, code_to_apply, func.function_name, module_abspath
            )
            if optimized_func:
                new_code = lang_support.replace_function(new_code, func, optimized_func)
                modified = True

        if not modified:
            logger.warning(f"Could not find function {function_names} in {module_abspath}")
            return False

    # Check if there was actually a change
    if original_source_code.strip() == new_code.strip():
        return False

    module_abspath.write_text(new_code, encoding="utf8")
    return True


def _extract_function_from_code(
    lang_support: LanguageSupport, source_code: str, function_name: str, file_path: Path | None = None
) -> str | None:
    """Extract a specific function's source code from a code string.

    Includes JSDoc/docstring comments if present.

    Args:
        lang_support: Language support instance.
        source_code: The full source code containing the function.
        function_name: Name of the function to extract.
        file_path: Path to the file (used to determine correct analyzer for JS/TS).

    Returns:
        The function's source code (including doc comments), or None if not found.

    """
    try:
        # Use the language support to find functions in the source
        # file_path is needed for JS/TS to determine correct analyzer (TypeScript vs JavaScript)
        functions = lang_support.discover_functions_from_source(source_code, file_path)
        for func in functions:
            if func.function_name == function_name:
                # Extract the function's source using line numbers
                # Use doc_start_line if available to include JSDoc/docstring
                lines = source_code.splitlines(keepends=True)
                effective_start = func.doc_start_line or func.starting_line
                if effective_start and func.ending_line and effective_start <= len(lines):
                    func_lines = lines[effective_start - 1 : func.ending_line]
                    return "".join(func_lines)
    except Exception as e:
        logger.debug(f"Error extracting function {function_name}: {e}")

    return None


def _add_global_declarations_for_language(
    optimized_code: str, original_source: str, module_abspath: Path, language: Language
) -> str:
    """Add new global declarations from optimized code to original source.

    Finds module-level declarations (const, let, var, class, type, interface, enum)
    in the optimized code that don't exist in the original source and adds them.

    New declarations are inserted after any existing declarations they depend on.
    For example, if optimized code has `const _has = FOO.bar.bind(FOO)`, and `FOO`
    is already declared in the original source, `_has` will be inserted after `FOO`.

    Args:
        optimized_code: The optimized code that may contain new declarations.
        original_source: The original source code.
        module_abspath: Path to the module file (for parser selection).
        language: The language of the code.

    Returns:
        Original source with new declarations added in dependency order.

    """
    from codeflash.languages.base import Language

    if language not in (Language.JAVASCRIPT, Language.TYPESCRIPT):
        return original_source

    try:
        from codeflash.languages.javascript.treesitter import get_analyzer_for_file

        analyzer = get_analyzer_for_file(module_abspath)

        original_declarations = analyzer.find_module_level_declarations(original_source)
        optimized_declarations = analyzer.find_module_level_declarations(optimized_code)

        if not optimized_declarations:
            return original_source

        existing_names = _get_existing_names(original_declarations, analyzer, original_source)
        new_declarations = _filter_new_declarations(optimized_declarations, existing_names)

        if not new_declarations:
            return original_source

        # Build a map of existing declaration names to their end lines (1-indexed)
        existing_decl_end_lines = {decl.name: decl.end_line for decl in original_declarations}

        # Insert each new declaration after its dependencies
        result = original_source
        for decl in new_declarations:
            result = _insert_declaration_after_dependencies(
                result, decl, existing_decl_end_lines, analyzer, module_abspath
            )
            # Update the map with the newly inserted declaration for subsequent insertions
            # Re-parse to get accurate line numbers after insertion
            updated_declarations = analyzer.find_module_level_declarations(result)
            existing_decl_end_lines = {d.name: d.end_line for d in updated_declarations}

        return result

    except Exception as e:
        logger.debug(f"Error adding global declarations: {e}")
        return original_source


def _get_existing_names(original_declarations: list, analyzer: TreeSitterAnalyzer, original_source: str) -> set[str]:
    """Get all names that already exist in the original source (declarations + imports)."""
    existing_names = {decl.name for decl in original_declarations}

    original_imports = analyzer.find_imports(original_source)
    for imp in original_imports:
        if imp.default_import:
            existing_names.add(imp.default_import)
        for name, alias in imp.named_imports:
            existing_names.add(alias if alias else name)
        if imp.namespace_import:
            existing_names.add(imp.namespace_import)

    return existing_names


def _filter_new_declarations(optimized_declarations: list, existing_names: set[str]) -> list:
    """Filter declarations to only those that don't exist in the original source."""
    new_declarations = []
    seen_sources: set[str] = set()

    # Sort by line number to maintain order from optimized code
    sorted_declarations = sorted(optimized_declarations, key=lambda d: d.start_line)

    for decl in sorted_declarations:
        if decl.name not in existing_names and decl.source_code not in seen_sources:
            new_declarations.append(decl)
            seen_sources.add(decl.source_code)

    return new_declarations


def _insert_declaration_after_dependencies(
    source: str,
    declaration,
    existing_decl_end_lines: dict[str, int],
    analyzer: TreeSitterAnalyzer,
    module_abspath: Path,
) -> str:
    """Insert a declaration after the last existing declaration it depends on.

    Args:
        source: Current source code.
        declaration: The declaration to insert.
        existing_decl_end_lines: Map of existing declaration names to their end lines.
        analyzer: TreeSitter analyzer.
        module_abspath: Path to the module file.

    Returns:
        Source code with the declaration inserted at the correct position.

    """
    # Find identifiers referenced in this declaration
    referenced_names = analyzer.find_referenced_identifiers(declaration.source_code)

    # Find the latest end line among all referenced declarations
    insertion_line = _find_insertion_line_for_declaration(source, referenced_names, existing_decl_end_lines, analyzer)

    lines = source.splitlines(keepends=True)

    # Ensure proper spacing
    decl_code = declaration.source_code
    if not decl_code.endswith("\n"):
        decl_code += "\n"

    # Add blank line before if inserting after content
    if insertion_line > 0 and lines[insertion_line - 1].strip():
        decl_code = "\n" + decl_code

    before = lines[:insertion_line]
    after = lines[insertion_line:]

    return "".join([*before, decl_code, *after])


def _find_insertion_line_for_declaration(
    source: str, referenced_names: set[str], existing_decl_end_lines: dict[str, int], analyzer: TreeSitterAnalyzer
) -> int:
    """Find the line where a declaration should be inserted based on its dependencies.

    Args:
        source: Source code.
        referenced_names: Names referenced by the declaration.
        existing_decl_end_lines: Map of declaration names to their end lines (1-indexed).
        analyzer: TreeSitter analyzer.

    Returns:
        Line index (0-based) where the declaration should be inserted.

    """
    # Find the maximum end line among referenced declarations
    max_dependency_line = 0
    for name in referenced_names:
        if name in existing_decl_end_lines:
            max_dependency_line = max(max_dependency_line, existing_decl_end_lines[name])

    if max_dependency_line > 0:
        # Insert after the last dependency (end_line is 1-indexed, we need 0-indexed)
        return max_dependency_line

    # No dependencies found - insert after imports
    lines = source.splitlines(keepends=True)
    return _find_line_after_imports(lines, analyzer, source)


def _find_line_after_imports(lines: list[str], analyzer: TreeSitterAnalyzer, source: str) -> int:
    """Find the line index after all imports.

    Args:
        lines: Source lines.
        analyzer: TreeSitter analyzer.
        source: Full source code.

    Returns:
        Line index (0-based) for insertion after imports.

    """
    try:
        imports = analyzer.find_imports(source)
        if imports:
            return max(imp.end_line for imp in imports)
    except Exception as exc:
        logger.debug(f"Exception in _find_line_after_imports: {exc}")

    # Default: insert at beginning (after shebang/directive comments)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("//") and not stripped.startswith("#!"):
            return i

    return 0


def get_optimized_code_for_module(relative_path: Path, optimized_code: CodeStringsMarkdown) -> str:
    file_to_code_context = optimized_code.file_to_path()
    module_optimized_code = file_to_code_context.get(str(relative_path))
    if module_optimized_code is None:
        # Fallback: if there's only one code block with None file path,
        # use it regardless of the expected path (the AI server doesn't always include file paths)
        if "None" in file_to_code_context and len(file_to_code_context) == 1:
            module_optimized_code = file_to_code_context["None"]
            logger.debug(f"Using code block with None file_path for {relative_path}")
        else:
            logger.warning(
                f"Optimized code not found for {relative_path} In the context\n-------\n{optimized_code}\n-------\n"
                "re-check your 'markdown code structure'"
                f"existing files are {file_to_code_context.keys()}"
            )
            module_optimized_code = ""
    return module_optimized_code


def is_zero_diff(original_code: str, new_code: str) -> bool:
    return normalize_code(original_code) == normalize_code(new_code)


def replace_optimized_code(
    callee_module_paths: set[Path],
    candidates: list[OptimizedCandidate],
    code_context: CodeOptimizationContext,
    function_to_optimize: FunctionToOptimize,
    validated_original_code: dict[Path, ValidCode],
    project_root: Path,
) -> tuple[set[Path], dict[str, dict[Path, str]]]:
    initial_optimized_code = {
        candidate.optimization_id: replace_functions_and_add_imports(
            validated_original_code[function_to_optimize.file_path].source_code,
            [function_to_optimize.qualified_name],
            candidate.source_code,
            function_to_optimize.file_path,
            function_to_optimize.file_path,
            code_context.preexisting_objects,
            project_root,
        )
        for candidate in candidates
    }
    callee_original_code = {
        module_path: validated_original_code[module_path].source_code for module_path in callee_module_paths
    }
    intermediate_original_code: dict[str, dict[Path, str]] = {
        candidate.optimization_id: (
            callee_original_code | {function_to_optimize.file_path: initial_optimized_code[candidate.optimization_id]}
        )
        for candidate in candidates
    }
    module_paths = callee_module_paths | {function_to_optimize.file_path}
    optimized_code = {
        candidate.optimization_id: {
            module_path: replace_functions_and_add_imports(
                intermediate_original_code[candidate.optimization_id][module_path],
                (
                    [
                        callee.qualified_name
                        for callee in code_context.helper_functions
                        if callee.file_path == module_path and callee.definition_type != "class"
                    ]
                ),
                candidate.source_code,
                function_to_optimize.file_path,
                module_path,
                [],
                project_root,
            )
            for module_path in module_paths
        }
        for candidate in candidates
    }
    return module_paths, optimized_code


def is_optimized_module_code_zero_diff(
    candidates: list[OptimizedCandidate],
    validated_original_code: dict[Path, ValidCode],
    optimized_code: dict[str, dict[Path, str]],
    module_paths: set[Path],
) -> dict[str, dict[Path, bool]]:
    return {
        candidate.optimization_id: {
            callee_module_path: normalize_code(optimized_code[candidate.optimization_id][callee_module_path])
            == validated_original_code[callee_module_path].normalized_code
            for callee_module_path in module_paths
        }
        for candidate in candidates
    }


def candidates_with_diffs(
    candidates: list[OptimizedCandidate],
    validated_original_code: ValidCode,
    optimized_code: dict[str, dict[Path, str]],
    module_paths: set[Path],
) -> list[OptimizedCandidate]:
    return [
        candidate
        for candidate in candidates
        if not all(
            is_optimized_module_code_zero_diff(candidates, validated_original_code, optimized_code, module_paths)[
                candidate.optimization_id
            ].values()
        )
    ]


def replace_optimized_code_in_worktrees(
    optimized_code: dict[str, dict[Path, str]],
    candidates: list[OptimizedCandidate],  # Should be candidates_with_diffs
    worktrees: list[Path],
    git_root: Path,  # Handle None case
) -> None:
    for candidate, worktree in zip(candidates, worktrees[1:]):
        for module_path in optimized_code[candidate.optimization_id]:
            (worktree / module_path.relative_to(git_root)).write_text(
                optimized_code[candidate.optimization_id][module_path], encoding="utf8"
            )  # Check with is_optimized_module_code_zero_diff


def function_to_optimize_original_worktree_fqn(
    function_to_optimize: FunctionToOptimize, worktrees: list[Path], git_root: Path
) -> str:
    return (
        str(worktrees[0].name / function_to_optimize.file_path.relative_to(git_root).with_suffix("")).replace("/", ".")
        + "."
        + function_to_optimize.qualified_name
    )
