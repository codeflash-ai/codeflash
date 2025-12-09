"""Adapted from line_profiler (https://github.com/pyutils/line_profiler) written by Enthought, Inc. (BSD License)."""

from __future__ import annotations

import functools
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

import libcst as cst
from line_profiler import LineProfiler

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.formatter import sort_imports

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CodeOptimizationContext


class LineProfilerDecoratorAdder(cst.CSTTransformer):
    """Transformer that adds a decorator to a function with a specific qualified name."""

    # TODO we don't support nested functions yet so they can only be inside classes, dont use qualified names, instead use the structure
    def __init__(self, qualified_name: str, decorator_name: str) -> None:
        """Initialize the transformer.

        Args:
        ----
            qualified_name: The fully qualified name of the function to add the decorator to (e.g., "MyClass.nested_func.target_func").
            decorator_name: The name of the decorator to add.

        """
        super().__init__()
        self.qualified_name_parts = qualified_name.split(".")
        self.decorator_name = decorator_name

        # Track our current context path, only add when we encounter a class
        self.context_stack = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Track when we enter a class
        self.context_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:  # noqa: ARG002
        # Pop the context when we leave a class
        self.context_stack.pop()
        return updated_node

    # f6fa22dd-d711-425b-bed4-d2844ad6af36
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Track when we enter a function
        self.context_stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Check if the current context path matches our target qualified name
        if self.context_stack == self.qualified_name_parts:
            # Check if the decorator is already present
            has_decorator = any(
                self._is_target_decorator(decorator.decorator) for decorator in original_node.decorators
            )

            # Only add the decorator if it's not already there
            if not has_decorator:
                new_decorator = cst.Decorator(decorator=cst.Name(value=self.decorator_name))

                # Add our new decorator to the existing decorators
                updated_decorators = [new_decorator, *list(updated_node.decorators)]
                updated_node = updated_node.with_changes(decorators=tuple(updated_decorators))

        # Pop the context when we leave a function
        self.context_stack.pop()
        return updated_node

    def _is_target_decorator(self, decorator_node: Union[cst.Name, cst.Attribute, cst.Call]) -> bool:
        """Check if a decorator matches our target decorator name."""
        if isinstance(decorator_node, cst.Name):
            return decorator_node.value == self.decorator_name
        if isinstance(decorator_node, cst.Call) and isinstance(decorator_node.func, cst.Name):
            return decorator_node.func.value == self.decorator_name
        return False


# Helper to get full module path as string
def get_module_path(node: Union[cst.Name, cst.Attribute]) -> str:
    parts = []
    while isinstance(node, cst.Attribute):
        parts.insert(0, node.attr.value)
        node = node.value
    if isinstance(node, cst.Name):
        parts.insert(0, node.value)
    return ".".join(parts)


class ProfileEnableTransformer(cst.CSTTransformer):
    def __init__(self, filename: str) -> None:
        # Flag to track if we found the import statement
        self.found_import = False
        # Track indentation of the import statement
        self.import_indentation = None
        self.filename = filename

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        module_path = get_module_path(original_node.module)

        if module_path == "codeflash.code_utils.line_profile_utils" and any(
            name.name.value == "LineProfilerDecorator" for name in original_node.names
        ):
            print("Found import statement")
            self.found_import = True
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: ARG002
        if not self.found_import:
            return updated_node

        # Create a list of statements from the original module
        new_body = list(updated_node.body)

        # Find the index of the import statement
        import_index = None
        for i, stmt in enumerate(new_body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for small_stmt in stmt.body:
                    if isinstance(small_stmt, cst.ImportFrom):
                        module_path = get_module_path(small_stmt.module)
                        if module_path == "codeflash.code_utils.line_profile_utils" and any(
                            name.name.value == "LineProfilerDecorator" for name in small_stmt.names
                        ):
                            import_index = i
                            break
                if import_index is not None:
                    break

        if import_index is not None:
            assignment = cst.parse_statement(f"codeflash_line_profile = LineProfilerDecorator('{self.filename}')")
            new_body.insert(import_index + 1, assignment)

        # Create a new module with the updated body
        return updated_node.with_changes(body=new_body)


def add_decorator_to_qualified_function(module: cst.Module, qualified_name: str, decorator_name: str) -> cst.Module:
    """Add a decorator to a function with the exact qualified name in the source code.

    Args:
    ----
        module: The Python source code as a CST module.
        qualified_name: The fully qualified name of the function to add the decorator to (e.g., "MyClass.nested_func.target_func").
        decorator_name: The name of the decorator to add.

    Returns:
    -------
        The modified CST module.

    """
    transformer = LineProfilerDecoratorAdder(qualified_name, decorator_name)
    return module.visit(transformer)


def add_profile_enable(original_code: str, line_profile_output_file: str) -> str:
    # TODO modify by using a libcst transformer
    module = cst.parse_module(original_code)
    transformer = ProfileEnableTransformer(line_profile_output_file)
    modified_module = module.visit(transformer)
    return modified_module.code


class ImportAdder(cst.CSTTransformer):
    def __init__(self, import_statement) -> None:  # noqa: ANN001
        self.import_statement = import_statement
        self.has_import = False

    def leave_Module(self, original_node, updated_node):  # noqa: ANN001, ANN201, ARG002
        # If the import is already there, don't add it again
        if self.has_import:
            return updated_node

        # Parse the import statement into a CST node
        import_node = cst.parse_statement(self.import_statement)

        # Add the import to the module's body
        return updated_node.with_changes(body=[import_node, *list(updated_node.body)])

    def visit_ImportFrom(self, node) -> None:  # noqa: ANN001
        # Check if the profile is already imported from line_profiler
        if node.module and node.module.value == "line_profiler":
            for import_alias in node.names:
                if import_alias.name.value == "profile":
                    self.has_import = True


def add_decorator_imports(function_to_optimize: FunctionToOptimize, code_context: CodeOptimizationContext) -> Path:
    """Add a profile decorator to a function in a Python file and all its helper functions."""
    # self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
    # grouped iteration, file to fns to optimize, from line_profiler import profile as codeflash_line_profile
    file_paths = defaultdict(list)
    line_profile_output_file = get_run_tmp_file(Path("baseline_lprof.txt"))
    file_paths[function_to_optimize.file_path].append(function_to_optimize.qualified_name)
    for elem in code_context.helper_functions:
        file_paths[elem.file_path].append(elem.qualified_name)
    for file_path, fns_present in file_paths.items():
        # open file
        file_contents = file_path.read_text("utf-8")
        # parse to cst
        module_node = cst.parse_module(file_contents)
        for fn_name in fns_present:
            # add decorator
            module_node = add_decorator_to_qualified_function(module_node, fn_name, "codeflash_line_profile")
        # add imports
        # Create a transformer to add the import
        transformer = ImportAdder("from codeflash.code_utils.line_profile_utils import LineProfilerDecorator")
        # Apply the transformer to add the import
        module_node = module_node.visit(transformer)
        modified_code = sort_imports(code=module_node.code, float_to_top=True)
        # write to file
        with file_path.open("w", encoding="utf-8") as file:
            file.write(modified_code)
    # Adding profile.enable line for changing the savepath of the data, do this only for the main file and not the helper files
    file_contents = function_to_optimize.file_path.read_text("utf-8")
    modified_code = add_profile_enable(file_contents, line_profile_output_file.as_posix())
    function_to_optimize.file_path.write_text(modified_code, "utf-8")
    return line_profile_output_file


class LineProfilerDecorator:
    """Decorator class that profiles multiple functions and saves stats automatically."""

    def __init__(self, output_file: str | Path) -> None:
        self.output_file: Path = Path(output_file)
        self.profiler: LineProfiler = LineProfiler()
        # Ensure parent folder exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorate a function to profile it and save stats after execution."""
        self.profiler.add_function(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            try:
                return self.profiler(func)(*args, **kwargs)
            finally:
                # Save stats after each call
                with self.output_file.open("w") as f:
                    self.profiler.print_stats(f)

        return wrapper
