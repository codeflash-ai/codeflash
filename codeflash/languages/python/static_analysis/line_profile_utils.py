"""Adapted from line_profiler (https://github.com/pyutils/line_profiler) written by Enthought, Inc. (BSD License)."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

import libcst as cst

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.formatter import sort_imports
from codeflash.languages.python.static_analysis._ast import get_attribute_parts

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

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # Pop the context when we leave a class
        self.context_stack.pop()
        return updated_node

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


class ProfileEnableTransformer(cst.CSTTransformer):
    def __init__(self, filename: str) -> None:
        # Flag to track if we found the import statement
        self.found_import = False
        # Track indentation of the import statement
        self.import_indentation = None
        self.filename = filename

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        # Check if this is the line profiler import statement
        if (
            isinstance(original_node.module, cst.Name)
            and original_node.module.value == "line_profiler"
            and any(
                name.name.value == "profile" and (not name.asname or name.asname.name.value == "codeflash_line_profile")
                for name in original_node.names
            )
        ):
            self.found_import = True
            # Get the indentation from the original node
            if hasattr(original_node, "leading_lines"):
                leading_whitespace = original_node.leading_lines[-1].whitespace if original_node.leading_lines else ""
                self.import_indentation = leading_whitespace

        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.found_import:
            return updated_node

        # Create a list of statements from the original module
        new_body = list(updated_node.body)

        # Find the index of the import statement
        import_index = None
        for i, stmt in enumerate(new_body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for small_stmt in stmt.body:
                    if isinstance(small_stmt, cst.ImportFrom) and (
                        isinstance(small_stmt.module, cst.Name)
                        and small_stmt.module.value == "line_profiler"
                        and any(
                            name.name.value == "profile"
                            and (not name.asname or name.asname.name.value == "codeflash_line_profile")
                            for name in small_stmt.names
                        )
                    ):
                        import_index = i
                        break
                if import_index is not None:
                    break

        if import_index is not None:
            # Create the new enable statement to insert after the import
            enable_statement = cst.parse_statement(f"codeflash_line_profile.enable(output_prefix='{self.filename}')")

            # Insert the new statement after the import statement
            new_body.insert(import_index + 1, enable_statement)

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
    def __init__(self, import_statement) -> None:
        self.import_statement = import_statement
        self.has_import = False

    def leave_Module(self, original_node, updated_node):  # noqa: ANN201
        # If the import is already there, don't add it again
        if self.has_import:
            return updated_node

        # Parse the import statement into a CST node
        import_node = cst.parse_statement(self.import_statement)

        # Add the import to the module's body
        return updated_node.with_changes(body=[import_node, *list(updated_node.body)])

    def visit_ImportFrom(self, node) -> None:
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
    line_profile_output_file = get_run_tmp_file(Path("baseline_lprof"))
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
        transformer = ImportAdder("from line_profiler import profile as codeflash_line_profile")
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


JIT_DECORATORS: dict[str, set[str]] = {
    "numba": {"jit", "njit", "vectorize", "guvectorize", "stencil", "cfunc", "generated_jit"},
    "numba.cuda": {"jit"},
    "torch": {"compile"},
    "torch.jit": {"script", "trace"},
    "tensorflow": {"function"},
    "jax": {"jit"},
}


class JitDecoratorDetector(ast.NodeVisitor):
    """AST visitor that detects JIT compilation decorators considering import aliases."""

    def __init__(self) -> None:
        # Maps local name -> (module, original_name)
        # e.g., {"nb": ("numba", None), "my_jit": ("numba", "jit")}
        self.import_aliases: dict[str, tuple[str, str | None]] = {}
        self.found_jit_decorator = False

    def visit_Import(self, node: ast.Import) -> None:
        """Track regular imports like 'import numba' or 'import numba as nb'."""
        for alias in node.names:
            # alias.name is the module name, alias.asname is the alias (or None)
            local_name = alias.asname if alias.asname else alias.name
            # For module imports, we store (module_name, None) to indicate it's a module import
            self.import_aliases[local_name] = (alias.name, None)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports like 'from numba import jit' or 'from numba import jit as my_jit'."""
        if node.module is None:
            self.generic_visit(node)
            return

        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            # For from imports, we store (module_name, imported_name)
            self.import_aliases[local_name] = (node.module, alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function decorators for JIT decorators."""
        for decorator in node.decorator_list:
            if self._is_jit_decorator(decorator):
                self.found_jit_decorator = True
                return
        self.generic_visit(node)

    def _is_jit_decorator(self, node: ast.expr) -> bool:
        """Check if a decorator node is a known JIT decorator."""
        # Handle Call nodes (e.g., @jit() or @numba.jit(nopython=True))
        if isinstance(node, ast.Call):
            return self._is_jit_decorator(node.func)

        # Handle simple Name nodes (e.g., @jit when imported directly)
        if isinstance(node, ast.Name):
            return self._check_name_decorator(node.id)

        # Handle Attribute nodes (e.g., @numba.jit or @nb.jit)
        if isinstance(node, ast.Attribute):
            return self._check_attribute_decorator(node)

        return False

    def _check_name_decorator(self, name: str) -> bool:
        """Check if a simple name decorator (e.g., @jit) is a JIT decorator."""
        if name not in self.import_aliases:
            return False

        module, imported_name = self.import_aliases[name]

        if imported_name is None:
            # This is a module import used as decorator (unlikely but possible)
            return False

        # Check if this is a known JIT decorator from the module
        return self._is_known_jit_decorator(module, imported_name)

    def _check_attribute_decorator(self, node: ast.Attribute) -> bool:
        """Check if an attribute decorator (e.g., @numba.jit) is a JIT decorator."""
        # Build the full attribute chain
        parts = get_attribute_parts(node) or []
        if not parts:
            return False

        # The first part might be an alias
        first_part = parts[0]
        rest_parts = parts[1:]

        # Check if first_part is an imported alias
        if first_part in self.import_aliases:
            module, imported_name = self.import_aliases[first_part]

            if imported_name is None:
                # It's a module import (e.g., import numba as nb)
                # The full path is module + rest_parts
                if rest_parts:
                    full_module = module
                    decorator_name = rest_parts[-1]
                    if len(rest_parts) > 1:
                        full_module = f"{module}.{'.'.join(rest_parts[:-1])}"
                    return self._is_known_jit_decorator(full_module, decorator_name)
            # It's a from import of something that has attributes
            # e.g., from torch import jit; @jit.script
            elif rest_parts:
                full_module = f"{module}.{imported_name}"
                decorator_name = rest_parts[-1]
                if len(rest_parts) > 1:
                    full_module = f"{full_module}.{'.'.join(rest_parts[:-1])}"
                return self._is_known_jit_decorator(full_module, decorator_name)
        # first_part is used directly (e.g., @numba.jit without import alias)
        # Reconstruct the full path
        elif rest_parts:
            full_module = first_part
            if len(rest_parts) > 1:
                full_module = f"{first_part}.{'.'.join(rest_parts[:-1])}"
            decorator_name = rest_parts[-1]
            return self._is_known_jit_decorator(full_module, decorator_name)

        return False

    def _is_known_jit_decorator(self, module: str, decorator_name: str) -> bool:
        """Check if a decorator from a module is a known JIT decorator."""
        if module in JIT_DECORATORS:
            return decorator_name in JIT_DECORATORS[module]
        return False


def contains_jit_decorator(code: str) -> bool:
    """Check if the code contains JIT compilation decorators from numba, torch, tensorflow, or jax.

    This function uses AST parsing to accurately detect JIT decorators even when:
    - They are imported with aliases (e.g., import numba as nb; @nb.jit)
    - They are imported directly (e.g., from numba import jit; @jit)
    - They are called with arguments (e.g., @jit(nopython=True))
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    detector = JitDecoratorDetector()
    detector.visit(tree)
    return detector.found_jit_decorator
