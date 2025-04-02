"""Adapted from line_profiler (https://github.com/pyutils/line_profiler) written by Enthought, Inc. (BSD License)"""
from collections import defaultdict
from pathlib import Path
from typing import Union

import isort
import libcst as cst

from codeflash.code_utils.code_utils import get_run_tmp_file


class LineProfilerDecoratorAdder(cst.CSTTransformer):
    """Transformer that adds a decorator to a function with a specific qualified name."""

    #TODO we don't support nested functions yet so they can only be inside classes, dont use qualified names, instead use the structure
    def __init__(self, qualified_name: str, decorator_name: str):
        """Initialize the transformer.

        Args:
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
        function_name = original_node.name.value

        # Check if the current context path matches our target qualified name
        if self.context_stack==self.qualified_name_parts:
            # Check if the decorator is already present
            has_decorator = any(
                self._is_target_decorator(decorator.decorator)
                for decorator in original_node.decorators
            )

            # Only add the decorator if it's not already there
            if not has_decorator:
                new_decorator = cst.Decorator(
                    decorator=cst.Name(value=self.decorator_name)
                )

                # Add our new decorator to the existing decorators
                updated_decorators = [new_decorator] + list(updated_node.decorators)
                updated_node = updated_node.with_changes(
                    decorators=tuple(updated_decorators)
                )

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
    def __init__(self,filename):
        # Flag to track if we found the import statement
        self.found_import = False
        # Track indentation of the import statement
        self.import_indentation = None
        self.filename = filename

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        # Check if this is the line profiler import statement
        if (isinstance(original_node.module, cst.Name) and
                original_node.module.value == "line_profiler" and
                any(name.name.value == "profile" and
                    (not name.asname or name.asname.name.value == "codeflash_line_profile")
                    for name in original_node.names)):

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
                    if isinstance(small_stmt, cst.ImportFrom):
                        if (isinstance(small_stmt.module, cst.Name) and
                                small_stmt.module.value == "line_profiler" and
                                any(name.name.value == "profile" and
                                    (not name.asname or name.asname.name.value == "codeflash_line_profile")
                                    for name in small_stmt.names)):
                            import_index = i
                            break
                if import_index is not None:
                    break

        if import_index is not None:
            # Create the new enable statement to insert after the import
            enable_statement = cst.parse_statement(
                f"codeflash_line_profile.enable(output_prefix='{self.filename}')"
            )

            # Insert the new statement after the import statement
            new_body.insert(import_index + 1, enable_statement)

        # Create a new module with the updated body
        return updated_node.with_changes(body=new_body)

def add_decorator_to_qualified_function(module, qualified_name, decorator_name):
    """Add a decorator to a function with the exact qualified name in the source code.

    Args:
        module: The Python source code as a string.
        qualified_name: The fully qualified name of the function to add the decorator to (e.g., "MyClass.nested_func.target_func").
        decorator_name: The name of the decorator to add.

    Returns:
        The modified source code as a string.

    """
    # Parse the source code into a CST

    # Apply our transformer
    transformer = LineProfilerDecoratorAdder(qualified_name, decorator_name)
    modified_module = module.visit(transformer)

    # Convert the modified CST back to source code
    return modified_module

def add_profile_enable(original_code: str, line_profile_output_file: str) -> str:
    # TODO modify by using a libcst transformer
    module = cst.parse_module(original_code)
    transformer = ProfileEnableTransformer(line_profile_output_file)
    modified_module = module.visit(transformer)
    return modified_module.code


class ImportAdder(cst.CSTTransformer):
    def __init__(self, import_statement):
        self.import_statement = import_statement
        self.has_import = False

    def leave_Module(self, original_node, updated_node):
        # If the import is already there, don't add it again
        if self.has_import:
            return updated_node

        # Parse the import statement into a CST node
        import_node = cst.parse_statement(self.import_statement)

        # Add the import to the module's body
        return updated_node.with_changes(
            body=[import_node] + list(updated_node.body)
        )

    def visit_ImportFrom(self, node):
        # Check if the profile is already imported from line_profiler
        if node.module and node.module.value == "line_profiler":
            for import_alias in node.names:
                if import_alias.name.value == "profile":
                    self.has_import = True


def add_decorator_imports(function_to_optimize, code_context):
    """Adds a profile decorator to a function in a Python file and all its helper functions."""
    #self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
    #grouped iteration, file to fns to optimize, from line_profiler import profile as codeflash_line_profile
    file_paths = defaultdict(list)
    line_profile_output_file = get_run_tmp_file(Path("baseline_lprof"))
    file_paths[function_to_optimize.file_path].append(function_to_optimize.qualified_name)
    for elem in code_context.helper_functions:
        file_paths[elem.file_path].append(elem.qualified_name)
    for file_path,fns_present in file_paths.items():
        #open file
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
        modified_code = isort.code(module_node.code, float_to_top=True)
        # write to file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(modified_code)
    #Adding profile.enable line for changing the savepath of the data, do this only for the main file and not the helper files
    file_contents = function_to_optimize.file_path.read_text("utf-8")
    modified_code = add_profile_enable(file_contents,str(line_profile_output_file))
    function_to_optimize.file_path.write_text(modified_code,"utf-8")
    return line_profile_output_file
