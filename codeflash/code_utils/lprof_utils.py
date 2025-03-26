import isort
import libcst as cst
from pathlib import Path
from typing import Union

from codeflash.code_utils.code_utils import get_run_tmp_file


class DecoratorAdder(cst.CSTTransformer):
    """Transformer that adds a decorator to a function with a specific qualified name."""

    def __init__(self, qualified_name: str, decorator_name: str):
        """
        Initialize the transformer.

        Args:
            qualified_name: The fully qualified name of the function to add the decorator to (e.g., "MyClass.nested_func.target_func").
            decorator_name: The name of the decorator to add.
        """
        super().__init__()
        self.qualified_name_parts = qualified_name.split(".")
        self.decorator_name = decorator_name

        # Track our current context path
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
        if self._matches_qualified_path():
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

    def _matches_qualified_path(self) -> bool:
        """Check if the current context stack matches the qualified name."""
        if len(self.context_stack) != len(self.qualified_name_parts):
            return False

        for i, name in enumerate(self.qualified_name_parts):
            if self.context_stack[i] != name:
                return False

        return True

    def _is_target_decorator(self, decorator_node: Union[cst.Name, cst.Attribute, cst.Call]) -> bool:
        """Check if a decorator matches our target decorator name."""
        if isinstance(decorator_node, cst.Name):
            return decorator_node.value == self.decorator_name
        elif isinstance(decorator_node, cst.Call) and isinstance(decorator_node.func, cst.Name):
            return decorator_node.func.value == self.decorator_name
        return False


def add_decorator_to_qualified_function(module, qualified_name, decorator_name):
    """
    Add a decorator to a function with the exact qualified name in the source code.

    Args:
        module: The Python source code as a string.
        qualified_name: The fully qualified name of the function to add the decorator to (e.g., "MyClass.nested_func.target_func").
        decorator_name: The name of the decorator to add.

    Returns:
        The modified source code as a string.
    """
    # Parse the source code into a CST

    # Apply our transformer
    transformer = DecoratorAdder(qualified_name, decorator_name)
    modified_module = module.visit(transformer)

    # Convert the modified CST back to source code
    return modified_module

def add_profile_enable(original_code: str, db_file: str) -> str:
    module = cst.parse_module(original_code)
    found_index = -1

    for idx, statement in enumerate(module.body):
        if isinstance(statement, cst.SimpleStatementLine):
            for stmt in statement.body:
                if isinstance(stmt, cst.ImportFrom):
                    if stmt.module and stmt.module.value == 'line_profiler':
                        for name in stmt.names:
                            if isinstance(name, cst.ImportAlias):
                                if name.name.value == 'profile' and name.asname is None:
                                    found_index = idx
                                    break
                        if found_index != -1:
                            break
        if found_index != -1:
            break

    if found_index == -1:
        return original_code  # or raise an exception if the import is not found

    # Create the new line to insert
    new_line = f"profile.enable(output_prefix='{db_file}')\n"
    new_statement = cst.parse_statement(new_line)

    # Insert the new statement into the module's body
    new_body = list(module.body)
    new_body.insert(found_index + 1, new_statement)
    modified_module = module.with_changes(body=new_body)

    return modified_module.code


class ImportAdder(cst.CSTTransformer):
    def __init__(self, import_statement='from line_profiler import profile'):
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
    #self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root
    #todo change function signature to get filepaths of fn, helpers and db
    # modify libcst parser to visit with qualified name
    file_paths = list()
    fn_list = list()
    db_file = get_run_tmp_file(Path("baseline"))
    file_paths.append(function_to_optimize.file_path)
    fn_list.append(function_to_optimize.qualified_name)
    for elem in code_context.helper_functions:
        file_paths.append(elem.file_path)
        fn_list.append(elem.qualified_name)
    """Adds a decorator to a function in a Python file."""
    for file_path, fn_name in zip(file_paths, fn_list):
        #open file
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents = file.read()
        # parse to cst
        module_node = cst.parse_module(file_contents)
        # add decorator
        module_node = add_decorator_to_qualified_function(module_node, fn_name, 'profile')
        # add imports
        # Create a transformer to add the import
        transformer = ImportAdder("from line_profiler import profile")
        # Apply the transformer to add the import
        module_node = module_node.visit(transformer)
        modified_code = isort.code(module_node.code, float_to_top=True)
        # write to file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(modified_code)
    #Adding profile.enable line for changing the savepath of the data, do this only for the main file and not the helper files, can use libcst seems like an overkill, will go just with some simple string manipulation
    with open(file_paths[0],'r') as f:
        file_contents = f.read()
    modified_code = add_profile_enable(file_contents,db_file)
    with open(file_paths[0],'w') as f:
        f.write(modified_code)
    return db_file


def prepare_lprofiler_files(prefix: str = "") -> tuple[Path]:
    """Prepare line profiler output file."""
    lprofiler_database_file = get_run_tmp_file(Path(prefix))
    return lprofiler_database_file