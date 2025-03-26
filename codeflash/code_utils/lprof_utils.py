import isort
import libcst as cst
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file

def add_decorator_cst(module_node, function_path, decorator_name):
    """
    Adds a decorator to a function or method definition in a LibCST module node.

    Args:
        module_node: LibCST module node
        function_path: String path to the function (e.g., 'function_name' or 'ClassName.method_name')
        decorator_name: Name of the decorator to add
    """
    path_parts = function_path.split('.')

    class AddDecoratorTransformer(cst.CSTTransformer):
        def __init__(self):
            super().__init__()
            self.current_class = None

        def visit_ClassDef(self, node):
            # Track when we enter a class that matches our path
            if len(path_parts) > 1 and node.name.value == path_parts[0]:
                self.current_class = node.name.value
            return True

        def leave_ClassDef(self, original_node, updated_node):
            # Reset class tracking when leaving a class node
            if self.current_class == original_node.name.value:
                self.current_class = None
            return updated_node

        def leave_FunctionDef(self, original_node, updated_node):
            # Handle standalone functions
            if len(path_parts) == 1 and original_node.name.value == path_parts[0] and self.current_class is None:
                return self._add_decorator(updated_node)
            # Handle class methods
            elif len(path_parts) == 2 and self.current_class == path_parts[0] and original_node.name.value == path_parts[1]:
                return self._add_decorator(updated_node)
            return updated_node

        def _add_decorator(self, node):
            # Create and add the decorator
            new_decorator = cst.Decorator(
                decorator=cst.Name(value=decorator_name)
            )

            # Check if this decorator already exists
            for decorator in node.decorators:
                if (isinstance(decorator.decorator, cst.Name) and
                        decorator.decorator.value == decorator_name):
                    return node  # Decorator already exists

            updated_decorators = list(node.decorators)
            updated_decorators.insert(0, new_decorator)

            return node.with_changes(decorators=updated_decorators)

    transformer = AddDecoratorTransformer()
    updated_module = module_node.visit(transformer)
    return updated_module

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
        module_node = add_decorator_cst(module_node, fn_name, 'profile')
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