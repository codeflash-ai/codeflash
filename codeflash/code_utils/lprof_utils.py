import isort
import libcst as cst
from pathlib import Path
from codeflash.code_utils.code_utils import get_run_tmp_file

def add_decorator_cst(module_node, function_name, decorator_name):
    """Adds a decorator to a function definition in a LibCST module node."""

    class AddDecoratorTransformer(cst.CSTTransformer):
        def leave_FunctionDef(self, original_node, updated_node):
            if original_node.name.value == function_name:
                new_decorator = cst.Decorator(
                    decorator=cst.Name(value=decorator_name)
                )

                updated_decorators = list(updated_node.decorators)
                updated_decorators.insert(0, new_decorator)

                return updated_node.with_changes(decorators=updated_decorators)
            return updated_node

    transformer = AddDecoratorTransformer()
    updated_module = module_node.visit(transformer)
    return updated_module

def add_decorator_imports(file_paths, fn_list, db_file):
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
        file_contents = f.readlines()
    for idx, line in enumerate(file_contents):
        if 'from line_profiler import profile' in line:
            file_contents.insert(idx+1, f"profile.enable(output_prefix='{db_file}')\n")
            break
    with open(file_paths[0],'w') as f:
        f.writelines(file_contents)




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

def prepare_lprofiler_files(prefix: str = "") -> tuple[Path]:
    """Prepare line profiler output file."""
    lprofiler_database_file = get_run_tmp_file(Path(prefix))
    return lprofiler_database_file