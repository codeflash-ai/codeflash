import isort
import libcst as cst

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class AddDecoratorTransformer(cst.CSTTransformer):
    def __init__(self, function_name, class_name=None):
        super().__init__()
        self.function_name = function_name
        self.class_name = class_name
        self.in_target_class = (class_name is None)  # If no class name, always "in target class"
        self.added_codeflash_trace = False

    def leave_ClassDef(self, original_node, updated_node):
        if self.class_name and original_node.name.value == self.class_name:
            self.in_target_class = False
        return updated_node

    def visit_ClassDef(self, node):
        if self.class_name and node.name.value == self.class_name:
            self.in_target_class = True
        return True

    def leave_FunctionDef(self, original_node, updated_node):
        if not self.in_target_class or original_node.name.value != self.function_name:
            return updated_node

        # Create the codeflash_trace decorator
        decorator = cst.Decorator(
            decorator=cst.Name(value="codeflash_trace")
        )

        # Add the new decorator after any existing decorators
        updated_decorators = list(updated_node.decorators) + [decorator]
        self.added_codeflash_trace = True
        # Return the updated node with the new decorator
        return updated_node.with_changes(
            decorators=updated_decorators
        )

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Create import statement for codeflash_trace
        if not self.added_codeflash_trace:
            return updated_node
        import_stmt = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Attribute(
                        value=cst.Attribute(
                            value=cst.Name(value="codeflash"),
                            attr=cst.Name(value="benchmarking")
                        ),
                        attr=cst.Name(value="codeflash_trace")
                    ),
                    names=[
                        cst.ImportAlias(
                            name=cst.Name(value="codeflash_trace")
                        )
                    ]
                )
            ]
        )

        # Insert at the beginning of the file
        new_body = [import_stmt, *list(updated_node.body)]

        return updated_node.with_changes(body=new_body)

def add_codeflash_decorator_to_code(code: str, function_to_optimize: FunctionToOptimize) -> str:
    """Add codeflash_trace to a function.

    Args:
        code: The source code as a string
        function_to_optimize: The FunctionToOptimize instance containing function details

    Returns:
        The modified source code as a string
    """
    # Extract class name if present
    class_name = None
    if len(function_to_optimize.parents) == 1 and function_to_optimize.parents[0].type == "ClassDef":
        class_name = function_to_optimize.parents[0].name

    transformer = AddDecoratorTransformer(
        function_name=function_to_optimize.function_name,
        class_name=class_name
    )

    module = cst.parse_module(code)
    modified_module = module.visit(transformer)
    return modified_module.code


def instrument_codeflash_trace_decorator(
        function_to_optimize: FunctionToOptimize
) -> None:
    """Instrument __init__ function with codeflash_trace decorator if it's in a class."""
    # Instrument fto class
    original_code = function_to_optimize.file_path.read_text(encoding="utf-8")
    new_code = add_codeflash_decorator_to_code(
        original_code,
        function_to_optimize
    )
    # Modify the code
    modified_code = isort.code(code=new_code, float_to_top=True)

    # Write the modified code back to the file
    function_to_optimize.file_path.write_text(modified_code, encoding="utf-8")
