import ast
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def instrument_code(function_to_optimize: FunctionToOptimize) -> None:
    """Instrument __init__ function with codeflash_capture decorator if it's in a class."""
    # Find the class parent
    if len(function_to_optimize.parents) == 1 and function_to_optimize.parents[0].type == "ClassDef":
        class_parent = function_to_optimize.parents[0]
    else:
        return

    # Read code from file
    with open(function_to_optimize.file_path) as f:
        original_code = f.read()

    # Add decorator to init
    modified_code = add_codeflash_capture_to_init(
        class_name=class_parent.name,
        function_name=function_to_optimize.function_name,
        tmp_dir_path=str(get_run_tmp_file(Path("test_return_values"))),
        code=original_code,
    )

    # Write modified code back to file
    with open(function_to_optimize.file_path, "w") as f:
        f.write(modified_code)


def add_codeflash_capture_to_init(class_name: str, function_name: str, tmp_dir_path: str, code: str) -> str:
    """Add codeflash_capture decorator to __init__ function in the specified class."""
    # Parse the code into an AST
    tree = ast.parse(code)

    # Apply our transformation
    transformer = InitDecorator(class_name, function_name, tmp_dir_path)
    modified_tree = transformer.visit(tree)
    if transformer.inserted_decorator:
        ast.fix_missing_locations(modified_tree)

    # Convert back to source code
    return ast.unparse(modified_tree)


class InitDecorator(ast.NodeTransformer):
    """AST transformer that adds codeflash_capture decorator to specific class's __init__."""

    def __init__(self, target_class_name: str, function_name: str, tmp_dir_path: str):
        self.target_class_name = target_class_name
        self.function_name = function_name
        self.tmp_dir_path = tmp_dir_path
        self.has_import = False
        self.inserted_decorator = False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        # Check if our import already exists
        if node.module == "codeflash.verification.codeflash_capture" and any(
            alias.name == "codeflash_capture" for alias in node.names
        ):
            self.has_import = True
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        # Add import statement
        if not self.has_import and self.inserted_decorator:
            import_stmt = ast.parse("from codeflash.verification.codeflash_capture import codeflash_capture").body[0]
            node.body.insert(0, import_stmt)

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Only modify the target class
        if node.name != self.target_class_name:
            return node

        # Look for __init__ method
        has_init = False

        # Create the decorator
        decorator = ast.Call(
            func=ast.Name(id="codeflash_capture", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="function_name", value=ast.Constant(value=self.function_name)),
                ast.keyword(arg="tmp_dir_path", value=ast.Constant(value=self.tmp_dir_path)),
            ],
        )

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                has_init = True

                # Add decorator at the start of the list if not already present
                if not any(
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "codeflash_capture"
                    for d in item.decorator_list
                ):
                    item.decorator_list.insert(0, decorator)
                    self.inserted_decorator = True

        if not has_init:
            # Create super().__init__(*args, **kwargs) call
            super_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[]),
                        attr="__init__",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Starred(value=ast.Name(id="args", ctx=ast.Load()))],
                    keywords=[ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))],
                )
            )
            # Create function arguments: self, *args, **kwargs
            arguments = ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=None)],
                vararg=ast.arg(arg="args"),
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=ast.arg(arg="kwargs"),
                defaults=[],
            )

            # Create the complete function
            init_func = ast.FunctionDef(
                name="__init__", args=arguments, body=[super_call], decorator_list=[decorator], returns=None
            )

            node.body.insert(0, init_func)
            self.inserted_decorator = True

        return node
