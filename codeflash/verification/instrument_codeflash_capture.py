from __future__ import annotations

import ast
from pathlib import Path

import isort

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def instrument_codeflash_capture(
    function_to_optimize: FunctionToOptimize, file_path_to_helper_class: dict[Path, set[str]], tests_root: Path
) -> None:
    """Instrument __init__ function with codeflash_capture decorator if it's in a class."""
    # Find the class parent
    if len(function_to_optimize.parents) == 1 and function_to_optimize.parents[0].type == "ClassDef":
        class_parent = function_to_optimize.parents[0]
    else:
        return
    # Remove duplicate fto class from helper classes
    if (
        function_to_optimize.file_path in file_path_to_helper_class
        and class_parent.name in file_path_to_helper_class[function_to_optimize.file_path]
    ):
        file_path_to_helper_class[function_to_optimize.file_path].remove(class_parent.name)
    # Instrument fto class
    original_code = function_to_optimize.file_path.read_text(encoding="utf-8")
    # Add decorator to init
    modified_code = add_codeflash_capture_to_init(
        target_classes={class_parent.name},
        fto_name=function_to_optimize.function_name,
        tmp_dir_path=str(get_run_tmp_file(Path("test_return_values"))),
        code=original_code,
        tests_root=tests_root,
        is_fto=True,
    )
    function_to_optimize.file_path.write_text(modified_code, encoding="utf-8")

    # Instrument helper classes
    for file_path, helper_classes in file_path_to_helper_class.items():
        original_code = file_path.read_text(encoding="utf-8")
        modified_code = add_codeflash_capture_to_init(
            target_classes=helper_classes,
            fto_name=function_to_optimize.function_name,
            tmp_dir_path=str(get_run_tmp_file(Path("test_return_values"))),
            code=original_code,
            tests_root=tests_root,
            is_fto=False,
        )
        file_path.write_text(modified_code, encoding="utf-8")


def add_codeflash_capture_to_init(
    target_classes: set[str], fto_name: str, tmp_dir_path: str, code: str, tests_root: Path, is_fto: bool = False
) -> str:
    """Add codeflash_capture decorator to __init__ function in the specified class."""
    tree = ast.parse(code)
    transformer = InitDecorator(target_classes, fto_name, tmp_dir_path, tests_root, is_fto)
    modified_tree = transformer.visit(tree)
    if transformer.inserted_decorator:
        ast.fix_missing_locations(modified_tree)

    # Convert back to source code
    return isort.code(code=ast.unparse(modified_tree), float_to_top=True)


class InitDecorator(ast.NodeTransformer):
    """AST transformer that adds codeflash_capture decorator to specific class's __init__."""

    def __init__(
        self, target_classes: set[str], fto_name: str, tmp_dir_path: str, tests_root: Path, is_fto=False
    ) -> None:
        self.target_classes = target_classes
        self.fto_name = fto_name
        self.tmp_dir_path = tmp_dir_path
        self.is_fto = is_fto
        self.has_import = False
        self.tests_root = tests_root
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
        if node.name not in self.target_classes:
            return node

        # Look for __init__ method
        has_init = False

        # Create the decorator
        decorator = ast.Call(
            func=ast.Name(id="codeflash_capture", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="function_name", value=ast.Constant(value=".".join([node.name, "__init__"]))),
                ast.keyword(arg="tmp_dir_path", value=ast.Constant(value=self.tmp_dir_path)),
                ast.keyword(arg="tests_root", value=ast.Constant(value=str(self.tests_root))),
                ast.keyword(arg="is_fto", value=ast.Constant(value=self.is_fto)),
            ],
        )

        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef)
                and item.name == "__init__"
                and item.args.args
                and isinstance(item.args.args[0], ast.arg)
                and item.args.args[0].arg == "self"
            ):
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
