from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.formatter import sort_imports
from codeflash.languages.python.context.code_context_extractor import _ATTRS_DECORATOR_NAMES, _ATTRS_NAMESPACES

if TYPE_CHECKING:
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
        tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
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
            tmp_dir_path=get_run_tmp_file(Path("test_return_values")).as_posix(),
            code=original_code,
            tests_root=tests_root,
            is_fto=False,
        )
        file_path.write_text(modified_code, encoding="utf-8")


def add_codeflash_capture_to_init(
    target_classes: set[str], fto_name: str, tmp_dir_path: str, code: str, tests_root: Path, *, is_fto: bool = False
) -> str:
    """Add codeflash_capture decorator to __init__ function in the specified class."""
    tree = ast.parse(code)
    transformer = InitDecorator(target_classes, fto_name, tmp_dir_path, tests_root, is_fto=is_fto)
    modified_tree = transformer.visit(tree)
    if transformer.inserted_decorator:
        ast.fix_missing_locations(modified_tree)

    # Convert back to source code
    return sort_imports(code=ast.unparse(modified_tree), float_to_top=True)


class InitDecorator(ast.NodeTransformer):
    """AST transformer that adds codeflash_capture decorator to specific class's __init__."""

    def __init__(
        self, target_classes: set[str], fto_name: str, tmp_dir_path: str, tests_root: Path, *, is_fto: bool = False
    ) -> None:
        self.target_classes = target_classes
        self.fto_name = fto_name
        self.tmp_dir_path = tmp_dir_path
        self.is_fto = is_fto
        self.has_import = False
        self.tests_root = tests_root
        self.inserted_decorator = False
        self._attrs_classes_to_patch: dict[str, ast.Call] = {}

        # Precompute decorator components to avoid reconstructing on every node visit
        # Only the `function_name` field changes per class
        self._base_decorator_keywords = [
            ast.keyword(arg="tmp_dir_path", value=ast.Constant(value=self.tmp_dir_path)),
            ast.keyword(arg="tests_root", value=ast.Constant(value=self.tests_root.as_posix())),
            ast.keyword(arg="is_fto", value=ast.Constant(value=self.is_fto)),
        ]
        self._base_decorator_func = ast.Name(id="codeflash_capture", ctx=ast.Load())

        # Preconstruct starred/kwargs for super init injection for perf
        self._super_starred = ast.Starred(value=ast.Name(id="args", ctx=ast.Load()))
        self._super_kwarg = ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))
        self._super_func = ast.Attribute(
            value=ast.Call(func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[]),
            attr="__init__",
            ctx=ast.Load(),
        )
        self._init_vararg = ast.arg(arg="args")
        self._init_kwarg = ast.arg(arg="kwargs")
        self._init_self_arg = ast.arg(arg="self", annotation=None)

        # Precreate commonly reused AST fragments for classes that lack __init__
        # Create the super().__init__(*args, **kwargs) Expr (reuse prebuilt pieces)
        self._super_call_expr = ast.Expr(
            value=ast.Call(func=self._super_func, args=[self._super_starred], keywords=[self._super_kwarg])
        )
        # Create function arguments: self, *args, **kwargs (reuse arg nodes)
        self._init_arguments = ast.arguments(
            posonlyargs=[],
            args=[self._init_self_arg],
            vararg=self._init_vararg,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=self._init_kwarg,
            defaults=[],
        )

        # Pre-build reusable AST nodes for _build_attrs_patch_block
        self._load_ctx = ast.Load()
        self._store_ctx = ast.Store()
        self._args_name_load = ast.Name(id="args", ctx=self._load_ctx)
        self._kwargs_name_load = ast.Name(id="kwargs", ctx=self._load_ctx)
        self._self_arg_node = ast.arg(arg="self")
        self._args_arg_node = ast.arg(arg="args")
        self._kwargs_arg_node = ast.arg(arg="kwargs")
        self._self_name_load = ast.Name(id="self", ctx=self._load_ctx)
        self._starred_args = ast.Starred(value=self._args_name_load, ctx=self._load_ctx)
        self._kwargs_keyword = ast.keyword(arg=None, value=self._kwargs_name_load)

        # Pre-parse the import statement to avoid repeated parsing in visit_Module
        self._import_stmt = ast.parse("from codeflash.verification.codeflash_capture import codeflash_capture").body[0]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        # Check if our import already exists
        if node.module == "codeflash.verification.codeflash_capture" and any(
            alias.name == "codeflash_capture" for alias in node.names
        ):
            self.has_import = True
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)

        # Insert module-level monkey-patch wrappers for attrs classes immediately after their
        # class definitions.  We do this before inserting the import so indices stay stable.
        if self._attrs_classes_to_patch:
            new_body: list[ast.stmt] = []
            for stmt in node.body:
                new_body.append(stmt)
                if isinstance(stmt, ast.ClassDef) and stmt.name in self._attrs_classes_to_patch:
                    new_body.extend(self._build_attrs_patch_block(stmt.name, self._attrs_classes_to_patch[stmt.name]))
            node.body = new_body

        # Add import statement
        if not self.has_import and self.inserted_decorator:
            node.body.insert(0, self._import_stmt)

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Only modify the target class
        if node.name not in self.target_classes:
            return node

        has_init = False
        # Build decorator node ONCE for each class, not per loop iteration
        decorator = ast.Call(
            func=self._base_decorator_func,
            args=[],
            keywords=[
                ast.keyword(arg="function_name", value=ast.Constant(value=f"{node.name}.__init__")),
                *self._base_decorator_keywords,
            ],
        )

        # Only scan node.body once for both __init__ and decorator check
        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef)
                and item.name == "__init__"
                and item.args.args
                and isinstance(item.args.args[0], ast.arg)
                and item.args.args[0].arg == "self"
            ):
                has_init = True

                # Check for existing decorator in-place, stop after finding one
                for d in item.decorator_list:
                    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "codeflash_capture":
                        break
                else:
                    # No decorator found
                    item.decorator_list.insert(0, decorator)
                    self.inserted_decorator = True

                break

        if not has_init:
            # Skip dataclasses — their __init__ is auto-generated at class creation time and isn't in the AST.
            # The synthetic __init__ with super().__init__(*args, **kwargs) overrides it and fails because
            # object.__init__() doesn't accept the dataclass field kwargs.
            # TODO: support by saving a reference to the generated __init__ before overriding, e.g.
            # _orig_init = ClassName.__init__; then calling _orig_init(self, *args, **kwargs) in the wrapper
            for dec in node.decorator_list:
                dec_name = self._expr_name(dec)
                if dec_name is not None and dec_name.endswith("dataclass"):
                    return node
                if dec_name is not None:
                    parts = dec_name.split(".")
                    if len(parts) >= 2 and parts[-2] in _ATTRS_NAMESPACES and parts[-1] in _ATTRS_DECORATOR_NAMES:
                        if isinstance(dec, ast.Call):
                            for kw in dec.keywords:
                                if kw.arg == "init" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                                    return node
                        self._attrs_classes_to_patch[node.name] = decorator
                        self.inserted_decorator = True
                        return node

            # Create super().__init__(*args, **kwargs) call (use prebuilt AST fragments)

            # Skip NamedTuples — their __init__ is synthesized and cannot be overwritten.
            for base in node.bases:
                base_name = self._expr_name(base)
                if base_name is not None and base_name.endswith("NamedTuple"):
                    return node

            # Create super().__init__(*args, **kwargs) call (use prebuilt AST fragments)
            super_call = self._super_call_expr
            # Create the complete function using prebuilt arguments/body but attach the class-specific decorator

            # Create the complete function
            init_func = ast.FunctionDef(
                name="__init__", args=self._init_arguments, body=[super_call], decorator_list=[decorator], returns=None
            )

            node.body.insert(0, init_func)
            self.inserted_decorator = True

        return node

    def _build_attrs_patch_block(self, class_name: str, decorator: ast.Call) -> list[ast.stmt]:
        orig_name = f"_codeflash_orig_{class_name}_init"
        patched_name = f"_codeflash_patched_{class_name}_init"

        # _codeflash_orig_ClassName_init = ClassName.__init__

        # Create class name nodes once
        class_name_load = ast.Name(id=class_name, ctx=self._load_ctx)

        # _codeflash_orig_ClassName_init = ClassName.__init__
        save_orig = ast.Assign(
            targets=[ast.Name(id=orig_name, ctx=self._store_ctx)],
            value=ast.Attribute(value=class_name_load, attr="__init__", ctx=self._load_ctx),
        )

        # def _codeflash_patched_ClassName_init(self, *args, **kwargs):
        #     return _codeflash_orig_ClassName_init(self, *args, **kwargs)
        patched_func = ast.FunctionDef(
            name=patched_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[self._self_arg_node],
                vararg=self._args_arg_node,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=self._kwargs_arg_node,
                defaults=[],
            ),
            body=cast(
                "list[ast.stmt]",
                [
                    ast.Return(
                        value=ast.Call(
                            func=ast.Name(id=orig_name, ctx=self._load_ctx),
                            args=[self._self_name_load, self._starred_args],
                            keywords=[self._kwargs_keyword],
                        )
                    )
                ],
            ),
            decorator_list=cast("list[ast.expr]", []),
            returns=None,
        )

        # ClassName.__init__ = codeflash_capture(...)(_codeflash_patched_ClassName_init)
        assign_patched = ast.Assign(
            targets=[
                ast.Attribute(value=ast.Name(id=class_name, ctx=self._load_ctx), attr="__init__", ctx=self._store_ctx)
            ],
            value=ast.Call(func=decorator, args=[ast.Name(id=patched_name, ctx=self._load_ctx)], keywords=[]),
        )

        return [save_orig, patched_func, assign_patched]

    def _expr_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return self._expr_name(node.func)
        if isinstance(node, ast.Attribute):
            parent = self._expr_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None
