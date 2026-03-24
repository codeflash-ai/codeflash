"""AST/CST-based function discovery and inspection."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

import libcst as cst
from pydantic.dataclasses import dataclass

from codeflash_core.models import FunctionParent, FunctionToOptimize
from codeflash_python.discovery.filter_criteria import FunctionFilterCriteria

if TYPE_CHECKING:
    from pathlib import Path

    from libcst import CSTNode
    from libcst.metadata import CodeRange

logger = logging.getLogger("codeflash_python")


def is_class_defined_in_file(class_name: str, file_path: Path) -> bool:
    if not file_path.exists():
        return False
    with file_path.open(encoding="utf8") as file:
        source = file.read()
    tree = ast.parse(source)
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))


# =============================================================================
# CST-based function discovery
# =============================================================================


class ReturnStatementVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_return_statement: bool = False

    def visit_Return(self, node: cst.Return) -> None:
        self.has_return_statement = True


class FunctionVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider, cst.metadata.ParentNodeProvider)

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path: Path = file_path
        self.functions: list[FunctionToOptimize] = []

    @staticmethod
    def is_pytest_fixture(node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            dec = decorator.decorator
            if isinstance(dec, cst.Call):
                dec = dec.func
            if isinstance(dec, cst.Attribute) and dec.attr.value == "fixture":
                if isinstance(dec.value, cst.Name) and dec.value.value == "pytest":
                    return True
            if isinstance(dec, cst.Name) and dec.value == "fixture":
                return True
        return False

    @staticmethod
    def is_property(node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            dec = decorator.decorator
            if isinstance(dec, cst.Name) and dec.value in ("property", "cached_property"):
                return True
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        return_visitor: ReturnStatementVisitor = ReturnStatementVisitor()
        node.visit(return_visitor)
        if return_visitor.has_return_statement and not self.is_pytest_fixture(node) and not self.is_property(node):
            pos: CodeRange = self.get_metadata(cst.metadata.PositionProvider, node)
            parents: CSTNode | None = self.get_metadata(cst.metadata.ParentNodeProvider, node)
            ast_parents: list[FunctionParent] = []
            while parents is not None:
                if isinstance(parents, cst.FunctionDef):
                    # Skip nested functions — only discover top-level and class-level functions
                    return
                if isinstance(parents, cst.ClassDef):
                    ast_parents.append(FunctionParent(parents.name.value, parents.__class__.__name__))
                parents = self.get_metadata(cst.metadata.ParentNodeProvider, parents, default=None)
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name.value,
                    file_path=self.file_path,
                    parents=list(reversed(ast_parents)),
                    starting_line=pos.start.line,
                    ending_line=pos.end.line,
                    is_async=bool(node.asynchronous),
                    language="python",
                )
            )


def discover_functions(
    source: str, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
) -> list[FunctionToOptimize]:
    criteria = filter_criteria or FunctionFilterCriteria()

    tree = cst.parse_module(source)

    wrapper = cst.metadata.MetadataWrapper(tree)
    function_visitor = FunctionVisitor(file_path=file_path)
    wrapper.visit(function_visitor)

    functions: list[FunctionToOptimize] = []
    for func in function_visitor.functions:
        if not criteria.include_async and func.is_async:
            continue

        if not criteria.include_methods and func.parents:
            continue

        if criteria.require_return and func.starting_line is None:
            continue

        func_with_is_method = FunctionToOptimize(
            function_name=func.function_name,
            file_path=file_path,
            parents=func.parents,
            starting_line=func.starting_line,
            ending_line=func.ending_line,
            starting_col=func.starting_col,
            ending_col=func.ending_col,
            is_async=func.is_async,
            is_method=len(func.parents) > 0 and any(p.type == "ClassDef" for p in func.parents),
            language=func.language,
        )
        functions.append(func_with_is_method)

    return functions


@dataclass(frozen=True)
class FunctionProperties:
    is_top_level: bool
    has_args: bool | None
    is_staticmethod: bool | None
    is_classmethod: bool | None
    staticmethod_class_name: str | None


class TopLevelFunctionOrMethodVisitor(ast.NodeVisitor):
    def __init__(
        self, file_name: Path, function_or_method_name: str, class_name: str | None = None, line_no: int | None = None
    ) -> None:
        self.file_name = file_name
        self.class_name = class_name
        self.function_name = function_or_method_name
        self.is_top_level = False
        self.function_has_args: bool | None = None
        self.line_no = line_no
        self.is_staticmethod = False
        self.is_classmethod = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_name is None and node.name == self.function_name:
            self.is_top_level = True
            self.function_has_args = any(
                (
                    bool(node.args.args),
                    bool(node.args.kwonlyargs),
                    bool(node.args.kwarg),
                    bool(node.args.posonlyargs),
                    bool(node.args.vararg),
                )
            )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.class_name is None and node.name == self.function_name:
            self.is_top_level = True
            self.function_has_args = any(
                (
                    bool(node.args.args),
                    bool(node.args.kwonlyargs),
                    bool(node.args.kwarg),
                    bool(node.args.posonlyargs),
                    bool(node.args.vararg),
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # iterate over the class methods
        if node.name == self.class_name:
            for body_node in node.body:
                if (
                    isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and body_node.name == self.function_name
                ):
                    self.is_top_level = True
                    if any(
                        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                        for decorator in body_node.decorator_list
                    ):
                        self.is_classmethod = True
                    elif any(
                        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                        for decorator in body_node.decorator_list
                    ):
                        self.is_staticmethod = True
                    return
        elif self.line_no:
            # If we have line number info, check if class has a static method with the same line number
            # This way, if we don't have the class name, we can still find the static method
            for body_node in node.body:
                if (
                    isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and body_node.name == self.function_name
                    and body_node.lineno in {self.line_no, self.line_no + 1}
                    and any(
                        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                        for decorator in body_node.decorator_list
                    )
                ):
                    self.is_staticmethod = True
                    self.is_top_level = True
                    self.class_name = node.name
                    return

        return


def inspect_top_level_functions_or_methods(
    file_name: Path, function_or_method_name: str, class_name: str | None = None, line_no: int | None = None
) -> FunctionProperties | None:
    with file_name.open(encoding="utf8") as file:
        try:
            ast_module = ast.parse(file.read())
        except Exception:
            return None
    visitor = TopLevelFunctionOrMethodVisitor(
        file_name=file_name, function_or_method_name=function_or_method_name, class_name=class_name, line_no=line_no
    )
    visitor.visit(ast_module)
    staticmethod_class_name = visitor.class_name if visitor.is_staticmethod else None
    return FunctionProperties(
        is_top_level=visitor.is_top_level,
        has_args=visitor.function_has_args,
        is_staticmethod=visitor.is_staticmethod,
        is_classmethod=visitor.is_classmethod,
        staticmethod_class_name=staticmethod_class_name,
    )
