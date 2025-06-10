from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import isort
import libcst as cst

if TYPE_CHECKING:
    from pathlib import Path

    from libcst import BaseStatement, ClassDef, FlattenSentinel, FunctionDef, RemovalSentinel

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class AddDecoratorTransformer(cst.CSTTransformer):
    def __init__(self, target_functions: set[tuple[str, str]]) -> None:
        super().__init__()
        self.target_functions = target_functions
        self.added_codeflash_trace = False
        self.class_name = ""
        self.function_name = ""
        self.decorator = cst.Decorator(decorator=cst.Name(value="codeflash_trace"))

    def leave_ClassDef(
        self, original_node: ClassDef, updated_node: ClassDef
    ) -> Union[BaseStatement, FlattenSentinel[BaseStatement], RemovalSentinel]:
        if self.class_name == original_node.name.value:
            self.class_name = ""  # Even if nested classes are not visited, this function is still called on them
        return updated_node

    def visit_ClassDef(self, node: ClassDef) -> Optional[bool]:
        if self.class_name:  # Don't go into nested class
            return False
        self.class_name = node.name.value  # noqa: RET503

    def visit_FunctionDef(self, node: FunctionDef) -> Optional[bool]:
        if self.function_name:  # Don't go into nested function
            return False
        self.function_name = node.name.value  # noqa: RET503

    def leave_FunctionDef(self, original_node: FunctionDef, updated_node: FunctionDef) -> FunctionDef:
        if self.function_name == original_node.name.value:
            self.function_name = ""
        if (self.class_name, original_node.name.value) in self.target_functions:
            # Add the new decorator after any existing decorators, so it gets executed first
            updated_decorators = [*list(updated_node.decorators), self.decorator]
            self.added_codeflash_trace = True
            return updated_node.with_changes(decorators=updated_decorators)

        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: ARG002
        # Create import statement for codeflash_trace
        if not self.added_codeflash_trace:
            return updated_node
        import_stmt = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Attribute(
                        value=cst.Attribute(value=cst.Name(value="codeflash"), attr=cst.Name(value="benchmarking")),
                        attr=cst.Name(value="codeflash_trace"),
                    ),
                    names=[cst.ImportAlias(name=cst.Name(value="codeflash_trace"))],
                )
            ]
        )

        # Insert at the beginning of the file. We'll use isort later to sort the imports.
        new_body = [import_stmt, *list(updated_node.body)]

        return updated_node.with_changes(body=new_body)


def add_codeflash_decorator_to_code(code: str, functions_to_optimize: list[FunctionToOptimize]) -> str:
    """Add codeflash_trace to a function.

    Args:
    ----
        code: The source code as a string
        functions_to_optimize: List of FunctionToOptimize instances containing function details

    Returns:
    -------
        The modified source code as a string

    """
    target_functions = set()
    for function_to_optimize in functions_to_optimize:
        class_name = ""
        if len(function_to_optimize.parents) == 1 and function_to_optimize.parents[0].type == "ClassDef":
            class_name = function_to_optimize.parents[0].name
        target_functions.add((class_name, function_to_optimize.function_name))

    transformer = AddDecoratorTransformer(target_functions=target_functions)

    module = cst.parse_module(code)
    modified_module = module.visit(transformer)
    return modified_module.code


def instrument_codeflash_trace_decorator(file_to_funcs_to_optimize: dict[Path, list[FunctionToOptimize]]) -> None:
    """Instrument codeflash_trace decorator to functions to optimize."""
    for file_path, functions_to_optimize in file_to_funcs_to_optimize.items():
        original_code = file_path.read_text(encoding="utf-8")
        new_code = add_codeflash_decorator_to_code(original_code, functions_to_optimize)
        # Modify the code
        modified_code = isort.code(code=new_code, float_to_top=True)

        # Write the modified code back to the file
        file_path.write_text(modified_code, encoding="utf-8")
