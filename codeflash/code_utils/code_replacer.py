from __future__ import annotations

from typing import IO

import libcst as cst
from libcst import FunctionDef

from codeflash.code_utils.code_extractor import add_needed_imports_from_module


class OptimFunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(
        self,
        function_name: str,
        class_name: str | None,
        contextual_functions: set[tuple[str, str]],
        preexisting_functions: list[str] | None = None,
    ) -> None:
        super().__init__()
        if preexisting_functions is None:
            preexisting_functions = []
        self.function_name = function_name
        self.class_name = class_name
        self.optim_body: FunctionDef | None = None
        self.optim_new_class_functions: list[cst.FunctionDef] = []
        self.optim_new_functions: list[cst.FunctionDef] = []
        self.preexisting_functions = preexisting_functions
        self.contextual_functions = contextual_functions.union(
            {(self.class_name, self.function_name)},
        )

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        parent2 = None
        try:
            if parent is not None and isinstance(parent, cst.Module):
                parent2 = self.get_metadata(cst.metadata.ParentNodeProvider, parent)
        except:
            pass
        if node.name.value == self.function_name:
            self.optim_body = node
        elif (
            self.preexisting_functions
            and node.name.value not in self.preexisting_functions
            and (
                isinstance(parent, cst.Module)
                or (parent2 is not None and not isinstance(parent2, cst.ClassDef))
            )
        ):
            self.optim_new_functions.append(node)

    def visit_ClassDef_body(self, node: cst.ClassDef) -> None:
        for child_node in node.body.body:
            if (
                isinstance(child_node, cst.FunctionDef)
                and (
                    node.name.value,
                    child_node.name.value,
                )
                not in self.contextual_functions
            ):
                self.optim_new_class_functions.append(child_node)


class OptimFunctionReplacer(cst.CSTTransformer):
    def __init__(
        self,
        function_name: str,
        optim_body: cst.FunctionDef,
        optim_new_class_functions: list[cst.FunctionDef],
        optim_new_functions: list[cst.FunctionDef],
        class_name: str | None = None,
    ) -> None:
        super().__init__()
        self.function_name = function_name
        self.optim_body = optim_body
        self.optim_new_class_functions = optim_new_class_functions
        self.optim_new_functions = optim_new_functions
        self.class_name = class_name
        self.depth: int = 0
        self.in_class: bool = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        if original_node.name.value == self.function_name and (
            self.depth == 0 or (self.depth == 1 and self.in_class)
        ):
            return self.optim_body
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.depth += 1
        if self.in_class:
            return False
        self.in_class = (self.depth == 1) and (node.name.value == self.class_name)
        return self.in_class

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        self.depth -= 1
        if self.in_class and (self.depth == 0) and (original_node.name.value == self.class_name):
            self.in_class = False
            return updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=(list(updated_node.body.body) + self.optim_new_class_functions),
                ),
            )
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        node = updated_node
        max_function_index = None
        class_index = None
        for index, _node in enumerate(node.body):
            if isinstance(_node, cst.FunctionDef):
                max_function_index = index
            if isinstance(_node, cst.ClassDef):
                class_index = index
        if max_function_index is not None:
            node = node.with_changes(
                body=(
                    *node.body[: max_function_index + 1],
                    *self.optim_new_functions,
                    *node.body[max_function_index + 1 :],
                ),
            )
        elif class_index is not None:
            node = node.with_changes(
                body=(
                    *node.body[: class_index + 1],
                    *self.optim_new_functions,
                    *node.body[class_index + 1 :],
                ),
            )
        else:
            node = node.with_changes(body=(*self.optim_new_functions, *node.body))
        return node


def replace_functions_in_file(
    source_code: str,
    original_function_names: list[str],
    optimized_code: str,
    preexisting_functions: list[str],
    contextual_functions: set[tuple[str, str]],
) -> str:
    parsed_function_names = []
    for original_function_name in original_function_names:
        if original_function_name.count(".") == 0:
            class_name, function_name = None, original_function_name
        elif original_function_name.count(".") == 1:
            class_name, function_name = original_function_name.split(".")
        else:
            raise ValueError(f"Don't know how to find {original_function_name} yet!")
        parsed_function_names.append((function_name, class_name))

    module = cst.metadata.MetadataWrapper(cst.parse_module(optimized_code))

    for i, (function_name, class_name) in enumerate(parsed_function_names):
        visitor = OptimFunctionCollector(
            function_name,
            class_name,
            contextual_functions,
            preexisting_functions,
        )
        module.visit(visitor)

        if visitor.optim_body is None and not preexisting_functions:
            continue
        if visitor.optim_body is None:
            raise ValueError(f"Did not find the function {function_name} in the optimized code")

        transformer = OptimFunctionReplacer(
            visitor.function_name,
            visitor.optim_body,
            visitor.optim_new_class_functions,
            visitor.optim_new_functions,
            class_name=class_name,
        )
        original_module = cst.parse_module(source_code)
        modified_tree = original_module.visit(transformer)
        source_code = modified_tree.code

    return source_code


def replace_functions_and_add_imports(
    source_code: str,
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: str,
    module_abspath: str,
    preexisting_functions: list[str],
    contextual_functions: set[tuple[str, str]],
    project_root_path: str,
) -> str:
    return add_needed_imports_from_module(
        optimized_code,
        replace_functions_in_file(
            source_code,
            function_names,
            optimized_code,
            preexisting_functions,
            contextual_functions,
        ),
        file_path_of_module_with_function_to_optimize,
        module_abspath,
        project_root_path,
    )


def replace_function_definitions_in_module(
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: str,
    module_abspath: str,
    preexisting_functions: list[str],
    contextual_functions: set[tuple[str, str]],
    project_root_path: str,
) -> bool:
    file: IO[str]
    with open(module_abspath, encoding="utf8") as file:
        source_code: str = file.read()
    new_code: str = replace_functions_and_add_imports(
        source_code,
        function_names,
        optimized_code,
        file_path_of_module_with_function_to_optimize,
        module_abspath,
        preexisting_functions,
        contextual_functions,
        project_root_path,
    )
    if new_code == source_code:
        return False
    with open(module_abspath, "w", encoding="utf8") as file:
        file.write(new_code)
    return True
