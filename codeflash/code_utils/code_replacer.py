from typing import List, Union, Optional

import libcst as cst
from libcst import SimpleStatementLine, FunctionDef


class OptimFunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(self, function_name: str, preexisting_functions: Optional[List[str]] = None):
        super().__init__()
        if preexisting_functions is None:
            preexisting_functions = []
        self.function_name = function_name
        self.optim_body: Union[FunctionDef, None]s = None
        self.optim_new_class_functions = []
        self.optim_new_functions = []
        self.optim_imports = []
        self.preexisting_functions = preexisting_functions

    def visit_FunctionDef(self, node: cst.FunctionDef):
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        parent2 = None
        try:
            if parent is not None and isinstance(parent, cst.Module):
                parent2 = self.get_metadata(cst.metadata.ParentNodeProvider, parent)
        except:
            pass
        if node.name.value == self.function_name:
            self.optim_body = node
        elif node.name.value not in self.preexisting_functions and (
            isinstance(parent, cst.Module)
            or (parent2 is not None and not isinstance(parent2, cst.ClassDef))
        ):
            self.optim_new_functions.append(node)

    def visit_ClassDef_body(self, node: cst.ClassDef) -> None:
        for class_node in node.body.body:
            if isinstance(class_node, cst.FunctionDef) and class_node.name.value not in [
                "__init__",
                self.function_name,
            ]:
                self.optim_new_class_functions.append(class_node)

    def leave_SimpleStatementLine(self, original_node: "SimpleStatementLine") -> None:
        if isinstance(original_node.body[0], cst.Import):
            self.optim_imports.append(original_node)
        elif isinstance(original_node.body[0], cst.ImportFrom):
            self.optim_imports.append(original_node)


class OptimFunctionReplacer(cst.CSTTransformer):
    def __init__(
        self,
        function_name: str,
        optim_body: cst.FunctionDef,
        optim_new_class_functions: List[cst.FunctionDef],
        optim_imports: List[Union[cst.Import, cst.ImportFrom]],
        optim_new_functions,
        class_name=None,
    ):
        super().__init__()
        self.function_name = function_name
        self.optim_body = optim_body
        self.optim_new_class_functions = optim_new_class_functions
        self.optim_new_imports = optim_imports
        self.optim_new_functions = optim_new_functions
        self.class_name = class_name

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node.name.value == self.function_name:
            return self.optim_body
        return updated_node

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if self.class_name is not None and original_node.name.value == self.class_name:
            return updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=(list(updated_node.body.body) + self.optim_new_class_functions),
                )
            )
        else:
            return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if len(self.optim_new_imports) == 0:
            node = updated_node
        else:
            node = updated_node.with_changes(body=(*self.optim_new_imports, *updated_node.body))
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
                )
            )
        elif class_index is not None:
            node = node.with_changes(
                body=(
                    *node.body[: class_index + 1],
                    *self.optim_new_functions,
                    *node.body[class_index + 1 :],
                )
            )
        else:
            node = node.with_changes(body=(*self.optim_new_functions, *node.body))
        return node

    # TODO: Implement the logic to not duplicate imports. This is supported by libcst, figure out how to use it.
    # def leave_Module(self, original_node: "Module", updated_node: "Module") -> "Module":
    #     print(self.context)
    #     for import_node in self.optim_new_imports:
    #         # updated_node = updated_node.with_changes(
    #         #     body=(*updated_node.body, import_node)
    #         # )
    #         if isinstance(import_node, cst.Import):
    #             #print(import_node.names)
    #             for name in import_node.names:
    #                 print(name)
    #                 print(name.asname.name.value)
    #                 asname = name.asname.name.value if name.asname else None
    #                 AddImportsVisitor.add_needed_import(self.context, name.name.value, asname=asname)
    #         if isinstance(import_node, cst.ImportFrom):
    #             print(import_node)
    #             for name in import_node.names:
    #                 asname = name.asname.name.value if name.asname else None
    #                 AddImportsVisitor.add_needed_import(self.context, module =import_node.module.value,  obj=name.name.value, asname=asname)
    #     #print(updated_node)


def replace_function_in_file(
    original_file_path: str,
    original_function_name: str,
    optimized_function: str,
    preexisting_functions: list[str],
) -> str:
    with open(original_file_path, "r") as file:
        source_code = file.read()
    class_name = None
    if original_function_name.count(".") == 0:
        function_name = original_function_name
    elif original_function_name.count(".") == 1:
        class_name, function_name = original_function_name.split(".")
    else:
        raise ValueError(f"Don't know how to find {original_function_name} yet!")
    visitor = OptimFunctionCollector(function_name, preexisting_functions)
    module = cst.metadata.MetadataWrapper(cst.parse_module(optimized_function))
    visited = module.visit(visitor)

    if visitor.optim_body is None:
        raise ValueError(f"Did not find the function {function_name} in the optimized code")
    # TODO: If a dependency has been optimized and that dependency is imported in the file,
    # then we need to update the original file where the dependency is imported from
    transformer = OptimFunctionReplacer(
        visitor.function_name,
        visitor.optim_body,
        visitor.optim_new_class_functions,
        visitor.optim_imports,
        visitor.optim_new_functions,
        class_name=class_name,
    )
    original_module = cst.parse_module(source_code)
    modified_tree = original_module.visit(transformer)
    return modified_tree.code
