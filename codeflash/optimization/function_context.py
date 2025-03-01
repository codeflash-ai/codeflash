from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import jedi
import tiktoken
from jedi.api.classes import Name

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_extractor import get_code
from codeflash.code_utils.code_utils import (
    get_qualified_name,
    module_name_from_file_path,
    path_belongs_to_site_packages,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent, FunctionSource

if TYPE_CHECKING:
    from pathlib import Path


def belongs_to_method(name: Name, class_name: str, method_name: str) -> bool:
    """Check if the given name belongs to the specified method."""
    return belongs_to_function(name, method_name) and belongs_to_class(name, class_name)


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function."""
    if name.name == function_name:  # Handles function definition and recursive function calls
        return False
    if name := name.parent():
        if name.type == "function":
            return name.name == function_name
    return False


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if given jedi Name is a direct child of the specified class."""
    while name := name.parent():
        if name.type == "class":
            return name.name == class_name
    return False


def belongs_to_function_qualified(name: Name, qualified_function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function, matched by qualified function name."""
    try:
        if get_qualified_name(name.module_name, name.full_name) == qualified_function_name:
            # Handles function definition and recursive function calls
            return False
        if name := name.parent():
            if name.type == "function":
                return get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        return False
    except ValueError:
        return False


def get_type_annotation_context(
    function: FunctionToOptimize, jedi_script: jedi.Script, project_root_path: Path
) -> tuple[list[FunctionSource], set[tuple[str, str]]]:
    function_name: str = function.function_name
    file_path: Path = function.file_path
    file_contents: str = file_path.read_text(encoding="utf8")
    try:
        module: ast.Module = ast.parse(file_contents)
    except SyntaxError as e:
        logger.exception(f"get_type_annotation_context - Syntax error in code: {e}")
        return [], set()
    sources: list[FunctionSource] = []
    ast_parents: list[FunctionParent] = []
    contextual_dunder_methods = set()

    def get_annotation_source(
        j_script: jedi.Script, name: str, node_parents: list[FunctionParent], line_no: int, col_no: str
    ) -> None:
        try:
            definition: list[Name] = j_script.goto(
                line=line_no, column=col_no, follow_imports=True, follow_builtin_imports=False
            )
        except Exception as ex:
            if hasattr(name, "full_name"):
                logger.exception(f"Error while getting definition for {name.full_name}: {ex}")
            else:
                logger.exception(f"Error while getting definition: {ex}")
            definition = []
        if definition:  # TODO can be multiple definitions
            definition_path = definition[0].module_path

            # The definition is part of this project and not defined within the original function
            if (
                str(definition_path).startswith(str(project_root_path) + os.sep)
                and definition[0].full_name
                and not path_belongs_to_site_packages(definition_path)
                and not belongs_to_function(definition[0], function_name)
            ):
                source_code = get_code([FunctionToOptimize(definition[0].name, definition_path, node_parents[:-1])])
                if source_code[0]:
                    sources.append(
                        FunctionSource(
                            fully_qualified_name=definition[0].full_name,
                            jedi_definition=definition[0],
                            source_code=source_code[0],
                            file_path=definition_path,
                            qualified_name=definition[0].full_name.removeprefix(definition[0].module_name + "."),
                            only_function_name=definition[0].name,
                        )
                    )
                    contextual_dunder_methods.update(source_code[1])

    def visit_children(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module, node_parents: list[FunctionParent]
    ) -> None:
        child: ast.AST | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module
        for child in ast.iter_child_nodes(node):
            visit(child, node_parents)

    def visit_all_annotation_children(
        node: ast.Subscript | ast.Name | ast.BinOp, node_parents: list[FunctionParent]
    ) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            visit_all_annotation_children(node.left, node_parents)
            visit_all_annotation_children(node.right, node_parents)
        if isinstance(node, ast.Name) and hasattr(node, "id"):
            name: str = node.id
            line_no: int = node.lineno
            col_no: int = node.col_offset
            get_annotation_source(jedi_script, name, node_parents, line_no, col_no)
        if isinstance(node, ast.Subscript):
            if hasattr(node, "slice"):
                if isinstance(node.slice, ast.Subscript):
                    visit_all_annotation_children(node.slice, node_parents)
                elif isinstance(node.slice, ast.Tuple):
                    for elt in node.slice.elts:
                        if isinstance(elt, (ast.Name, ast.Subscript)):
                            visit_all_annotation_children(elt, node_parents)
                elif isinstance(node.slice, ast.Name):
                    visit_all_annotation_children(node.slice, node_parents)
            if hasattr(node, "value"):
                visit_all_annotation_children(node.value, node_parents)

    def visit(
        node: ast.AST | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module,
        node_parents: list[FunctionParent],
    ) -> None:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name and node_parents == function.parents:
                    arg: ast.arg
                    for arg in node.args.args:
                        if arg.annotation:
                            visit_all_annotation_children(arg.annotation, node_parents)
                    if node.returns:
                        visit_all_annotation_children(node.returns, node_parents)

            if not isinstance(node, ast.Module):
                node_parents.append(FunctionParent(node.name, type(node).__name__))
            visit_children(node, node_parents)
            if not isinstance(node, ast.Module):
                node_parents.pop()

    visit(module, ast_parents)

    return sources, contextual_dunder_methods


def _find_class_references_in_ast(
    file_contents: str, function_name: str, script: jedi.Script, function_parents: Optional[list[FunctionParent]] = None
) -> list[Name]:
    class_refs: list[Name] = []

    try:
        tree: ast.Module = ast.parse(file_contents)
    except SyntaxError:
        return class_refs

    target_function_found: list[bool] = [False]

    class ClassReferenceVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_target_function: bool = False
            self.current_class: Optional[str] = None
            self.parent_stack: list[tuple[str, str]] = []

        def _is_target_function(self, node: ast.FunctionDef) -> bool:
            if not function_parents:
                return node.name == function_name
            if node.name != function_name or len(self.parent_stack) < len(function_parents):
                return False
            for i, parent in enumerate(reversed(function_parents)):
                idx: int = len(self.parent_stack) - 1 - i
                if idx < 0 or self.parent_stack[idx] != (parent.name, parent.type):
                    return False
            return True

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._enter_scope(node, "ClassDef")
            self._process_bases(node)
            self.generic_visit(node)
            self._exit_scope()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._enter_scope(node, "FunctionDef")
            if self._is_target_function(node):
                self.in_target_function = True
                target_function_found[0] = True
            self.generic_visit(node)
            self.in_target_function = False
            self._exit_scope()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

        def visit_Call(self, node: ast.Call) -> None:
            if self.in_target_function or target_function_found[0]:
                self._process_call(node)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if self.in_target_function or target_function_found[0]:
                self._process_except_handler(node)
            self.generic_visit(node)

        def _enter_scope(self, node: ast.AST, node_type: str) -> None:
            self.parent_stack.append((node.name, node_type))
            if node_type == "ClassDef":
                self.current_class = node.name

        def _exit_scope(self) -> None:
            self.parent_stack.pop()
            if self.parent_stack and self.parent_stack[-1][1] == "ClassDef":
                self.current_class = self.parent_stack[-1][0]
            else:
                self.current_class = None

        def _process_bases(self, node: ast.ClassDef) -> None:
            for base in node.bases:
                if isinstance(base, ast.Name):
                    self._process_class_reference(base.id, node.lineno, base.col_offset)
                elif isinstance(base, ast.Attribute):
                    full_name = self._get_attribute_name(base)
                    if full_name:
                        self._process_class_reference(full_name, node.lineno, base.col_offset)

        def _process_call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in {"isinstance", "issubclass"} and len(node.args) == 2:
                self._process_isinstance_issubclass(node)

        def _process_isinstance_issubclass(self, node: ast.Call) -> None:
            class_arg = node.args[1]
            if isinstance(class_arg, ast.Name):
                self._process_class_reference(class_arg.id, class_arg.lineno, class_arg.col_offset)
            elif isinstance(class_arg, ast.Tuple):
                for elt in class_arg.elts:
                    if isinstance(elt, (ast.Name, ast.Attribute)):
                        full_name = self._get_attribute_name(elt) if isinstance(elt, ast.Attribute) else elt.id
                        if full_name:
                            self._process_class_reference(full_name, elt.lineno, elt.col_offset)
            elif isinstance(class_arg, ast.Attribute):
                full_name = self._get_attribute_name(class_arg)
                if full_name:
                    self._process_class_reference(full_name, class_arg.lineno, class_arg.col_offset)

        def _process_except_handler(self, node: ast.ExceptHandler) -> None:
            if isinstance(node.type, ast.Name):
                self._process_class_reference(node.type.id, node.lineno, node.col_offset)
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, (ast.Name, ast.Attribute)):
                        full_name = self._get_attribute_name(elt) if isinstance(elt, ast.Attribute) else elt.id
                        if full_name:
                            self._process_class_reference(full_name, elt.lineno, elt.col_offset)
            elif isinstance(node.type, ast.Attribute):
                full_name = self._get_attribute_name(node.type)
                if full_name:
                    self._process_class_reference(full_name, node.lineno, node.col_offset)

        def _get_attribute_name(self, node: ast.Attribute) -> Optional[str]:
            if isinstance(node, ast.Name):
                return node.id
            base = self._get_attribute_name(node.value)
            return f"{base}.{node.attr}" if base else None

        def _process_class_reference(self, class_name: str, line: int, col: int) -> None:
            try:
                definitions: list[Name] = script.goto(line=line, column=col, follow_imports=True)
                for definition in definitions:
                    if definition.type == "class":
                        class_refs.append(definition)
            except Exception as e:
                logger.exception(f"Error looking up class reference {class_name}: {e}")

    visitor = ClassReferenceVisitor()
    visitor.visit(tree)

    return class_refs


def get_function_variables_definitions(
    function_to_optimize: FunctionToOptimize, project_root_path: Path
) -> tuple[list[FunctionSource], set[tuple[str, str]]]:
    function_name = function_to_optimize.function_name
    file_path = function_to_optimize.file_path
    file_contents = file_path.read_text(encoding="utf8")
    script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
    sources: list[FunctionSource] = []
    contextual_dunder_methods = set()
    # TODO: The function name condition can be stricter so that it does not clash with other class names etc.
    # TODO: The function could have been imported as some other name,
    #  we should be checking for the translation as well. Also check for the original function name.
    names = []
    for ref in script.get_names(all_scopes=True, definitions=False, references=True):
        if ref.full_name:
            if function_to_optimize.parents:
                # Check if the reference belongs to the specified class when FunctionParent is provided
                if belongs_to_method(ref, function_to_optimize.parents[-1].name, function_name):
                    names.append(ref)
            elif belongs_to_function(ref, function_name):
                names.append(ref)

    class_refs = _find_class_references_in_ast(file_contents, function_name, script, function_to_optimize.parents)
    for class_ref in class_refs:
        try:
            definition_path = class_ref.module_path

            if (
                str(definition_path).startswith(str(project_root_path) + os.sep)
                and not path_belongs_to_site_packages(definition_path)
                and class_ref.full_name
            ):
                module_name = module_name_from_file_path(definition_path, project_root_path)
                m = re.match(rf"{module_name}\.(.*)\.{class_ref.name}", class_ref.full_name)
                parents = []
                if m:
                    parents = [FunctionParent(m.group(1), "ClassDef")]

                source_code = get_code(
                    [FunctionToOptimize(function_name=class_ref.name, file_path=definition_path, parents=parents)]
                )

                if source_code[0]:
                    sources.append(
                        FunctionSource(
                            fully_qualified_name=class_ref.full_name,
                            jedi_definition=class_ref,
                            source_code=source_code[0],
                            file_path=definition_path,
                            qualified_name=class_ref.full_name.removeprefix(class_ref.module_name + "."),
                            only_function_name=class_ref.name,
                        )
                    )
                    contextual_dunder_methods.update(source_code[1])
        except Exception as e:
            logger.exception(f"Error processing class reference {class_ref.name}: {e}")

    for name in names:
        try:
            definitions: list[Name] = name.goto(follow_imports=True, follow_builtin_imports=False)
        except Exception as e:
            try:
                logger.exception(f"Error while getting definition for {name.full_name}: {e}")
            except Exception as e:
                # name.full_name can also throw exceptions sometimes
                logger.exception(f"Error while getting definition: {e}")
            definitions = []
        if definitions:
            # TODO: there can be multiple definitions, see how to handle such cases
            definition = definitions[0]
            definition_path = definition.module_path

            # The definition is part of this project and not defined within the original function
            if (
                str(definition_path).startswith(str(project_root_path) + os.sep)
                and not path_belongs_to_site_packages(definition_path)
                and definition.full_name
                and not belongs_to_function(definition, function_name)
            ):
                module_name = module_name_from_file_path(definition_path, project_root_path)
                m = re.match(rf"{module_name}\.(.*)\.{definitions[0].name}", definitions[0].full_name)
                parents = []
                if m:
                    parents = [FunctionParent(m.group(1), "ClassDef")]

                source_code = get_code(
                    [FunctionToOptimize(function_name=definitions[0].name, file_path=definition_path, parents=parents)]
                )
                if source_code[0]:
                    sources.append(
                        FunctionSource(
                            fully_qualified_name=definition.full_name,
                            jedi_definition=definition,
                            source_code=source_code[0],
                            file_path=definition_path,
                            qualified_name=definition.full_name.removeprefix(definition.module_name + "."),
                            only_function_name=definition.name,
                        )
                    )
                    contextual_dunder_methods.update(source_code[1])
    annotation_sources, annotation_dunder_methods = get_type_annotation_context(
        function_to_optimize, script, project_root_path
    )
    sources[:0] = annotation_sources  # prepend the annotation sources
    contextual_dunder_methods.update(annotation_dunder_methods)
    existing_fully_qualified_names = set()
    no_parent_sources: dict[Path, dict[str, set[FunctionSource]]] = defaultdict(lambda: defaultdict(set))
    parent_sources = set()
    for source in sources:
        if (fully_qualified_name := source.fully_qualified_name) not in existing_fully_qualified_names:
            if not source.qualified_name.count("."):
                no_parent_sources[source.file_path][source.qualified_name].add(source)
            else:
                parent_sources.add(source)
            existing_fully_qualified_names.add(fully_qualified_name)
    deduped_parent_sources = [
        source
        for source in parent_sources
        if source.file_path not in no_parent_sources
        or source.qualified_name.rpartition(".")[0] not in no_parent_sources[source.file_path]
    ]
    deduped_no_parent_sources = [
        source for k1 in no_parent_sources for k2 in no_parent_sources[k1] for source in no_parent_sources[k1][k2]
    ]
    return deduped_no_parent_sources + deduped_parent_sources, contextual_dunder_methods


MAX_PROMPT_TOKENS = 4096  # 128000  # gpt-4-128k


def get_constrained_function_context_and_helper_functions(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    code_to_optimize: str,
    max_tokens: int = MAX_PROMPT_TOKENS,
) -> tuple[str, list[FunctionSource], set[tuple[str, str]]]:
    helper_functions, dunder_methods = get_function_variables_definitions(function_to_optimize, project_root_path)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    code_to_optimize_tokens = tokenizer.encode(code_to_optimize)

    if not function_to_optimize.parents:
        helper_functions_sources = [function.source_code for function in helper_functions]
    else:
        helper_functions_sources = [
            function.source_code
            for function in helper_functions
            if not function.qualified_name.count(".")
            or function.qualified_name.split(".")[0] != function_to_optimize.parents[0].name
        ]
    helper_functions_tokens = [len(tokenizer.encode(function)) for function in helper_functions_sources]

    context_list = []
    context_len = len(code_to_optimize_tokens)
    logger.debug(f"ORIGINAL CODE TOKENS LENGTH: {context_len}")
    logger.debug(f"ALL DEPENDENCIES TOKENS LENGTH: {sum(helper_functions_tokens)}")
    for function_source, source_len in zip(helper_functions_sources, helper_functions_tokens):
        if context_len + source_len <= max_tokens:
            context_list.append(function_source)
            context_len += source_len
        else:
            break
    logger.debug(f"FINAL OPTIMIZATION CONTEXT TOKENS LENGTH: {context_len}")
    helper_code: str = "\n".join(context_list)
    return helper_code, helper_functions, dunder_methods
