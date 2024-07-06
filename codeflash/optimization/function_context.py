from __future__ import annotations

import ast
import logging
import os
import re
from collections import defaultdict

import jedi
import tiktoken
from jedi.api.classes import Name

from codeflash.code_utils.code_extractor import get_code
from codeflash.code_utils.code_utils import module_name_from_file_path, path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize
from codeflash.models.models import FunctionSource


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if the given name belongs to the specified class."""
    if name.full_name and name.full_name.startswith(f"{name.module_name}.{class_name}."):
        return True
    return False


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname


def get_type_annotation_context(
    function: FunctionToOptimize,
    jedi_script: jedi.Script,
    project_root_path: str,
) -> list[FunctionSource]:
    function_name: str = function.function_name
    file_path: str = function.file_path
    with open(file_path, encoding="utf8") as file:
        file_contents: str = file.read()
    try:
        module: ast.Module = ast.parse(file_contents)
    except SyntaxError as e:
        logging.exception(f"get_type_annotation_context - Syntax error in code: {e}")
        return []
    sources: list[FunctionSource] = []
    ast_parents: list[FunctionParent] = []
    contextual_dunder_methods = set()

    def get_annotation_source(
        j_script: jedi.Script,
        name: str,
        node_parents: list[FunctionParent],
        line_no: int,
        col_no: str,
    ) -> None:
        try:
            definition: list[Name] = j_script.goto(
                line=line_no,
                column=col_no,
                follow_imports=True,
                follow_builtin_imports=False,
            )
        except Exception as ex:
            if hasattr(name, "full_name"):
                logging.exception(
                    f"Error while getting definition for {name.full_name}: {ex}",
                )
            else:
                logging.exception(f"Error while getting definition: {ex}")
            definition = []
        if definition:  # TODO can be multiple definitions
            definition_path = str(definition[0].module_path)
            # The definition is part of this project and not defined within the original function
            if (
                definition_path.startswith(project_root_path + os.sep)
                and definition[0].full_name
                and not path_belongs_to_site_packages(definition_path)
                and not belongs_to_function(definition[0], function_name)
            ):
                source_code = get_code(
                    [
                        FunctionToOptimize(
                            definition[0].name,
                            definition_path,
                            node_parents[:-1],
                        ),
                    ],
                )
                if source_code[0]:
                    sources.append(
                        FunctionSource(
                            fully_qualified_name=definition[0].full_name,
                            jedi_definition=definition[0],
                            source_code=source_code[0],
                            file_path=definition_path,
                            qualified_name=definition[0].full_name.removeprefix(
                                definition[0].module_name + ".",
                            ),
                            only_function_name=definition[0].name,
                        ),
                    )
                    contextual_dunder_methods.update(source_code[1])

    def visit_children(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module,
        node_parents: list[FunctionParent],
    ) -> None:
        child: ast.AST | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module
        for child in ast.iter_child_nodes(node):
            visit(child, node_parents)

    def visit_all_annotation_children(
        node: ast.Subscript | ast.Name | ast.BinOp,
        node_parents: list[FunctionParent],
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


def get_function_variables_definitions(
    function_to_optimize: FunctionToOptimize,
    project_root_path: str,
) -> tuple[list[FunctionSource], set[tuple[str, str]]]:
    function_name = function_to_optimize.function_name
    file_path = function_to_optimize.file_path
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
                if belongs_to_class(
                    ref,
                    function_to_optimize.parents[-1].name,
                ) and belongs_to_function(ref, function_name):
                    names.append(ref)
            elif belongs_to_function(ref, function_name):
                names.append(ref)

    for name in names:
        try:
            definitions: list[Name] = name.goto(
                follow_imports=True,
                follow_builtin_imports=False,
            )
        except Exception as e:
            try:
                logging.exception(f"Error while getting definition for {name.full_name}: {e}")
            except Exception as e:
                # name.full_name can also throw exceptions sometimes
                logging.exception(f"Error while getting definition: {e}")
            definitions = []
        if definitions:
            # TODO: there can be multiple definitions, see how to handle such cases
            definition = definitions[0]
            definition_path = str(definition.module_path)
            # The definition is part of this project and not defined within the original function
            if (
                definition_path.startswith(project_root_path + os.sep)
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
                    [
                        FunctionToOptimize(
                            function_name=definitions[0].name,
                            file_path=definition_path,
                            parents=parents,
                        ),
                    ],
                )
                if source_code[0]:
                    sources.append(
                        FunctionSource(
                            fully_qualified_name=definition.full_name,
                            jedi_definition=definition,
                            source_code=source_code[0],
                            file_path=definition_path,
                            qualified_name=definition.full_name.removeprefix(
                                definition.module_name + ".",
                            ),
                            only_function_name=definition.name,
                        ),
                    )
                    contextual_dunder_methods.update(source_code[1])
    annotation_sources, annotation_dunder_methods = get_type_annotation_context(
        function_to_optimize,
        script,
        project_root_path,
    )
    sources[:0] = annotation_sources  # prepend the annotation sources
    contextual_dunder_methods.update(annotation_dunder_methods)
    existing_fully_qualified_names = set()
    no_parent_sources: dict[str, dict[str, set[FunctionSource]]] = defaultdict(
        lambda: defaultdict(set),
    )
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
        source
        for k1 in no_parent_sources
        for k2 in no_parent_sources[k1]
        for source in no_parent_sources[k1][k2]
    ]
    return deduped_no_parent_sources + deduped_parent_sources, contextual_dunder_methods


MAX_PROMPT_TOKENS = 4096  # 128000  # gpt-4-128k


def get_constrained_function_context_and_helper_functions(
    function_to_optimize: FunctionToOptimize,
    project_root_path: str,
    code_to_optimize: str,
    max_tokens: int = MAX_PROMPT_TOKENS,
) -> tuple[str, list[FunctionSource], set[tuple[str, str]]]:
    # TODO: Not just do static analysis, but also find the datatypes of function arguments by running the existing
    #  unittests and inspecting the arguments to resolve the real definitions and dependencies.
    helper_functions, dunder_methods = get_function_variables_definitions(
        function_to_optimize,
        project_root_path,
    )
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
    logging.debug(f"ORIGINAL CODE TOKENS LENGTH: {context_len}")
    logging.debug(f"ALL DEPENDENCIES TOKENS LENGTH: {sum(helper_functions_tokens)}")
    for function_source, source_len in zip(helper_functions_sources, helper_functions_tokens):
        if context_len + source_len <= max_tokens:
            context_list.append(function_source)
            context_len += source_len
        else:
            break
    logging.debug(f"FINAL OPTIMIZATION CONTEXT TOKENS LENGTH: {context_len}")
    helper_code: str = "\n".join(context_list)
    return helper_code, helper_functions, dunder_methods
