from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import jedi
import libcst as cst
from jedi.api.classes import Name

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeString, CodeStringsMarkdown
from codeflash.optimization.function_context import belongs_to_class, belongs_to_function


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, token_limit: int = 8000
) -> tuple[str, str]:
    function_name = function_to_optimize.function_name
    file_path = function_to_optimize.file_path
    script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
    file_path_to_qualified_function_names = defaultdict(set)
    file_path_to_qualified_function_names[file_path].add(function_to_optimize.qualified_name)
    read_only_code_markdown = CodeStringsMarkdown()
    final_read_writable_code = ""
    names = []
    for ref in script.get_names(all_scopes=True, definitions=False, references=True):
        if ref.full_name:
            if function_to_optimize.parents:
                # Check if the reference belongs to the specified class when FunctionParent is provided
                if belongs_to_class(ref, function_to_optimize.parents[-1].name) and belongs_to_function(
                    ref, function_name
                ):
                    names.append(ref)
            elif belongs_to_function(ref, function_name):
                names.append(ref)

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
                and definition.module_name != definition.full_name
            ):
                file_path_to_qualified_function_names[definition_path].add(
                    get_qualified_name(definition.module_name, definition.full_name)
                )
    for file_path, qualified_function_names in file_path_to_qualified_function_names.items():
        try:
            og_code_containing_helpers = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            read_writable_code = get_read_writable_code(og_code_containing_helpers, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-writable code: {e}")
            continue

        if read_writable_code:
            final_read_writable_code += f"\n{read_writable_code}"
            final_read_writable_code = add_needed_imports_from_module(
                src_module_code=og_code_containing_helpers,
                dst_module_code=final_read_writable_code,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions_fqn=qualified_function_names,
            )

        try:
            read_only_code = get_read_only_code(og_code_containing_helpers, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue

        read_only_code_with_imports = CodeString(
            code=add_needed_imports_from_module(
                src_module_code=og_code_containing_helpers,
                dst_module_code=read_only_code,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions_fqn=qualified_function_names,
            ),
            file_path=Path(file_path),
        )
        if read_only_code_with_imports.code:
            read_only_code_markdown.code_strings.append(read_only_code_with_imports)
    # final_read_writable_codestring = CodeString(code=final_read_writable_code)
    # tokenizer = tiktoken.encoding_for_model("gpt-4o")
    # final_read_writable_tokens = len(tokenizer.encode(final_read_writable_code))
    # if final_read_writable_tokens > token_limit:
    #     logger.debug(
    #         "Read writable code exceeded token limit, removing helper functions and only keeping function to optimize"
    #     )
    #     try:
    #         read_writable_code = get_read_writable_code(og_code_containing_helpers, qualified_function_names)
    #     except ValueError as e:
    #         logger.debug(f"Error while getting read-writable code: {e}")
    #         continue
    print(read_only_code_markdown.markdown)
    return CodeString(code=final_read_writable_code).code, read_only_code_markdown.markdown


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Returns the section attribute names (e.g., body, orelse) for a given node if they exist."""
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


def prune_cst_for_read_writable_code(
    node: cst.CSTNode, target_functions: set[str], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node and its children to keep nodes that lead to target functions."""
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        if qualified_name in target_functions:
            return node, True
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")
        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value
        new_body = []
        found_target = False

        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                qualified_name = f"{class_prefix}.{stmt.name.value}"
                if qualified_name in target_functions:
                    new_body.append(stmt)
                    found_target = True

        # If no target functions found, remove the class entirely
        if not new_body:
            return None, False

        return node.with_changes(body=cst.IndentedBlock(body=new_body)), found_target

    # For other nodes, we preserve them only if they contain target functions in their children.
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_read_writable_code(child, target_functions, prefix)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_writable_code(original_content, target_functions, prefix)
            if found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered

    if not found_any_target:
        return None, False

    return (node.with_changes(**updates) if updates else node), True


def get_read_writable_code(code: str, target_functions: set[str]) -> str:
    """Creates a read-writable code string by parsing and filtering the code to keep only
    target functions and the minimal surrounding structure.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_writable_code(module, target_functions)
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""


def prune_cst_for_read_only_code(
    node: cst.CSTNode, target_functions: set[str], prefix: str = "", remove_docstrings: bool = False
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for read-only context:

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        # If it's a target function, remove it but mark found_target = True
        if qualified_name in target_functions:
            return None, True
        # Keep only dunder methods
        if is_dunder_method(node.name.value):
            return node, False
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")

        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # First pass: detect if there is a target function in the class
        found_in_class = False
        new_body = []
        for stmt in node.body.body:
            filtered, found_target = prune_cst_for_read_only_code(
                stmt, target_functions, class_prefix, remove_docstrings
            )
            found_in_class |= found_target

            if isinstance(filtered, cst.FunctionDef):
                # Check if it's a target or non-dunder method
                qname = f"{class_prefix}.{filtered.name.value}"
                if qname in target_functions or not is_dunder_method(filtered.name.value):
                    continue
            if filtered:
                new_body.append(filtered)

        if not found_in_class:
            return None, False

        return node.with_changes(body=node.body.with_changes(body=new_body)) if new_body else None, True

    # For other nodes, keep the node and recursively filter children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_read_only_code(
                    child, target_functions, prefix, remove_docstrings
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_only_code(
                original_content, target_functions, prefix, remove_docstrings
            )
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered

    if updates:
        return (node.with_changes(**updates), found_any_target)

    return node, found_any_target


def get_read_only_code(code: str, target_functions: set[str], remove_docstrings: bool = False) -> str:
    """Creates a read-only version of the code by parsing and filtering the code to keep only
    class contextual information, and other module scoped variables.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_only_code(module, target_functions, remove_docstrings)
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""
